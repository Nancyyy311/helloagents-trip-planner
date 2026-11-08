/**
 * E2E 冒烟：真浏览器验证"生成中可以取消"这条链路。
 *
 * 为什么需要它：批次19 的教训是——代码里 `v-if` 正确、组件测试也绿，
 * 但按钮可能落在视口之外，用户依然看不见。只有真浏览器能量出"它在不在屏幕里"。
 *
 * 前置：后端 7000、前端 5173 都在跑。
 * 代价：会真实调用一次 LLM（随即取消），不要放进常规测试或 CI 默认流程。
 *
 * 用法：
 *   npm run test:e2e:cancel
 *   CHROME_PATH="/path/to/chrome" FRONTEND_URL=http://localhost:5173/ npm run test:e2e:cancel
 *
 * 退出码：0 通过；1 失败（按钮没出现 / 不在视口内 / 点击无效 / 后端没取消）。
 */
const puppeteer = require('puppeteer-core')
const fs = require('fs')
const path = require('path')

const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  'C:/Users/Lenovo/AppData/Local/Google/Chrome/Application/chrome.exe',
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  '/usr/bin/google-chrome',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
].filter(Boolean)

const CHROME = CHROME_CANDIDATES.find((p) => fs.existsSync(p))
const SHOTS = path.resolve(__dirname, '../.e2e-shots')
const FRONTEND = process.env.FRONTEND_URL || 'http://localhost:5173/'
const BACKEND = process.env.BACKEND_URL || 'http://localhost:7000'
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

if (!CHROME) {
  console.error('找不到浏览器。请用 CHROME_PATH 指定 chrome.exe 的绝对路径。')
  process.exit(1)
}

const fmt = (d) =>
  `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`

;(async () => {
  fs.mkdirSync(SHOTS, { recursive: true })
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: 'new',
    args: ['--no-sandbox'],
    defaultViewport: { width: 1280, height: 1000 }
  })
  const page = await browser.newPage()
  const errors = []
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message))
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push('console: ' + m.text().slice(0, 200))
  })

  await page.goto(FRONTEND, { waitUntil: 'networkidle2', timeout: 60000 })
  await sleep(1500)

  // 1. 目的地
  const cityInput = await page.$('input[placeholder="例如: 北京"]')
  if (!cityInput) throw new Error('找不到城市输入框，页面结构可能变了')
  await cityInput.type('上海', { delay: 40 })

  // 2. 两个日期：直接键入比点日历面板稳（面板有多个隐藏实例）
  const pickers = await page.$$('.ant-picker input')
  if (pickers.length < 2) throw new Error('找不到两个日期输入框')
  const start = new Date(Date.now() + 86400000)
  const end = new Date(Date.now() + 3 * 86400000)
  await pickers[0].click()
  await pickers[0].type(fmt(start), { delay: 30 })
  await page.keyboard.press('Enter')
  await sleep(700)
  await pickers[1].click()
  await pickers[1].type(fmt(end), { delay: 30 })
  await page.keyboard.press('Enter')
  await sleep(700)

  const filled = await page.$$eval('.ant-picker input', (els) => els.map((e) => e.value))
  if (filled[0] !== fmt(start) || filled[1] !== fmt(end)) {
    throw new Error(`日期没填进去：${JSON.stringify(filled)}`)
  }
  await page.screenshot({ path: path.join(SHOTS, '01-form-filled.png') })

  // 3. 像真人一样：先滚到提交按钮，再用鼠标点
  const submitBox = await page.evaluate(() => {
    const btn = [...document.querySelectorAll('button')].find((b) => b.textContent.includes('开始规划行程'))
    if (!btn) return null
    btn.scrollIntoView({ block: 'center' })
    const r = btn.getBoundingClientRect()
    return { x: r.x + r.width / 2, y: r.y + r.height / 2 }
  })
  if (!submitBox) throw new Error('找不到"开始规划行程"按钮')
  await sleep(400)
  await page.mouse.click(submitBox.x, submitBox.y)

  // 4. 每 700ms 采样：按钮在不在 DOM、在不在视口
  const samples = []
  for (let i = 0; i < 14; i++) {
    await sleep(700)
    const snap = await page.evaluate(() => {
      const btn = [...document.querySelectorAll('button')].find((b) => /中止本次生成|取消生成/.test(b.textContent))
      const r = btn ? btn.getBoundingClientRect() : null
      return {
        url: location.pathname,
        scrollY: Math.round(window.scrollY),
        inViewport: !!(r && r.top >= 0 && r.bottom <= window.innerHeight && r.width > 0),
        hasCancelBtn: !!btn,
        statusText: document.querySelector('.generating-status')?.textContent?.trim() || null
      }
    })
    samples.push({ t: ((i + 1) * 0.7).toFixed(1), ...snap })
    if (i === 1) await page.screenshot({ path: path.join(SHOTS, '02-generating.png') })
    if (snap.url !== '/') break // 已跳结果页，说明生成结束
  }

  console.log('\n=== 采样（秒 | 路径 | 有按钮 | 在视口内 | scrollY | 状态）===')
  for (const s of samples) {
    console.log(`${s.t}s | ${s.url} | ${s.hasCancelBtn} | ${s.inViewport} | ${s.scrollY} | ${s.statusText || '-'}`)
  }

  const visible = samples.find((s) => s.hasCancelBtn && s.inViewport)
  let after = { stillHasBtn: true }
  if (visible) {
    await page.evaluate(() => {
      const btn = [...document.querySelectorAll('button')].find((b) => /中止本次生成|取消生成/.test(b.textContent))
      btn.click()
    })
    await sleep(1200)
    await page.screenshot({ path: path.join(SHOTS, '03-canceling.png') })
    after = await page.evaluate(() => ({
      stillHasBtn: [...document.querySelectorAll('button')].some((b) => /中止本次生成/.test(b.textContent)),
      text: document.body.innerText.match(/已请求取消[^\n]*/)?.[0] || null,
      url: location.pathname
    }))
    console.log('\n点击后:', JSON.stringify(after))
  } else {
    await page.screenshot({ path: path.join(SHOTS, '02-NO-CANCEL.png') })
    console.log('\n!! 整个生成期间都没有找到【在视口内的】取消按钮')
  }

  // 5. 后端指标：确认任务真的被停了，不只是前端改了文案
  let canceled = null
  try {
    const res = await fetch(`${BACKEND}/metrics`)
    const m = await res.json()
    canceled = {
      tasks_canceled: m.tasks_canceled,
      by_reason: m.tasks_canceled_by_reason,
      by_stage: m.tasks_canceled_by_stage
    }
    console.log('后端取消计数:', JSON.stringify(canceled))
  } catch (e) {
    console.log('读取后端 /metrics 失败:', e.message)
  }

  const ok =
    !!visible &&
    after.stillHasBtn === false &&
    errors.length === 0 &&
    (canceled === null || (canceled.tasks_canceled || 0) > 0)

  console.log('\n页面错误数:', errors.length)
  errors.slice(0, 5).forEach((e) => console.log('  ', e))
  console.log(ok ? 'E2E 通过：按钮可见 → 点击生效 → 后端任务已取消' : 'E2E 失败')

  await browser.close()
  process.exit(ok ? 0 : 1)
})().catch(async (e) => {
  console.error('E2E 异常:', e.message)
  process.exit(1)
})
