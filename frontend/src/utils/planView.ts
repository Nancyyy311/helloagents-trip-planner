/**
 * 行程结果页的纯展示逻辑。
 *
 * 为什么单独抽出来：这些函数原来写在 Result.vue 里（2051 行的单文件组件），
 * 逻辑本身与 Vue 状态无关，但因为闭包在组件内，只能靠"打开浏览器点一遍"验证。
 * 抽成纯函数后可以脱离浏览器单测——尤其是数组越界这类只在边界条件下出错的写法。
 *
 * 约定：本文件不允许 import 任何 Vue / 浏览器 API 以外的模块，
 * 保证在 Node 环境下可直接 import 测试。
 */

export type MealType = 'breakfast' | 'lunch' | 'dinner' | 'snack' | string
export type DayPeriod = 'day' | 'night'
export type MoveDirection = 'up' | 'down'

const MEAL_LABELS: Record<string, string> = {
  breakfast: '早餐',
  lunch: '午餐',
  dinner: '晚餐',
  snack: '小吃'
}

/** 餐次英文枚举转中文；未知枚举原样返回，避免页面出现空白。 */
export function getMealLabel(type: MealType): string {
  return MEAL_LABELS[type] || type
}

/** 天气字段可能是 null / 数字 / 带空白的字符串，统一成可安全 includes 的字符串。 */
export function normalizeWeatherText(weather: unknown): string {
  return String(weather || '').trim()
}

/**
 * 天气文案转图标。
 * 判断顺序有意义：先判"雷"再判"雨"，否则"雷阵雨"会命中 rain 分支丢掉雷电。
 */
export function getWeatherIcon(weather: unknown, period: DayPeriod): string {
  const text = normalizeWeatherText(weather)
  if (!text || text.includes('未知') || text.includes('暂无')) return '—'
  if (text.includes('雷')) return '⛈️'
  if (text.includes('暴雨') || text.includes('大雨')) return '🌧️'
  if (text.includes('雨')) return '🌦️'
  if (text.includes('雪')) return '❄️'
  if (text.includes('冰雹')) return '🌨️'
  if (text.includes('雾') || text.includes('霾')) return '🌫️'
  if (text.includes('沙') || text.includes('尘')) return '🌪️'
  if (text.includes('阴')) return '☁️'
  if (text.includes('云')) return period === 'night' ? '☁️' : '⛅'
  if (text.includes('晴')) return period === 'night' ? '🌙' : '☀️'
  return period === 'night' ? '🌙' : '🌤️'
}

/** 天气卡片配色类名。白天+夜间合并判断，因为一张卡片同时显示两段天气。 */
export function getWeatherCardClass(dayWeather: unknown, nightWeather: unknown): string {
  const text = `${normalizeWeatherText(dayWeather)} ${normalizeWeatherText(nightWeather)}`
  if (!text.trim() || text.includes('未知') || text.includes('暂无')) return 'weather-card-unknown'
  if (text.includes('雨') || text.includes('雷')) return 'weather-card-rain'
  if (text.includes('雪')) return 'weather-card-snow'
  if (text.includes('雾') || text.includes('霾') || text.includes('沙') || text.includes('尘')) return 'weather-card-haze'
  if (text.includes('阴') || text.includes('云')) return 'weather-card-cloud'
  return 'weather-card-sun'
}

/**
 * 行程里是否含"远期天气占位"。
 * 高德预报只覆盖今天起 4 天，超出部分后端会显式写占位文案；
 * 页面据此在天气区块顶部提示原因，避免用户以为是我们没查到。
 */
export function hasUnknownWeather(rows: unknown): boolean {
  if (!Array.isArray(rows)) return false
  return rows.some(row => {
    const item = (row || {}) as Record<string, unknown>
    return normalizeWeatherText(item.day_weather).includes('远期')
  })
}

/**
 * 温度展示：只有拿到真实数字才拼 °C。
 * 远期天气后端会给 "未知"，原写法 `{{ day_temp }}°C` 会渲染成"未知°C"，
 * 看起来像"温度是未知摄氏度"——比直接留空更误导。
 */
export function formatTemperature(value: unknown): string {
  if (typeof value === 'number' && Number.isFinite(value)) return `${value}°C`
  const text = normalizeWeatherText(value)
  if (!text || text.includes('未知') || text.includes('暂无')) return '—'
  if (/^-?\d+(\.\d+)?$/.test(text)) return `${text}°C`
  return text
}

/** 风向风力："东南 1-3级"；两段都缺时返回空串，模板据此整行不渲染。 */
export function formatWind(direction: unknown, power: unknown): string {
  const dir = normalizeWeatherText(direction)
  const pw = normalizeWeatherText(power)
  const usable = (text: string) => !!text && !text.includes('未知') && !text.includes('暂无')
  return [usable(dir) ? dir : '', usable(pw) ? pw : ''].filter(Boolean).join(' ')
}

/**
 * 判断某条早餐是否只是"住宿自带的早餐"。
 * 这类条目不值得去高德回填坐标（酒店本身已经有位置），
 * 原实现漏判会让结果页多发一批无谓的 POI 搜索请求。
 */
export function isLodgingBreakfastMeal(type: MealType, name: string): boolean {
  if (type !== 'breakfast') return false
  const safeName = String(name || '')
  return ['酒店早餐', '民宿早餐', '客栈早餐', '住宿早餐'].some(keyword => safeName.includes(keyword))
}

const PLACEHOLDER_COLORS = [
  { start: '#1677ff', end: '#69b1ff' },
  { start: '#0f766e', end: '#5eead4' },
  { start: '#475569', end: '#94a3b8' },
  { start: '#2563eb', end: '#93c5fd' },
  { start: '#334155', end: '#cbd5e1' }
]

/**
 * 无真实照片时生成占位图（内联 SVG data URI）。
 * 用 index 轮换取色，保证同一页面内的占位卡不撞色；负索引与超大索引归一化到合法范围。
 * 注意：这里必须用 base64 而不是 encodeURIComponent，中文景点名在部分浏览器内联 SVG 里会乱码。
 */
export function buildAttractionPlaceholder(name: string, index: number): string {
  const safeName = String(name || '')
  const size = PLACEHOLDER_COLORS.length
  const normalizedIndex = ((Math.trunc(index) % size) + size) % size
  const { start, end } = PLACEHOLDER_COLORS[normalizedIndex]

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300">
    <defs>
      <linearGradient id="grad${normalizedIndex}" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:${start};stop-opacity:1" />
        <stop offset="100%" style="stop-color:${end};stop-opacity:1" />
      </linearGradient>
    </defs>
    <rect width="400" height="300" fill="url(#grad${normalizedIndex})"/>
    <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-family="sans-serif" font-size="24" font-weight="bold" fill="white">${safeName}</text>
  </svg>`

  return `data:image/svg+xml;base64,${btoa(unescape(encodeURIComponent(svg)))}`
}

/**
 * 原地交换数组相邻元素，返回是否真的发生了交换。
 *
 * 抽出来的原因：原实现是 `if (direction === 'up' && attrIndex > 0) {...}`，
 * 边界判断和交换写在一起，无法单测。首项上移、末项下移、越界索引都必须是无操作。
 */
export function moveInList<T>(list: T[], index: number, direction: MoveDirection): boolean {
  if (!Array.isArray(list)) return false
  if (!Number.isInteger(index) || index < 0 || index >= list.length) return false
  const target = direction === 'up' ? index - 1 : index + 1
  if (target < 0 || target >= list.length) return false
  const tmp = list[index]
  list[index] = list[target]
  list[target] = tmp
  return true
}

/** 每天至少保留一个景点，否则行程卡片会变成空壳。 */
export function canRemoveAttraction(list: unknown[]): boolean {
  return Array.isArray(list) && list.length > 1
}
