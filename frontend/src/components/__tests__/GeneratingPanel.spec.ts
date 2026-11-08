// @vitest-environment jsdom
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import Antd from 'ant-design-vue'
import GeneratingPanel from '@/components/GeneratingPanel.vue'

/**
 * 这组用例存在的理由很具体：用户反馈"页面里根本没看到取消界面"。
 * 只读代码确认 `v-if` 条件是对的，证明不了它真的渲染出来——
 * 所以这里用 jsdom 真挂载一次，断言按钮确实在 DOM 里、文案对、点了会往外抛事件。
 */

function mountPanel(props: Record<string, unknown> = {}) {
  return mount(GeneratingPanel, {
    props: { percent: 40, status: '📋 正在生成行程计划...', cancelRequested: false, ...props },
    global: { plugins: [Antd] }
  })
}

describe('GeneratingPanel 取消入口', () => {
  it('生成中必须渲染出可点击的取消按钮', () => {
    const wrapper = mountPanel()
    const btn = wrapper.find('button')
    expect(btn.exists()).toBe(true)
    expect(wrapper.text()).toContain('中止本次生成')
  })

  it('取消按钮是实心主按钮，不是小号文字按钮（用户找不到的根因）', () => {
    const wrapper = mountPanel()
    const classes = wrapper.find('button').classes()
    expect(classes).toContain('ant-btn-primary')
    expect(classes).toContain('ant-btn-dangerous')
    expect(classes).not.toContain('ant-btn-text')
    expect(classes).not.toContain('ant-btn-sm')
  })

  it('点击按钮向外抛出 cancel 事件', async () => {
    const wrapper = mountPanel()
    await wrapper.find('button').trigger('click')
    expect(wrapper.emitted('cancel')).toHaveLength(1)
  })

  it('已请求取消后按钮消失，换成等待提示，避免连点', () => {
    const wrapper = mountPanel({ cancelRequested: true })
    expect(wrapper.find('button').exists()).toBe(false)
    expect(wrapper.text()).toContain('已请求取消')
    expect(wrapper.text()).toContain('通常 1 秒内')
  })

  it('进度与状态文案来自 props，不是硬编码', () => {
    const wrapper = mountPanel({ percent: 70, status: '⏳ 任务排队中...' })
    expect(wrapper.text()).toContain('⏳ 任务排队中...')
    expect(wrapper.find('.ant-progress').exists()).toBe(true)
  })
})
