<template>
  <div class="generating-panel">
    <a-progress
      :percent="percent"
      status="active"
      :stroke-color="{ '0%': '#0f766e', '100%': '#2563eb' }"
      :stroke-width="8"
    />
    <p class="generating-status">{{ status }}</p>

    <template v-if="!cancelRequested">
      <p class="generating-tip">
        生成通常需要 15~30 秒。不想等可以随时中止，不会留下半成品。
      </p>
      <a-button
        danger
        type="primary"
        size="large"
        class="cancel-btn"
        aria-label="中止本次生成"
        @click="emit('cancel')"
      >
        <CloseCircleOutlined />
        <span>中止本次生成</span>
      </a-button>
    </template>
    <p v-else class="generating-canceling">
      ⏹️ 已请求取消，正在等待当前步骤结束（通常 1 秒内）
    </p>
  </div>
</template>

<script setup lang="ts">
import { CloseCircleOutlined } from '@ant-design/icons-vue'

/**
 * 生成中的进度面板。
 *
 * 为什么单独抽成组件：这段 UI 里有一个"取消"入口，它是异步任务链路的
 * 用户侧闭环，属于**必须能被验证存在**的东西，而不是藏在 900 行页面里的
 * 一个小号文字按钮。抽出来之后可以用 jsdom 挂载断言"按钮真的渲染了、
 * 点击真的向外抛事件"，而不是靠读代码相信它存在。
 */
defineProps<{
  percent: number
  status: string
  cancelRequested: boolean
}>()

const emit = defineEmits<{ (e: 'cancel'): void }>()
</script>

<style scoped>
.generating-panel {
  text-align: center;
  padding: 22px 20px 20px;
  border: 1px dashed #1677ff;
  border-radius: 10px;
  background: #f7fbff;
}

.generating-status {
  margin: 14px 0 0;
  color: #0958d9;
  font-size: 15px;
  font-weight: 650;
}

.generating-tip {
  margin: 8px 0 0;
  color: #64748b;
  font-size: 13px;
}

/* 原来是 type="text" 的小号文字按钮，看着像脚注，用户根本找不到。
   改成实心红色主按钮：它是个"放弃当前操作"的决定，必须显眼。 */
.cancel-btn {
  margin-top: 12px;
  min-width: 190px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
}

.generating-canceling {
  margin: 12px 0 0;
  color: #d46b08;
  font-size: 14px;
}
</style>
