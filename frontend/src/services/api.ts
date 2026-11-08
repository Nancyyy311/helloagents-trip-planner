import axios from 'axios'
import type { TripFormData, TripPlanResponse } from '@/types'

const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim()

export const API_BASE_URL = configuredApiBaseUrl
  ? configuredApiBaseUrl.replace(/\/$/, '')
  : ''

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  // 必须大于后端的 PLANNER_REQUEST_TIMEOUT（600s）。
  // 两边都设 600000 时，前端几乎必然先触发超时（还要叠加网络和序列化开销），
  // 于是用户看到的是 axios 的 "timeout of 600000ms exceeded"，
  // 后端那条"模型超时"的明确报错永远没机会返回。
  timeout: 900000, // 15 分钟
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    console.log('发送请求:', config.method?.toUpperCase(), config.url)
    return config
  },
  (error) => {
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
apiClient.interceptors.response.use(
  (response) => {
    console.log('收到响应:', response.status, response.config.url)
    return response
  },
  (error) => {
    console.error('响应错误:', error.response?.status, error.message)
    return Promise.reject(error)
  }
)

/**
 * 生成旅行计划
 *
 * 改造点（批次9）：从"一次 18 秒的同步请求"改成"提交任务 + 轮询"。
 *
 * 为什么必须换：同步接口下，用户关掉页面时服务端**没有办法知道**。
 * 实测 `request.is_disconnected()` 在 FastAPI + BaseHTTPMiddleware 这套栈上
 * 只要请求体被读过就恒返回 False（详见 README_改造说明.md 4.8），
 * 服务端只能继续把 3 个候选跑完、把结果写进一个没人会来读的响应。
 *
 * 换成任务模型后，"客户端还在不在"变成一个明确的、客户端主动表达的信号：
 *   - 轮询就是心跳：还在轮询 = 还在等（后端 last_seen 会被刷新）；
 *   - 页面卸载时发一个 DELETE（keepalive）= 我不等了，后端立刻在下一个
 *     检查点停下，不再提交新的 LLM 调用；
 *   - 浏览器崩溃/断网连 DELETE 都发不出去时，后端 90 秒没收到轮询就按
 *     "孤儿任务"回收掉。三条路兜住，没有一条依赖服务端猜。
 */
export type PlanTaskStatus = 'pending' | 'running' | 'canceling' | 'done' | 'failed' | 'canceled'

export interface GenerateTripPlanOptions {
  /** 任务状态变化回调，用来把"排队中/生成中"如实显示给用户 */
  onStatus?: (status: PlanTaskStatus) => void
  /**
   * 任务提交成功后回调 task_id。
   *
   * 为什么需要它：`pagehide` 只能在"关页面"时取消，用户在生成过程中想主动
   * 放弃（换个城市重来），页面还开着，就必须拿到 task_id 才能发 DELETE。
   * 不暴露这个 id，后端的取消接口就永远只能被"关闭页面"这一个场景触发。
   */
  onSubmitted?: (taskId: string) => void
}

const POLL_INTERVAL_MS = 1200
const POLL_TIMEOUT_MS = 15 * 60 * 1000

/** 当前正在轮询的任务，供页面卸载时取消 */
let inFlightTaskId: string | null = null

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

/** 主动取消一个还在跑的任务（用户点取消、离开页面、重新生成） */
export async function cancelPlanTask(taskId: string, reason = 'client_deleted'): Promise<void> {
  try {
    await apiClient.delete(`/api/trip/plan/async/${taskId}`, { params: { reason } })
  } catch {
    // 404（已过期）/409（已结束）都不是错误：结果已经在那儿了，没什么可取消的
  }
}

// 页面卸载 = "我不等了"最明确的一次表达。
// 用 fetch + keepalive 而不是 axios：页面已经在卸载流程里，普通 XHR 会被浏览器直接掐断。
if (typeof window !== 'undefined') {
  window.addEventListener('pagehide', () => {
    if (!inFlightTaskId) return
    const taskId = inFlightTaskId
    inFlightTaskId = null
    try {
      fetch(`${API_BASE_URL}/api/trip/plan/async/${taskId}?reason=page_unload`, {
        method: 'DELETE',
        keepalive: true
      }).catch(() => undefined)
    } catch {
      /* 卸载阶段任何异常都不该影响页面 */
    }
  })
}

export async function generateTripPlan(
  formData: TripFormData,
  options: GenerateTripPlanOptions = {}
): Promise<TripPlanResponse> {
  let taskId: string
  try {
    const submitted = await apiClient.post<{ task_id: string }>('/api/trip/plan/async', formData)
    taskId = submitted.data.task_id
    inFlightTaskId = taskId
    options.onSubmitted?.(taskId)
  } catch (error: any) {
    throw new Error(error.response?.data?.detail || error.message || '提交生成任务失败')
  }

  const deadline = Date.now() + POLL_TIMEOUT_MS
  let lastStatus: PlanTaskStatus | null = null

  try {
    while (Date.now() < deadline) {
      await sleep(POLL_INTERVAL_MS)

      let snapshot: any
      try {
        const resp = await apiClient.get(`/api/trip/plan/async/${taskId}`)
        snapshot = resp.data
      } catch (error: any) {
        // 404 = 任务被后台清扫掉了（TTL/容量）。如实告诉用户重试，别装作还在等。
        if (error.response?.status === 404) {
          throw new Error('生成任务已失效，请重新生成')
        }
        throw new Error(error.response?.data?.detail || error.message || '查询生成进度失败')
      }

      const status = snapshot.status as PlanTaskStatus
      if (status !== lastStatus) {
        lastStatus = status
        options.onStatus?.(status)
      }

      if (status === 'done') {
        return {
          success: Boolean(snapshot.success),
          message: snapshot.message || '',
          data: snapshot.result,
          plan_id: snapshot.plan_id
        }
      }
      if (status === 'failed') {
        throw new Error(snapshot.error || snapshot.message || '生成旅行计划失败')
      }
      if (status === 'canceled') {
        throw new Error(snapshot.message || '生成已取消')
      }
      // pending / running / canceling：继续轮询
    }
    throw new Error('生成超时，请稍后重试')
  } finally {
    // 正常收完也要清掉，否则之后任何一次页面卸载都会对着旧 task 发 DELETE
    if (inFlightTaskId === taskId) inFlightTaskId = null
  }
}

/**
 * 健康检查
 */
export async function healthCheck(): Promise<any> {
  try {
    const response = await apiClient.get('/health')
    return response.data
  } catch (error: any) {
    console.error('健康检查失败:', error)
    throw new Error(error.message || '健康检查失败')
  }
}

export default apiClient
