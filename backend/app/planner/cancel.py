"""任务取消：把"用户已经不想要了"这件事实，安全地传进正在跑的生成流程。

为什么需要它
------------
一次 3 天规划要跑 3 个候选，约 15~40 秒；增量修改也要十几秒。这段时间里
用户完全可能已经关掉页面、切走、或者点了"重新生成"。原项目对此毫无感知：
后台线程照样把 3 次 LLM 调用和高德请求跑完，结果写进一张没人会来读的表，
钱实实在在花掉了。

为什么是"协作式取消"而不是"杀线程"
----------------------------------
LLM 的 Python SDK 是同步阻塞调用，一次 `client.chat.completions.create` 发出去，
Python 层没有任何手段从外部把它掐断（threading 没有安全的 kill）。
所以能做到、也只能做到的是：**在几个花钱的边界上检查一次"还需不需要继续"**，
一旦发现不需要就抛异常把整趟流程停下，保证不再提交新的调用。
本次已经在飞的那一次调用仍然会跑完——这是必须如实告诉调用方的边界。

设计取舍
--------
1. 用 `threading.Event` 而不是 asyncio 的取消机制：生成流程整个泡在 worker
   线程里（`asyncio.to_thread`），跨线程唯一可靠的信号就是 Event。
   Event 还顺带解决了"退避等待"的问题：原来重试前 `time.sleep(5)` 睡死 5 秒，
   现在用 `wait(5)`，取消时立刻返回。
2. `PlanCancelled` 继承 `Exception` 而不是 `BaseException`：保持常规异常语义，
   让上层 `except Exception` 的行为可预期。**代价**是必须在上层显式放行——
   `plan_trip` 的兜底 `except Exception` 会把任何异常翻译成"fallback 计划"，
   漏一处放行，取消就会变成一个 200 的假成功。这是本模块最需要测试守住的边界。
3. 记录 stage / cancelled_at / elapsed：取消发生在哪一步、喊停之后多久才真正停下，
   这两个数决定了"到底省下了多少"，必须能读出来，否则无法证明它有用。
"""

from __future__ import annotations

import threading
import time
from typing import Optional


class PlanCancelled(Exception):
    """生成流程被主动取消（不是失败，不该重试、也不该回退成模板计划）。"""

    def __init__(self, stage: str = "", reason: str = "client_aborted") -> None:
        self.stage = stage or "unknown"
        self.reason = reason or "client_aborted"
        super().__init__(f"plan cancelled at {self.stage}: {self.reason}")


class CancelToken:
    """跨线程的取消信号，带一个"最后停在哪一步"的诊断信息。

    用法：
        token = CancelToken()
        # ... 业务代码里在花钱的边界插入 token.checkpoint("before_llm")
        token.cancel("user_deleted_task")   # 另一个线程里喊停
    """

    __slots__ = ("_event", "_reason", "_stage", "_cancelled_at", "_waiting")

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason = ""
        self._stage = ""
        self._cancelled_at: Optional[float] = None
        # 取消时正在等锁/等退避，次数可用于判断"是卡在等待里被取消的"
        self._waiting = 0

    # ---- 状态 ----

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def stage(self) -> str:
        """最后一次 checkpoint 所在的阶段（取消时就是"死在哪一步"）。"""
        return self._stage

    @property
    def interrupts(self) -> int:
        """被取消打断的等待次数（退避 sleep / 等锁）。"""
        return self._waiting

    def elapsed_since_cancel(self) -> float:
        """从喊停到现在过了多久（秒）；没取消返回 0。"""
        if self._cancelled_at is None:
            return 0.0
        return time.time() - self._cancelled_at

    # ---- 写入 ----

    def cancel(self, reason: str = "client_aborted") -> bool:
        """喊停。重复调用安全，第一次的原因会被保留（更接近真实起因）。

        Returns:
            True 表示这次调用真正完成了状态切换（用于"取消了一次"的计数）。
        """
        if self._event.is_set():
            return False
        self._reason = reason or "client_aborted"
        self._cancelled_at = time.time()
        self._event.set()
        return True

    # ---- 读取（在各阶段调用）----

    def checkpoint(self, stage: str) -> None:
        """到了"要花钱"的边界，先问一句还需不需要继续。"""
        self._stage = stage
        if self._event.is_set():
            raise PlanCancelled(self._stage, self._reason)

    def sleep(self, seconds: float, stage: str = "") -> None:
        """可被取消打断的等待。

        退避重试原来用 `time.sleep(delay)`：一旦取消，还要把 5s / 8s 的退避睡完
        才轮到下一个 checkpoint，等于"喊停了还多等十几秒"。用 Event.wait 可以
        在被取消的瞬间返回。
        """
        if stage:
            self._stage = stage
        if seconds <= 0:
            self.checkpoint(stage or self._stage)
            return
        woke = self._event.wait(seconds)
        if woke:
            self._waiting += 1
            raise PlanCancelled(self._stage or stage, self._reason)

    def snapshot(self) -> dict:
        """给 /metrics 与任务状态用的只读快照。"""
        return {
            "cancelled": self.cancelled,
            "reason": self._reason,
            "stage": self._stage,
            "waited_after_cancel_s": round(self.elapsed_since_cancel(), 3),
            "interrupted_waits": self._waiting,
        }
