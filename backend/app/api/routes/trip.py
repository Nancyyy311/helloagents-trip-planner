"""旅行规划API路由"""

import asyncio
import hashlib
import json
import os
import threading
import time
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from ...models.schemas import (
    DayDiff,
    PlanDiff,
    RefineRequest,
    RefineResponse,
    TripPlan,
    TripPlanResponse,
    TripRequest,
)
from ...agents.trip_planner_agent import (
    MultiAgentTripPlanner,
    get_trip_planner_agent,
)
from ...planner.cancel import CancelToken, PlanCancelled
from ..guard import PlanningBusy, plan_gate, single_flight
from ..observability import metrics

router = APIRouter(prefix="/trip", tags=["旅行规划"])


# === 改造点：请求隔离 ===
# 每个生成请求/任务都用一个独立的 planner 实例，避免共享单例上的
# last_generation_status / _history 在并发时互相覆盖（串台）。
def _plan_trip_isolated(request: TripRequest, cancel_token: CancelToken | None = None):
    """在独立实例上生成计划，返回 (trip_plan, status, message)。"""
    planner = MultiAgentTripPlanner()
    trip_plan = planner.plan_trip(request, cancel_token=cancel_token)
    status = getattr(planner, "last_generation_status", "unknown")
    message = getattr(planner, "last_generation_message", "") or "旅行计划生成完成"
    return trip_plan, status, message


def _refine_isolated(
    request: TripRequest,
    previous_plan: TripPlan,
    instruction: str,
    cancel_token: CancelToken | None = None,
):
    """在独立实例上做增量修改，返回 (trip_plan, status, message)。"""
    planner = MultiAgentTripPlanner()
    trip_plan = planner.refine_trip(
        request, previous_plan, instruction, cancel_token=cancel_token
    )
    status = getattr(planner, "last_generation_status", "unknown")
    message = getattr(planner, "last_generation_message", "") or "增量修改完成"
    return trip_plan, status, message


# === 改造点：多轮会话（增量修改）的计划存储 ===
# 用户想改一句话（"第二天换成室内"），原项目只能整趟重新生成：等 60 秒，
# 而且新计划和上一版可能面目全非。这里把生成过的计划按 plan_id 存在内存里，
# 后续修改基于它做增量，只重新生成 1 次（约 40 秒）。
# 存储同样放在进程内存（单机够用），生产换 Redis 即可，接口不变。
_plans: dict[str, dict] = {}
_plans_lock = threading.Lock()
_PLAN_TTL_SECONDS = 3600
_PLAN_MAXSIZE = 200

# 合并等待上限：leader 自己的 LLM 预算就是 600 秒，等待方稍宽一点即可。
# 没有这个上限的话，leader 万一卡死在第三方 SDK 里，等待方会一起挂到天荒地老。
_COALESCE_WAIT_SECONDS = 620.0


def _plan_request_key(request: TripRequest) -> str:
    """把请求压成去重指纹。

    用完整字段的规范化 JSON 而不是只看城市+天数：偏好、预算、同行人
    任意一项不同，生成结果就该不同，不能合并。
    """
    payload = json.dumps(
        request.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# === 改造点：长任务异步化 ===
# 生成一次要几十秒，同步等待会超时/占连接。这里提供异步任务：
# POST /plan/async 立即返回 task_id，后台线程生成，前端轮询 GET /plan/async/{task_id}。
# 任务表放内存（单机够用），生产可换 Redis / Celery 等任务队列，接口不变。
_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()
_TASK_TTL_SECONDS = 3600
# 和计划表一样要有容量上限：一个任务里存着完整计划 JSON（几十 KB），
# 只有 TTL 的话，一小时内提交多少就堆多少，进程内存只涨不落。
_TASK_MAXSIZE = 200

# === 改造点：孤儿任务回收 ===
# 提交了任务、页面上却从此没有人再来问一次（既不轮询也不订阅 SSE），
# 说明用户已经走了。这种任务原项目会一路跑到底：3 次候选 LLM + 一串高德调用，
# 结果写进一张再也没人读的表。宽限 90 秒再动手，是因为：
#   - 前端轮询间隔 1.5s，正常情况一秒内就会 touch 一次；
#   - 浏览器把后台标签页的 setInterval 节流到约 1 分钟一次，
#     90s > 60s，切到别的 tab 不会被误杀；
#   - 设成 0 或负数即关闭回收（保守部署/调试时用）。
_TASK_ORPHAN_GRACE_SECONDS = float(os.getenv("TASK_ORPHAN_GRACE_SECONDS", "90") or 90)
# 后台清扫间隔：TTL 清理 + 孤儿回收共用同一个循环
_TASK_JANITOR_INTERVAL = float(os.getenv("TASK_JANITOR_INTERVAL_SECONDS", "5") or 5)

# SSE 推送参数：1 秒检查一次状态，15 秒发一次心跳，最多等 10 分钟
_SSE_POLL_INTERVAL = 1.0
_SSE_HEARTBEAT_SECONDS = 15.0
_SSE_MAX_SECONDS = 600.0

# 任务字典里不能出现在响应里的字段：
# cancel 是跨线程的 Event 对象，既不可 JSON 序列化，也不该暴露给前端。
_TASK_INTERNAL_FIELDS = ("cancel",)
# 任务终态：到这些状态后就不再接受取消，也允许被 TTL/容量淘汰
_TASK_TERMINAL = ("done", "failed", "canceled")


def _get_task_snapshot(task_id: str) -> dict | None:
    """持锁拷贝一份任务快照，避免返回过程中被后台线程改动。"""
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None:
            return None
        return {k: v for k, v in task.items() if k not in _TASK_INTERNAL_FIELDS}


def _get_task_token(task_id: str) -> CancelToken | None:
    """取出任务的取消令牌（只有取消接口和后台线程需要它）。"""
    with _tasks_lock:
        task = _tasks.get(task_id)
        return task.get("cancel") if task is not None else None


def _sse_frame(payload: dict, event: str | None = None) -> str:
    """按 SSE 协议拼一帧：可选 event 行 + data 行 + 空行结尾。"""
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _prune_tasks_locked() -> None:
    """清理过期任务，避免内存无限增长（需在持锁状态下调用）。

    两条约束缺一不可：
    1. 过期就删（TTL）；
    2. **正在跑的任务不能删**。删了用户就查不到结果，可钱已经花掉了。
    容量淘汰同样跳过 pending/running，只淘汰已经结束的。

    改造点：这里原来只判断"不是 pending/running"，新增的 `canceling`
    （已喊停、等线程在下一个检查点退出）会落进"可以删"的一侧——
    用户刚点了取消、再查一次却 404，会被当成"任务消失了"。改为白名单终态。
    """
    now = time.time()
    expired = [
        task_id
        for task_id, task in _tasks.items()
        if now - task.get("created_at", now) > _TASK_TTL_SECONDS
        and task.get("status") in _TASK_TERMINAL
    ]
    for task_id in expired:
        _tasks.pop(task_id, None)
        metrics.record_task_reaped()

    # 容量兜底：按创建顺序淘汰最旧的已完成任务
    if len(_tasks) >= _TASK_MAXSIZE:
        finished = [
            task_id
            for task_id, task in _tasks.items()
            if task.get("status") in _TASK_TERMINAL
        ]
        for task_id in finished[: max(0, len(_tasks) - _TASK_MAXSIZE + 1)]:
            _tasks.pop(task_id, None)
            metrics.record_task_reaped()


def _new_task() -> str:
    task_id = uuid.uuid4().hex
    now = time.time()
    with _tasks_lock:
        _prune_tasks_locked()
        _tasks[task_id] = {
            "status": "pending",
            "success": None,
            "message": "",
            "result": None,
            "error": None,
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            # 取消相关（cancel 是对象，响应里会被剔除，只暴露下面两个可读字段）
            "cancel": CancelToken(),
            "cancel_reason": "",
            "cancel_stage": "",
            # 谁还在看这个任务：轮询/SSE 都会刷新 last_seen
            "last_seen": now,
            "watched": False,
            "sse_subscribers": 0,
        }
    return task_id


def _touch_task(task_id: str) -> dict | None:
    """标记"还有人在看这个任务"，返回最新快照。

    没有这个心跳，服务端无法区分"用户还在等"和"用户早就走了"——
    孤儿回收就是靠它判断的，所以轮询接口必须调用它。
    """
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None:
            return None
        task["last_seen"] = time.time()
        task["watched"] = True
        return {k: v for k, v in task.items() if k not in _TASK_INTERNAL_FIELDS}


def _subscribe_task(task_id: str) -> bool:
    """SSE 订阅数 +1；任务不存在返回 False。"""
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None:
            return False
        task["sse_subscribers"] = int(task.get("sse_subscribers", 0)) + 1
        task["last_seen"] = time.time()
        task["watched"] = True
        return True


def _unsubscribe_task(task_id: str) -> None:
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is not None:
            task["sse_subscribers"] = max(0, int(task.get("sse_subscribers", 0)) - 1)


def cancel_task(task_id: str, reason: str = "client_deleted") -> tuple[str, dict | None]:
    """请求取消一个任务，返回 (结果, 快照)。

    结果的四种取值，对应接口层要给出的四种响应：

    - `missing`：任务不存在或已过期 -> 404
    - `finished`：已经跑完（done/failed/canceled），取消没有意义 -> 409，
      这是幂等语义下的正确做法：不假装成功，让调用方知道结果已经在里面了
    - `canceling`：任务在跑，已置取消信号，等线程在下一个检查点退出
    - `canceled`：任务还没开始（pending），直接改终态，后台线程会看到信号后空转返回

    为什么不用 202：调用方真正想知道的是"还会不会继续花钱"。
    对 pending 我们能立刻保证不会，对 running 只能保证"下一次花钱前停下"，
    所以用两个不同状态把这件事如实说清楚，而不是统一回一个模糊的"已受理"。
    """
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None:
            return "missing", None
        status = task.get("status")
        if status in _TASK_TERMINAL:
            return "finished", {k: v for k, v in task.items() if k not in _TASK_INTERNAL_FIELDS}

        token: CancelToken = task["cancel"]
        token.cancel(reason)
        task["cancel_reason"] = reason
        if status == "pending":
            task["status"] = "canceled"
            task["success"] = False
            task["message"] = "任务在开始前被取消"
            task["finished_at"] = time.time()
            result = "canceled"
        else:
            # 已经在跑：不能杀线程，只能等它在下一个检查点自己退出
            task["status"] = "canceling"
            task["message"] = "已请求取消，等待当前步骤结束"
            result = "canceling"
        snapshot = {k: v for k, v in task.items() if k not in _TASK_INTERNAL_FIELDS}

    # 指标在锁外记，避免持锁调用另一个带锁的组件
    if result == "canceled":
        metrics.record_task_canceled(reason, "before_start")
    return result, snapshot


def sweep_task_store() -> dict:
    """后台清扫一轮：过期清理 + 孤儿回收。由 main.py 的 janitor 协程定时调用。

    修掉一个原实现里的真实漏洞：`_prune_tasks_locked()` 只在**提交新任务**时
    被顺手调用。也就是说，用户提交过一个任务之后再也没人来过，这个任务会一直
    留在内存里直到进程重启——TTL 写了但没有任何人触发它。
    """
    reaped = 0
    abandoned = 0
    now = time.time()

    with _tasks_lock:
        before = len(_tasks)
        _prune_tasks_locked()
        reaped += before - len(_tasks)

    if _TASK_ORPHAN_GRACE_SECONDS > 0:
        candidates: list[str] = []
        with _tasks_lock:
            for task_id, task in _tasks.items():
                if task.get("status") not in ("pending", "running"):
                    continue
                if int(task.get("sse_subscribers", 0)) > 0:
                    continue
                if now - task.get("last_seen", now) > _TASK_ORPHAN_GRACE_SECONDS:
                    candidates.append(task_id)
        for task_id in candidates:
            result, _ = cancel_task(task_id, "orphan_reclaimed")
            if result in ("canceling", "canceled"):
                abandoned += 1
                metrics.record_task_abandoned()

    return {"reaped": reaped, "abandoned": abandoned}


def _update_task(task_id: str, **fields) -> None:
    with _tasks_lock:
        if task_id in _tasks:
            _tasks[task_id].update(fields)


def _run_plan_task(task_id: str, request: TripRequest) -> None:
    """后台任务：独立实例生成计划，把结果写回任务表。

    后台任务跑在 worker 线程里，和事件循环里的同步接口共用同一个闸门，
    否则异步接口就成了绕过并发的后门（挂着几十个 task 一样能压垮进程）。
    """
    token = _get_task_token(task_id)
    if token is None:
        # 任务已被清掉（TTL/容量淘汰），没有任何人需要这个结果
        return
    # 提交后立刻被取消（用户连点两下"取消"、或者前端根本没等）：
    # 连闸门都不该占，直接结束。
    if token.cancelled:
        _update_task(
            task_id,
            status="canceled",
            success=False,
            message="任务在开始前被取消",
            cancel_reason=token.reason,
            finished_at=time.time(),
        )
        return

    if not plan_gate.try_enter():
        metrics.record_planning_rejected()
        _update_task(
            task_id,
            status="failed",
            success=False,
            error=f"当前规划任务已达上限（{plan_gate.limit} 个），请稍后重试",
            finished_at=time.time(),
        )
        return
    _update_task(task_id, status="running", started_at=time.time())
    try:
        trip_plan, status, message = _plan_trip_isolated(request, cancel_token=token)
        # 任务可能在最后一步被取消，此时不该把"成功"写回去
        token.checkpoint("before_write_result")
        _update_task(
            task_id,
            status="done",
            success=(status == "llm_success"),
            message=message,
            result=trip_plan.model_dump(mode="json"),
            finished_at=time.time(),
            # 异步任务也返回 plan_id，前端拿到结果后同样可以继续增量修改
            plan_id=_remember_plan(trip_plan, request),
        )
    except PlanCancelled as exc:
        # 取消不是失败：不写 error（前端看到 error 会弹红框），只写"已取消"，
        # 并把停在哪个阶段记下来——这个字段决定了"省下了多少钱"。
        metrics.record_task_canceled(exc.reason, exc.stage)
        _update_task(
            task_id,
            status="canceled",
            success=False,
            message="任务已取消",
            error=None,
            cancel_reason=exc.reason,
            cancel_stage=exc.stage,
            finished_at=time.time(),
        )
        print(
            f"⏹️  任务 {task_id[:8]} 已取消: stage={exc.stage} reason={exc.reason} "
            f"（喊停后 {token.elapsed_since_cancel():.1f}s 才停下）",
            flush=True,
        )
    except Exception as e:
        _update_task(
            task_id, status="failed", success=False, error=str(e), finished_at=time.time()
        )
    finally:
        # 无论成功、失败还是取消，闸门都必须放掉，否则取消几次就把并发位漏光了
        plan_gate.leave()


def _remember_plan(trip_plan: TripPlan, request: TripRequest) -> str:
    """保存计划 + 原始请求，返回 plan_id。

    同时保存 request 是因为增量修改要复用 PlannerContext（命中缓存就不用再调高德），
    而上下文是按请求字段做缓存键的，没有原始请求就没法命中。
    """
    plan_id = uuid.uuid4().hex
    with _plans_lock:
        _prune_plans_locked()
        _plans[plan_id] = {
            "plan": trip_plan.model_dump(mode="json"),
            "request": request.model_dump(mode="json"),
            "created_at": time.time(),
            # 并发修改保护：同一计划同时只允许一轮 refine 在跑
            "refining": False,
        }
    return plan_id


def _claim_plan(plan_id: str) -> str:
    """占住一个计划准备修改，返回 ok / missing / busy。

    为什么需要：两轮 refine 同时打向同一个 plan_id 时，两边都是从同一版
    旧计划出发，各自生成一份新计划，后写的把先写的覆盖掉——用户点了两次
    "改成室内"，最后只剩一个改动生效，另一份白花了 15 秒和一次调用。
    这是典型的 lost update，用一把进程内的占用标志挡住即可。
    """
    with _plans_lock:
        item = _plans.get(plan_id)
        if item is None:
            return "missing"
        if item.get("refining"):
            return "busy"
        item["refining"] = True
        return "ok"


def _release_plan(plan_id: str) -> None:
    """释放占用。必须放在 finally 里，否则一次失败会把计划永久锁死。"""
    with _plans_lock:
        item = _plans.get(plan_id)
        if item is not None:
            item["refining"] = False


# === 改造点：增量修改也能被取消 ===
# `/plan` 走任务模型，取消靠 DELETE /plan/async/{id}；refine 是同步接口，
# 没有 task_id 可寻址。但 `_claim_plan` 已经保证"同一 plan_id 同时只有一轮
# refine"，所以 plan_id 天然就是这个任务的唯一标识——用它做键，
# 既不需要新增一套 id 体系，也不会和首次生成的任务表混淆。
_refine_tokens: dict[str, CancelToken] = {}


def _register_refine_token(plan_id: str) -> CancelToken:
    token = CancelToken()
    with _plans_lock:
        _refine_tokens[plan_id] = token
    return token


def _unregister_refine_token(plan_id: str, token: CancelToken) -> None:
    with _plans_lock:
        if _refine_tokens.get(plan_id) is token:
            _refine_tokens.pop(plan_id, None)


def cancel_refine(plan_id: str, reason: str = "client_deleted") -> tuple[str, dict | None]:
    """取消正在跑的增量修改，返回 (结果, token 快照)。

    - `idle`：该 plan 当前没有 refine 在跑（已结束或从没开始）-> 409
    - `canceling`：已置取消信号，等线程在下一个检查点退出
    """
    with _plans_lock:
        token = _refine_tokens.get(plan_id)
    if token is None:
        return "idle", None
    switched = token.cancel(reason)
    return ("canceling" if switched else "already_canceling"), token.snapshot()


def _load_plan(plan_id: str) -> tuple[TripPlan, TripRequest] | None:
    """按 plan_id 取回 (计划, 原始请求)；不存在或已过期返回 None。"""
    with _plans_lock:
        item = _plans.get(plan_id)
    if item is None:
        return None
    try:
        return TripPlan(**item["plan"]), TripRequest(**item["request"])
    except Exception as exc:
        print(f"⚠️  恢复计划失败 {plan_id}: {exc}")
        return None


def _prune_plans_locked() -> None:
    """清理过期计划并限制总容量（需在持锁状态下调用）。"""
    now = time.time()
    expired = [
        plan_id
        for plan_id, item in _plans.items()
        if now - item.get("created_at", now) > _PLAN_TTL_SECONDS
    ]
    for plan_id in expired:
        _plans.pop(plan_id, None)
    # 容量兜底：按插入顺序淘汰最旧的，但正在被修改的那条不能动
    # （淘汰了它，正在跑的 refine 结束后就找不到原计划，用户白等一轮）
    while len(_plans) >= _PLAN_MAXSIZE:
        victim = next(
            (plan_id for plan_id, item in _plans.items() if not item.get("refining")),
            None,
        )
        if victim is None:
            break
        _plans.pop(victim, None)


def diff_plans(before: TripPlan, after: TripPlan) -> PlanDiff:
    """对比两版计划，产出"只改了哪天"的证据。

    增量修改最大的卖点是"不动无关部分"，光喊口号没用 ——
    面试官问"你怎么证明只改了一天"，这张 diff 就是答案。
    """
    before_by_index = {day.day_index: day for day in before.days}
    changed: list[DayDiff] = []
    unchanged = 0

    for day in after.days:
        old = before_by_index.get(day.day_index)
        before_attractions = [item.name for item in (old.attractions if old else [])]
        after_attractions = [item.name for item in day.attractions]
        before_meals = [item.name for item in (old.meals if old else [])]
        after_meals = [item.name for item in day.meals]
        old_hotel = old.hotel.name if old and old.hotel else None
        new_hotel = day.hotel.name if day.hotel else None

        is_changed = (
            old is None
            or before_attractions != after_attractions
            or before_meals != after_meals
            or old_hotel != new_hotel
        )
        if is_changed:
            changed.append(
                DayDiff(
                    day_index=day.day_index,
                    date=day.date,
                    changed=True,
                    before_attractions=before_attractions,
                    after_attractions=after_attractions,
                    before_meals=before_meals,
                    after_meals=after_meals,
                )
            )
        else:
            unchanged += 1

    budget_before = before.budget.total if before.budget else 0
    budget_after = after.budget.total if after.budget else 0
    return PlanDiff(
        changed_days=changed,
        unchanged_days=unchanged,
        budget_before=budget_before,
        budget_after=budget_after,
        budget_delta=budget_after - budget_before,
        suggestions_changed=before.overall_suggestions != after.overall_suggestions,
    )


@router.post(
    "/plan",
    response_model=TripPlanResponse,
    summary="生成旅行计划",
    description="根据用户输入的旅行需求,生成详细的旅行计划"
)
async def plan_trip(payload: TripRequest):
    """
    生成旅行计划（同步接口）

    Args:
        payload: 旅行请求参数

    Returns:
        旅行计划响应
    """
    try:
        print(f"\n{'='*60}")
        print("📥 收到旅行规划请求:")
        print(f"   城市: {payload.city}")
        print(f"   日期: {payload.start_date} - {payload.end_date}")
        print(f"   天数: {payload.travel_days}")
        print(f"{'='*60}\n")

        # === 改造点：不阻塞事件循环 ===
        # plan_trip 内部是同步阻塞调用（几十秒）。原代码直接在 async 路由里调用，
        # 单 worker 下会把整个事件循环堵死，其他请求（连 /health）都卡住。
        # 用 asyncio.to_thread 丢到线程池执行，事件循环立即可服务其他请求。
        # === 改造点：相同请求合并（single flight）===
        # 用户连点两次"生成"、或前端超时重试，会发出两个一模一样的请求。
        # 不合并就是两份 3 候选的账单，而且结果几乎一样。
        coalesce_key = _plan_request_key(payload)
        is_leader, future = single_flight.claim(coalesce_key)
        coalesced = not is_leader

        if coalesced:
            metrics.record_coalesced()
            print("↩️  相同请求正在生成中，合并等待同一份结果（不重复调用 LLM）")
            try:
                trip_plan, generation_status, generation_message = await asyncio.wait_for(
                    asyncio.wrap_future(future), timeout=_COALESCE_WAIT_SECONDS
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504, detail="等待相同请求的结果超时，请稍后重试"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=502, detail=f"相同请求的生成失败: {e}"
                )
        else:
            # === 改造点：并发闸门 ===
            # 限流管"一分钟能进几个"，闸门管"同一时刻在做几个"。
            # 只有 leader 占闸门：等待方不产生 LLM 调用，不该占用规划额度。
            if not plan_gate.try_enter():
                single_flight.finish(coalesce_key, future, error=PlanningBusy("并发已满"))
                metrics.record_planning_rejected()
                raise HTTPException(
                    status_code=503,
                    detail=f"当前规划任务已达上限（{plan_gate.limit} 个），请稍后重试",
                    headers={"Retry-After": "5"},
                )
            print("🚀 开始生成旅行计划（线程池执行）...")
            # 关于"客户端断开就取消"：这里**刻意没有**做断连探测。
            # 实测结论（见 README_改造说明.md 4.8）：在 FastAPI + BaseHTTPMiddleware
            # 这套栈上，`request.is_disconnected()` 在请求体已经被读取之后恒返回
            # False（请求体未读时有效），放在独立 task 里调用同样无效；
            # 退而求其次用 StreamingResponse 定时写 keepalive，少量数据会被内核
            # 缓冲区吃掉，同样观察不到断开。
            # 所以同步接口这条路上，可靠的取消来源只有一个：客户端自己说不要了。
            # 前端因此改用 /plan/async + 轮询，并在页面卸载时发 DELETE。
            try:
                trip_plan, generation_status, generation_message = await asyncio.to_thread(
                    _plan_trip_isolated, payload
                )
            except Exception as e:
                # 先把失败抛给等待者，再往上冒泡
                single_flight.finish(coalesce_key, future, error=e)
                raise
            else:
                single_flight.finish(
                    coalesce_key,
                    future,
                    result=(trip_plan, generation_status, generation_message),
                )
            finally:
                plan_gate.leave()

        # === 改造点：错误语义（消除"假成功"）===
        # 只有真正走 LLM 生成成功（llm_success）才返回 success=True。
        # 走 fallback（LLM 失败后用模板兜底）时，如实返回 HTTP 502 + success=False，
        # 并把降级计划一并带回，让前端可自行决定"降级展示"还是"提示重试"。
        if generation_status == "fallback_success":
            print(f"⚠️  {generation_message}, 返回降级响应(success=false, 502)\n")
            return JSONResponse(
                status_code=502,
                content=TripPlanResponse(
                    success=False,
                    message=generation_message,
                    data=trip_plan,
                    plan_id=_remember_plan(trip_plan, payload),
                    coalesced=coalesced,
                ).model_dump(mode="json"),
            )

        print(f"✅ {generation_message}, 准备返回响应\n")
        # 改造点：返回 plan_id，前端拿着它就能"改一句话"做增量修改
        return TripPlanResponse(
            success=True,
            message=generation_message,
            data=trip_plan,
            plan_id=_remember_plan(trip_plan, payload),
            coalesced=coalesced,
        )

    except HTTPException:
        # 限流/并发/合并相关的状态码必须原样返回，
        # 不能被下面那个兜底 except 改成 500。
        raise
    except Exception as e:
        print(f"❌ 生成旅行计划失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"生成旅行计划失败: {str(e)}"
        )


@router.post(
    "/plan/{plan_id}/refine",
    response_model=RefineResponse,
    summary="增量修改旅行计划（多轮会话）",
    description=(
        "基于已生成的计划做最小改动：只改 instruction 涉及的内容，其余原样保留。"
        "返回修改后的完整计划 + 变更摘要（改了哪几天、预算变化多少）。"
    ),
)
async def refine_plan(plan_id: str, payload: RefineRequest):
    """增量修改：让用户"改一句话"，而不是整趟重新生成。

    为什么不做整趟重生成：
    - 慢（3 候选 + rerank 约 60 秒，而增量只跑 1 次约 40 秒）；
    - 危险（重新生成会把用户已经满意的天也随机改掉，等于抽奖）。
    """
    restored = _load_plan(plan_id)
    if restored is None:
        raise HTTPException(status_code=404, detail="计划不存在或已过期，请重新生成")

    # === 改造点：同一计划并发修改加保护 ===
    # 两轮 refine 同时进来，各自都基于同一份旧计划生成，后写的覆盖先写的，
    # 用户的一次修改会凭空消失（lost update）。这里用一个占用标志挡住。
    claim_state = _claim_plan(plan_id)
    if claim_state == "busy":
        metrics.record_refine_conflict()
        raise HTTPException(
            status_code=409,
            detail="该计划正在修改中，请等待当前这次完成后再试",
        )
    if claim_state == "missing":
        raise HTTPException(status_code=404, detail="计划不存在或已过期，请重新生成")

    previous_plan, request = restored
    print(f"\n📥 收到增量修改请求 plan_id={plan_id[:8]}... 要求: {payload.instruction}")

    # 增量修改也是一次真实 LLM 调用，必须和首次生成共用同一个并发闸门，
    # 否则"改一句话"就成了绕过限流的后门。
    if not plan_gate.try_enter():
        _release_plan(plan_id)
        metrics.record_planning_rejected()
        raise HTTPException(
            status_code=503,
            detail=f"当前规划任务已达上限（{plan_gate.limit} 个），请稍后重试",
            headers={"Retry-After": "5"},
        )

    try:
        # 同样用 to_thread：修改也是几十秒的同步阻塞调用。
        # === 改造点（本批）：refine 也能被取消 ===
        # 此前这里没传 cancel_token，等于"首次生成能停、改一句话停不了"——
        # 同样是四十秒的 LLM 调用，用户关掉页面照样跑完照扣钱。
        # 令牌按 plan_id 注册，客户端用 DELETE /plan/{plan_id}/refine 触发。
        refine_token = _register_refine_token(plan_id)
        try:
            new_plan, _status, message = await asyncio.to_thread(
                _refine_isolated, request, previous_plan, payload.instruction, refine_token
            )
        finally:
            _unregister_refine_token(plan_id, refine_token)
    except PlanCancelled as exc:
        # 取消不是失败：原计划没有被改动，如实告诉调用方"已取消、原计划完好"
        metrics.record_task_canceled(exc.reason, exc.stage)
        print(
            f"⏹️  增量修改已取消: stage={exc.stage} reason={exc.reason}（原计划未受影响）",
            flush=True,
        )
        raise HTTPException(status_code=499, detail="增量修改已取消，原计划未受影响")
    except Exception as e:
        print(f"❌ 增量修改失败: {e}")
        import traceback
        traceback.print_exc()
        # 关键：修改失败时明确告知"原计划未受影响"，前端可以继续用上一版
        raise HTTPException(
            status_code=502,
            detail=f"增量修改失败，原计划未受影响: {str(e)}",
        )
    finally:
        plan_gate.leave()
        _release_plan(plan_id)

    diff = diff_plans(previous_plan, new_plan)
    print(
        f"✅ 增量修改完成: 变更 {len(diff.changed_days)} 天 / 未变 {diff.unchanged_days} 天, "
        f"预算 {diff.budget_before} -> {diff.budget_after}",
        flush=True,
    )
    return RefineResponse(
        success=True,
        message=message,
        # 返回新的 plan_id，支持"连续改多轮"（改完第二天再改第一天）
        plan_id=_remember_plan(new_plan, request),
        data=new_plan,
        diff=diff,
    )


@router.delete(
    "/plan/{plan_id}/refine",
    summary="取消正在进行的增量修改",
    description=(
        "取消一次还没跑完的增量修改。原计划不受影响，调用方可以继续用上一版。"
        "该 plan 当前没有 refine 在跑时返回 409。"
    ),
)
async def cancel_refine_endpoint(plan_id: str, reason: str = "client_deleted"):
    """取消正在跑的增量修改。

    为什么需要这个接口：refine 同样是四十秒级的 LLM 调用，此前完全没有
    中止入口——用户关掉页面、或者想换一句要求重来，前一次调用照样跑完。
    首次生成有 DELETE /plan/async/{id}，refine 一直没有对应能力，这里补齐。
    """
    result, snapshot = cancel_refine(plan_id, reason)
    if result == "idle":
        raise HTTPException(
            status_code=409,
            detail="该计划当前没有正在进行的修改",
        )
    return {"plan_id": plan_id, "cancel": result, "token": snapshot}


@router.post(
    "/plan/async",
    summary="提交异步生成任务",
    description="立即返回 task_id，后台生成，前端轮询 /api/trip/plan/async/{task_id} 获取结果"
)
async def plan_trip_async(request: TripRequest, background_tasks: BackgroundTasks):
    """提交异步生成任务，立即返回 task_id。"""
    task_id = _new_task()
    background_tasks.add_task(_run_plan_task, task_id, request)
    return {
        "task_id": task_id,
        "status": "pending",
        "poll": f"/api/trip/plan/async/{task_id}",
    }


@router.get(
    "/plan/async/{task_id}",
    summary="查询异步任务状态",
    description="返回任务状态（pending/running/done/failed）与结果"
)
async def get_plan_task(task_id: str):
    """查询异步任务状态与结果（轮询方式）。

    用 `_touch_task` 而不是只读快照：这个调用同时是"我还在这儿等着"的心跳。
    少了它，一个老老实实轮询的前端也会被孤儿回收当成"没人要了"。
    """
    snapshot = _touch_task(task_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return {"task_id": task_id, **snapshot}


@router.delete(
    "/plan/async/{task_id}",
    summary="取消异步任务",
    description=(
        "取消一个还没跑完的生成任务。pending 立刻变为 canceled；"
        "running 会置取消信号，在下一个检查点停下并变为 canceled。"
    ),
)
async def cancel_plan_task(task_id: str, reason: str = "client_deleted"):
    """取消任务，停止继续消耗 LLM 与高德配额。

    原项目没有这个能力：任务一旦提交就跑到头。用户关掉页面、换成别的城市、
    连点两次生成，前一个任务照样把 3 个候选跑完，钱照扣。
    """
    result, snapshot = cancel_task(task_id, reason)
    if result == "missing":
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    if result == "finished":
        # 409 而不是 200：调用方需要知道"结果已经在那儿了，别再等了"
        raise HTTPException(
            status_code=409,
            detail=f"任务已结束（{snapshot.get('status') if snapshot else 'unknown'}），无需取消",
        )
    return {"task_id": task_id, "cancel": result, **snapshot}


@router.get(
    "/plan/async/{task_id}/stream",
    summary="订阅异步任务进度（SSE）",
    description="以 Server-Sent Events 推送状态变化，算完立刻推结果，前端无需轮询"
)
async def stream_plan_task(task_id: str):
    """SSE 推送任务进度。

    改造点（相对轮询）：轮询是"前端定时问"，空转请求多、结果最多晚一个间隔才看到；
    SSE 是"服务端状态一变就推"，一次连接拿到底，还能顺带发心跳保活。

    改造点（本批）：订阅计数 + 断连可观测。
    原实现是个 `while True`，完全不看客户端还在不在：浏览器关页面后连接被动消失，
    日志上没有任何痕迹，而任务还在后台跑着 3 个候选。

    注意：这里**不用** `request.is_disconnected()`。实测在请求体已被读取的场景下
    它恒返回 False（详见 README_改造说明.md 4.8），靠它判断断连是自欺欺人。
    可靠的做法是让 Starlette 自己收尾：客户端断开时它会取消这个生成器，
    finally 一定执行；在 finally 里看任务有没有跑到终态，就能准确判断
    "这次连接是正常收完的还是中途断的"。
    """
    if _get_task_snapshot(task_id) is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")

    async def event_generator():
        last_status = None
        started_at = time.time()
        last_heartbeat = time.time()
        _subscribe_task(task_id)
        metrics.record_sse_connect()
        try:
            while True:
                snapshot = _get_task_snapshot(task_id)
                if snapshot is None:
                    yield _sse_frame({"task_id": task_id, "message": "任务已过期"}, "error")
                    return

                status = snapshot.get("status")
                if status != last_status:
                    last_status = status
                    # 状态一变就推一帧，前端可以据此切换"排队中/生成中/取消中"
                    yield _sse_frame({"task_id": task_id, "status": status}, "status")

                if status in _TASK_TERMINAL:
                    yield _sse_frame({"task_id": task_id, **snapshot}, "result")
                    return

                if time.time() - started_at > _SSE_MAX_SECONDS:
                    yield _sse_frame(
                        {"task_id": task_id, "message": "等待超时，请改用轮询接口"},
                        "timeout",
                    )
                    return

                await asyncio.sleep(_SSE_POLL_INTERVAL)

                # 心跳：长时间没有状态变化时，避免网关/浏览器把空闲连接掐断
                if time.time() - last_heartbeat >= _SSE_HEARTBEAT_SECONDS:
                    yield ": ping\n\n"
                    last_heartbeat = time.time()
        finally:
            _unsubscribe_task(task_id)
            snapshot = _get_task_snapshot(task_id)
            if snapshot is not None and snapshot.get("status") not in _TASK_TERMINAL:
                # 任务还没跑到终态，连接就没了 —— 用户关了页面或者切走了。
                # 这时候任务还在后台烧钱，必须能看见。
                metrics.record_sse_disconnect()
                print(
                    f"🔌 SSE 断开 task={task_id[:8]} "
                    f"（连接维持 {time.time() - started_at:.1f}s，任务仍在 {snapshot.get('status')}）",
                    flush=True,
                )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 告诉 Nginx 不要缓冲，否则 SSE 会被攒着一起发
        },
    )


@router.get(
    "/health",
    summary="健康检查",
    description="检查旅行规划服务是否正常"
)
async def health_check():
    """
    健康检查

    改造点：原代码读 `agent.agent.name`，但 MultiAgentTripPlanner 并没有 `.agent`
    属性（它持有的是 `planner_agent` / `fallback_planner_agent`），因此每次都会抛
    AttributeError，被 except 捕获后恒定返回 503 —— 健康检查形同虚设。
    这里改为安全取值，并返回真正有意义的自检信息。
    """
    try:
        agent = get_trip_planner_agent()
        planner = getattr(agent, "planner_agent", None)
        return {
            "status": "healthy",
            "service": "trip-planner",
            "agent_name": getattr(planner, "name", "trip-planner"),
            "last_generation_status": getattr(agent, "last_generation_status", "idle"),
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"服务不可用: {str(e)}"
        )
