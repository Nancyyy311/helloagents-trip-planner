"""旅行规划API路由"""

import asyncio
import hashlib
import json
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
from ..guard import PlanningBusy, plan_gate, single_flight
from ..observability import metrics

router = APIRouter(prefix="/trip", tags=["旅行规划"])


# === 改造点：请求隔离 ===
# 每个生成请求/任务都用一个独立的 planner 实例，避免共享单例上的
# last_generation_status / _history 在并发时互相覆盖（串台）。
def _plan_trip_isolated(request: TripRequest):
    """在独立实例上生成计划，返回 (trip_plan, status, message)。"""
    planner = MultiAgentTripPlanner()
    trip_plan = planner.plan_trip(request)
    status = getattr(planner, "last_generation_status", "unknown")
    message = getattr(planner, "last_generation_message", "") or "旅行计划生成完成"
    return trip_plan, status, message


def _refine_isolated(request: TripRequest, previous_plan: TripPlan, instruction: str):
    """在独立实例上做增量修改，返回 (trip_plan, status, message)。"""
    planner = MultiAgentTripPlanner()
    trip_plan = planner.refine_trip(request, previous_plan, instruction)
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

# SSE 推送参数：1 秒检查一次状态，15 秒发一次心跳，最多等 10 分钟
_SSE_POLL_INTERVAL = 1.0
_SSE_HEARTBEAT_SECONDS = 15.0
_SSE_MAX_SECONDS = 600.0


def _get_task_snapshot(task_id: str) -> dict | None:
    """持锁拷贝一份任务快照，避免返回过程中被后台线程改动。"""
    with _tasks_lock:
        task = _tasks.get(task_id)
        return dict(task) if task is not None else None


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
    """
    now = time.time()
    expired = [
        task_id
        for task_id, task in _tasks.items()
        if now - task.get("created_at", now) > _TASK_TTL_SECONDS
        and task.get("status") not in ("pending", "running")
    ]
    for task_id in expired:
        _tasks.pop(task_id, None)

    # 容量兜底：按创建顺序淘汰最旧的已完成任务
    if len(_tasks) >= _TASK_MAXSIZE:
        finished = [
            task_id
            for task_id, task in _tasks.items()
            if task.get("status") in ("done", "failed")
        ]
        for task_id in finished[: max(0, len(_tasks) - _TASK_MAXSIZE + 1)]:
            _tasks.pop(task_id, None)


def _new_task() -> str:
    task_id = uuid.uuid4().hex
    with _tasks_lock:
        _prune_tasks_locked()
        _tasks[task_id] = {
            "status": "pending",
            "success": None,
            "message": "",
            "result": None,
            "error": None,
            "created_at": time.time(),
        }
    return task_id


def _update_task(task_id: str, **fields) -> None:
    with _tasks_lock:
        if task_id in _tasks:
            _tasks[task_id].update(fields)


def _run_plan_task(task_id: str, request: TripRequest) -> None:
    """后台任务：独立实例生成计划，把结果写回任务表。

    后台任务跑在 worker 线程里，和事件循环里的同步接口共用同一个闸门，
    否则异步接口就成了绕过并发的后门（挂着几十个 task 一样能压垮进程）。
    """
    if not plan_gate.try_enter():
        metrics.record_planning_rejected()
        _update_task(
            task_id,
            status="failed",
            success=False,
            error=f"当前规划任务已达上限（{plan_gate.limit} 个），请稍后重试",
        )
        return
    _update_task(task_id, status="running")
    try:
        trip_plan, status, message = _plan_trip_isolated(request)
        _update_task(
            task_id,
            status="done",
            success=(status == "llm_success"),
            message=message,
            result=trip_plan.model_dump(mode="json"),
            # 异步任务也返回 plan_id，前端拿到结果后同样可以继续增量修改
            plan_id=_remember_plan(trip_plan, request),
        )
    except Exception as e:
        _update_task(task_id, status="failed", success=False, error=str(e))
    finally:
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
async def plan_trip(request: TripRequest):
    """
    生成旅行计划（同步接口）

    Args:
        request: 旅行请求参数

    Returns:
        旅行计划响应
    """
    try:
        print(f"\n{'='*60}")
        print("📥 收到旅行规划请求:")
        print(f"   城市: {request.city}")
        print(f"   日期: {request.start_date} - {request.end_date}")
        print(f"   天数: {request.travel_days}")
        print(f"{'='*60}\n")

        # === 改造点：不阻塞事件循环 ===
        # plan_trip 内部是同步阻塞调用（几十秒）。原代码直接在 async 路由里调用，
        # 单 worker 下会把整个事件循环堵死，其他请求（连 /health）都卡住。
        # 用 asyncio.to_thread 丢到线程池执行，事件循环立即可服务其他请求。
        # === 改造点：相同请求合并（single flight）===
        # 用户连点两次"生成"、或前端超时重试，会发出两个一模一样的请求。
        # 不合并就是两份 3 候选的账单，而且结果几乎一样。
        coalesce_key = _plan_request_key(request)
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
            try:
                trip_plan, generation_status, generation_message = await asyncio.to_thread(
                    _plan_trip_isolated, request
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
                    plan_id=_remember_plan(trip_plan, request),
                    coalesced=coalesced,
                ).model_dump(mode="json"),
            )

        print(f"✅ {generation_message}, 准备返回响应\n")
        # 改造点：返回 plan_id，前端拿着它就能"改一句话"做增量修改
        return TripPlanResponse(
            success=True,
            message=generation_message,
            data=trip_plan,
            plan_id=_remember_plan(trip_plan, request),
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
        # 同样用 to_thread：修改也是几十秒的同步阻塞调用
        new_plan, _status, message = await asyncio.to_thread(
            _refine_isolated, request, previous_plan, payload.instruction
        )
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
    """查询异步任务状态与结果（轮询方式）。"""
    snapshot = _get_task_snapshot(task_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return {"task_id": task_id, **snapshot}


@router.get(
    "/plan/async/{task_id}/stream",
    summary="订阅异步任务进度（SSE）",
    description="以 Server-Sent Events 推送状态变化，算完立刻推结果，前端无需轮询"
)
async def stream_plan_task(task_id: str):
    """SSE 推送任务进度。

    改造点（相对轮询）：轮询是"前端定时问"，空转请求多、结果最多晚一个间隔才看到；
    SSE 是"服务端状态一变就推"，一次连接拿到底，还能顺带发心跳保活。
    """
    if _get_task_snapshot(task_id) is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")

    async def event_generator():
        last_status = None
        started_at = time.time()
        last_heartbeat = time.time()
        while True:
            snapshot = _get_task_snapshot(task_id)
            if snapshot is None:
                yield _sse_frame({"task_id": task_id, "message": "任务已过期"}, "error")
                return

            status = snapshot.get("status")
            if status != last_status:
                last_status = status
                # 状态一变就推一帧，前端可以据此切换"排队中/生成中"
                yield _sse_frame({"task_id": task_id, "status": status}, "status")

            if status in ("done", "failed"):
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
