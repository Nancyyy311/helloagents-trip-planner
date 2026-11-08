"""批次9：任务生命周期治理（取消 / 孤儿回收 / 断连 / 分级探活）。

这批改动的风险点很集中：**取消信号必须真的能穿透到花钱的那一行**，
而且不能被上层那些"兜底成 fallback 计划"的 except 悄悄吞掉。
所以用例的重点不是"接口返回 200"，而是：
- 取消之后 LLM 调用次数真的是 0；
- 抛出的是 PlanCancelled，而不是一份看起来正常的模板计划；
- 取消之后并发闸门被放掉（否则取消几次就把并发位漏光）。
"""

import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.api import guard, main as main_module
from app.api.main import app
from app.api.routes import trip as trip_routes
from app.agents import trip_planner_agent as agent_module
from app.models.schemas import TripRequest
from app.planner.cancel import CancelToken, PlanCancelled

client = TestClient(app)

SAMPLE = {
    "city": "北京",
    "start_date": "2026-11-10",
    "end_date": "2026-11-12",
    "travel_days": 3,
    "transportation": "公共交通",
    "accommodation": "经济型酒店",
    "preferences": ["文化历史"],
    "free_text_input": "",
    "party": {"adults": 1, "children": 0, "elders": 0, "total": 1, "companion_type": "solo"},
    "budget_constraint": {
        "amount": 3000,
        "scope": "total",
        "currency": "CNY",
        "budget_level": "standard",
        "strictness": "none",
    },
}


@pytest.fixture(autouse=True)
def _clean_stores():
    """任务表/计划表是进程级字典，用例之间必须清干净，否则串状态。"""
    with trip_routes._tasks_lock:
        trip_routes._tasks.clear()
    with trip_routes._plans_lock:
        trip_routes._plans.clear()
    yield
    with trip_routes._tasks_lock:
        trip_routes._tasks.clear()
    with trip_routes._plans_lock:
        trip_routes._plans.clear()


# --------------------------------------------------------------- CancelToken
def test_token_checkpoint_raises_only_after_cancel():
    token = CancelToken()
    token.checkpoint("before_llm")  # 没取消 -> 不抛
    assert token.cancelled is False

    assert token.cancel("client_deleted") is True
    assert token.cancelled is True
    # 复查一次 stage：取消点必须能被读出来，否则无法回答"停在哪一步"
    with pytest.raises(PlanCancelled) as info:
        token.checkpoint("before_llm:candidate_1")
    assert info.value.stage == "before_llm:candidate_1"
    assert info.value.reason == "client_deleted"
    assert token.stage == "before_llm:candidate_1"


def test_token_cancel_is_idempotent_and_keeps_first_reason():
    token = CancelToken()
    token.cancel("client_disconnected")
    token.cancel("orphan_reclaimed")
    # 第一次的原因更接近真实起因，后面的不该覆盖它
    assert token.reason == "client_disconnected"
    assert token.cancel("again") is False


def test_token_sleep_returns_immediately_on_cancel():
    """退避等待必须可被打断：原来 time.sleep(5) 会让"已取消"再白等 5 秒。"""
    token = CancelToken()
    threading.Timer(0.05, lambda: token.cancel("client_deleted")).start()
    started = time.perf_counter()
    with pytest.raises(PlanCancelled):
        token.sleep(5.0, "backoff:候选1")
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0, f"取消后仍睡了 {elapsed:.2f}s"
    assert token.interrupts == 1


# ------------------------------------------------- 取消穿透到 plan_trip 内部
class _FakeContextBuilder:
    """替换真实的 PlannerContextBuilder：不碰高德，只让流程能往前走。"""

    def collect(self, request):
        return {}

    def print_summary(self, context):
        pass

    def print_visualization(self, context):
        pass


def _bare_planner(token):
    """绕过 __init__ 造一个 planner：只装 plan_trip 会用到的那几个属性。

    不做真实初始化是为了不依赖 .env 里的 key；反过来也证明了一件事：
    取消这条路走不到任何网络调用。
    """
    planner = agent_module.MultiAgentTripPlanner.__new__(
        agent_module.MultiAgentTripPlanner
    )
    planner._cancel_token = token
    planner.planner_context_builder = _FakeContextBuilder()
    planner.amap_api_key = ""
    planner.planner_agent = object()
    planner.fallback_planner_agent = object()
    planner.planner_llm = object()
    planner.tool_llm = planner.planner_llm
    planner.last_generation_status = "idle"
    planner.last_generation_message = ""
    planner._tool_registry = None
    planner._tool_budget = None
    return planner


def test_plan_trip_aborts_before_llm_and_never_falls_back(monkeypatch):
    """核心验收：取消要抛 PlanCancelled，绝不能变成"200 + 模板计划"。

    这是本批最容易翻车的地方 —— plan_trip 最外层有个 `except Exception`
    会把任何异常翻译成 fallback 计划。取消一旦被它吞掉，用户会拿到一份
    看起来很正常的假计划，而账单已经产生。
    """
    token = CancelToken()
    planner = _bare_planner(token)

    llm_calls = []
    planner._call_llm_with_backoff = lambda label, call: llm_calls.append(label)

    def _fake_build_query(*args, **kwargs):
        # 在"上下文取完、马上就要发 LLM"的位置取消
        token.cancel("client_disconnected")
        return "planner-query"

    monkeypatch.setattr(agent_module, "build_planner_query", _fake_build_query)
    monkeypatch.setattr(
        agent_module, "build_amap_tool_registry", lambda key: (None, None)
    )

    with pytest.raises(PlanCancelled) as info:
        planner.plan_trip(TripRequest(**SAMPLE), cancel_token=token)

    assert info.value.reason == "client_disconnected"
    # 一个 LLM 调用都没发出去
    assert llm_calls == []
    # 状态如实标成 cancelled，而不是 llm_success / fallback_success
    assert planner.last_generation_status == "cancelled"


def test_plan_trip_with_pre_cancelled_token_does_no_network_work(monkeypatch):
    """提交后立刻取消：连高德上下文都不该去取（那也是有配额的）。"""
    token = CancelToken()
    token.cancel("user_deleted_task")
    planner = _bare_planner(token)

    collected = []
    planner.planner_context_builder = type(
        "B",
        (),
        {
            "collect": lambda self, req: collected.append(req) or {},
            "print_summary": lambda self, ctx: None,
            "print_visualization": lambda self, ctx: None,
        },
    )()

    with pytest.raises(PlanCancelled):
        planner.plan_trip(TripRequest(**SAMPLE), cancel_token=token)
    assert collected == []


def test_plan_trip_without_token_behaves_as_before():
    """没传令牌时 _checkpoint 是空操作，不能影响原有行为。"""
    planner = _bare_planner(None)
    planner._cancel_token = None
    planner._checkpoint("anything")  # 不抛即通过
    planner._sleep_or_cancel(0.01, "backoff")


def test_parallel_candidates_abort_fast_on_cancel(monkeypatch):
    """取消必须立刻生效，不能"等最慢的候选跑完再说"。

    原来用 `with ThreadPoolExecutor(...)`，退出时 shutdown(wait=True)，
    实测要 17 秒才变成 canceled，而且这 17 秒里并发闸门一直被占着——
    用户想换个城市重来，撞到的是 503。
    """
    token = CancelToken()
    planner = _bare_planner(token)
    # 每个"候选"耗时 2 秒，模拟真实的一次 LLM 调用
    planner._call_llm_with_backoff = lambda label, call: (time.sleep(2.0), "{}")[1]

    class _FakeSimpleAgent:
        def __init__(self, **kwargs):
            pass

        def run(self, *args, **kwargs):
            return "{}"

    monkeypatch.setattr(agent_module, "SimpleAgent", _FakeSimpleAgent)

    class _FakeAgent:
        llm = None

    threading.Timer(0.5, lambda: token.cancel("client_deleted")).start()
    started = time.perf_counter()
    with pytest.raises(PlanCancelled):
        planner._parallel_candidates(
            agent=_FakeAgent(),
            planner_query="q",
            request=TripRequest(**SAMPLE),
            label="test",
            planner_context={},
            target_candidate_count=3,
        )
    elapsed = time.perf_counter() - started
    # 等待粒度 0.4s：0.5s 取消，最迟约 0.9s 就该抛出来
    assert elapsed < 1.5, f"取消后仍等了 {elapsed:.1f}s，说明还在等候选跑完"


# ---------------------------------------------------------- 任务取消接口
def test_cancel_pending_task_is_immediate():
    """还没开始跑的任务：取消后立刻是终态，且不占并发闸门。"""
    task_id = trip_routes._new_task()
    r = client.delete(f"/api/trip/plan/async/{task_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cancel"] == "canceled"
    assert body["status"] == "canceled"

    # 后台线程即便随后被调度到，也必须直接退出
    trip_routes._run_plan_task(task_id, TripRequest(**SAMPLE))
    snapshot = trip_routes._get_task_snapshot(task_id)
    assert snapshot["status"] == "canceled"
    assert guard.plan_gate.snapshot()["inflight"] == 0


def test_cancel_running_task_stops_at_next_checkpoint_and_releases_gate(monkeypatch):
    """在跑的任务：只能等它在下一个检查点退出，退出后闸门必须被放掉。"""
    started = threading.Event()

    def slow_plan(req, cancel_token=None):
        started.set()
        for _ in range(400):
            # 模拟"每次要花一次 LLM 调用"的边界
            cancel_token.checkpoint("before_llm:candidate_1")
            time.sleep(0.01)
        return None, "llm_success", "不该跑到这里"

    monkeypatch.setattr(trip_routes, "_plan_trip_isolated", slow_plan)

    task_id = trip_routes._new_task()
    worker = threading.Thread(
        target=trip_routes._run_plan_task,
        args=(task_id, TripRequest(**SAMPLE)),
        daemon=True,
    )
    worker.start()
    assert started.wait(2.0), "后台任务没跑起来"

    r = client.delete(f"/api/trip/plan/async/{task_id}")
    assert r.status_code == 200, r.text
    assert r.json()["cancel"] == "canceling"

    worker.join(5.0)
    assert not worker.is_alive()

    snapshot = trip_routes._get_task_snapshot(task_id)
    assert snapshot["status"] == "canceled"
    assert snapshot["cancel_stage"] == "before_llm:candidate_1"
    # 闸门漏掉的话，取消两次就能把并发位占满，这是最隐蔽的副作用
    assert guard.plan_gate.snapshot()["inflight"] == 0


def test_cancel_finished_task_returns_409():
    """已经跑完的任务取消 -> 409，不假装成功（结果就在里面）。"""
    task_id = trip_routes._new_task()
    trip_routes._update_task(task_id, status="done", success=True)
    r = client.delete(f"/api/trip/plan/async/{task_id}")
    assert r.status_code == 409
    assert "已结束" in r.json()["detail"]


def test_cancel_missing_task_returns_404():
    r = client.delete("/api/trip/plan/async/does-not-exist")
    assert r.status_code == 404


# -------------------------------------------------------- 后台清扫 / 孤儿
def test_janitor_reaps_expired_task_without_new_submission():
    """修复：原来清理只挂在"提交新任务"上，没人提交就永远不清。"""
    task_id = trip_routes._new_task()
    trip_routes._update_task(task_id, status="done")
    with trip_routes._tasks_lock:
        trip_routes._tasks[task_id]["created_at"] = time.time() - 99999

    stats = trip_routes.sweep_task_store()
    assert stats["reaped"] >= 1
    assert trip_routes._get_task_snapshot(task_id) is None


def test_janitor_does_not_reap_canceling_task():
    """canceling 是"已喊停、线程还没退出"，此时删掉会让用户查不到结果。"""
    task_id = trip_routes._new_task()
    trip_routes._update_task(task_id, status="canceling")
    with trip_routes._tasks_lock:
        trip_routes._tasks[task_id]["created_at"] = time.time() - 99999

    trip_routes.sweep_task_store()
    assert trip_routes._get_task_snapshot(task_id) is not None


def test_orphan_task_is_reclaimed(monkeypatch):
    """提交后既没轮询也没订阅 -> 判定没人要了，取消掉别再烧钱。"""
    monkeypatch.setattr(trip_routes, "_TASK_ORPHAN_GRACE_SECONDS", 0.01)
    task_id = trip_routes._new_task()
    trip_routes._update_task(task_id, status="running")
    with trip_routes._tasks_lock:
        trip_routes._tasks[task_id]["last_seen"] = time.time() - 100

    stats = trip_routes.sweep_task_store()
    assert stats["abandoned"] == 1
    snapshot = trip_routes._get_task_snapshot(task_id)
    assert snapshot["cancel_reason"] == "orphan_reclaimed"
    assert snapshot["status"] in ("canceling", "canceled")


def test_watched_task_is_not_reclaimed(monkeypatch):
    """前端还在轮询的任务不能被当成孤儿——这正是 last_seen 存在的意义。"""
    monkeypatch.setattr(trip_routes, "_TASK_ORPHAN_GRACE_SECONDS", 5.0)
    task_id = trip_routes._new_task()
    trip_routes._update_task(task_id, status="running")

    client.get(f"/api/trip/plan/async/{task_id}")  # 轮询 = 心跳
    stats = trip_routes.sweep_task_store()
    assert stats["abandoned"] == 0
    assert trip_routes._get_task_snapshot(task_id)["status"] == "running"


def test_sse_subscriber_blocks_orphan_reclaim(monkeypatch):
    """SSE 订阅期间即使很久没状态变化也不能回收。"""
    monkeypatch.setattr(trip_routes, "_TASK_ORPHAN_GRACE_SECONDS", 0.01)
    task_id = trip_routes._new_task()
    trip_routes._update_task(task_id, status="running")
    trip_routes._subscribe_task(task_id)
    with trip_routes._tasks_lock:
        trip_routes._tasks[task_id]["last_seen"] = time.time() - 100

    stats = trip_routes.sweep_task_store()
    assert stats["abandoned"] == 0
    trip_routes._unsubscribe_task(task_id)


# --------------------------------------------------------------- 分级探活
def test_healthz_is_pure_liveness():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"


def test_readyz_ok_when_dependencies_configured(monkeypatch):
    monkeypatch.setattr(main_module.settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(main_module.settings, "amap_api_key", "amap-test")
    monkeypatch.setattr(main_module.settings, "use_personalized_planner", False)
    r = client.get("/readyz")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "ready"


def test_readyz_not_ready_without_llm_key(monkeypatch):
    """没有 LLM Key 时接了也必然失败 -> 503，让流量被摘走。"""
    monkeypatch.setattr(main_module.settings, "openai_api_key", "")
    r = client.get("/readyz")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "not_ready"
    assert "llm_key" in body["blocking"]


def test_readyz_degraded_still_200_without_amap_key(monkeypatch):
    """缺高德 Key 只是拿不到真实 POI，计划还能出 -> 200 + degraded，不摘流量。"""
    monkeypatch.setattr(main_module.settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(main_module.settings, "amap_api_key", "")
    monkeypatch.setattr(main_module.settings, "use_personalized_planner", False)
    r = client.get("/readyz")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "degraded"
    assert "amap_key" in body["degraded"]


def test_probes_are_exempt_from_rate_limit(monkeypatch):
    """探针被限流等于自己制造故障：连续 6 次探活必须全是 200。"""
    monkeypatch.setattr(guard.CONFIG, "enabled", True)
    guard.reset_limiter()
    codes = []
    for _ in range(6):
        codes.append(client.get("/readyz").status_code)
        codes.append(client.get("/healthz").status_code)
    assert set(codes) == {200}, codes
    monkeypatch.setattr(guard.CONFIG, "enabled", False)


# ------------------------------------------------------- 限流按成本分档
def test_task_read_endpoints_get_their_own_tier():
    """回归：轮询/SSE/取消不能被算进"规划"档。

    原来按 `/api/trip/plan/` 前缀一刀切，前端改成 1.2 秒轮询后，
    第 2 次轮询就撞上 plan 档的 6 次/分钟（实测 429）。
    真正该被严格限制的只有"会触发 LLM 调用"的那三个 POST。
    """
    assert guard.classify("/api/trip/plan/async/abc", "GET") == "poll"
    assert guard.classify("/api/trip/plan/async/abc/stream", "GET") == "poll"
    assert guard.classify("/api/trip/plan/async/abc", "DELETE") == "poll"
    assert guard.classify("/api/trip/plan", "POST") == "plan"
    assert guard.classify("/api/trip/plan/async", "POST") == "plan"
    assert guard.classify("/api/trip/plan/abc/refine", "POST") == "plan"


def test_polling_is_not_blocked_by_rate_limit(monkeypatch):
    """连续轮询 10 次进度必须全部 200（plan 档在同等次数下早就 429 了）。"""
    monkeypatch.setattr(guard.CONFIG, "enabled", True)
    guard.reset_limiter()
    task_id = trip_routes._new_task()
    codes = [client.get(f"/api/trip/plan/async/{task_id}").status_code for _ in range(10)]
    assert set(codes) == {200}, codes
    monkeypatch.setattr(guard.CONFIG, "enabled", False)


# ------------------------------------------------------------------- 指标
def test_metrics_expose_task_lifecycle_counters(monkeypatch):
    monkeypatch.setattr(trip_routes, "_TASK_ORPHAN_GRACE_SECONDS", 0.01)
    task_id = trip_routes._new_task()
    trip_routes._update_task(task_id, status="running")
    with trip_routes._tasks_lock:
        trip_routes._tasks[task_id]["last_seen"] = time.time() - 100
    trip_routes.sweep_task_store()
    r_cancel_reason = trip_routes._get_task_snapshot(task_id)["cancel_reason"]

    snapshot = client.get("/metrics").json()
    assert snapshot["tasks_abandoned"] >= 1
    # 取消原因要能从任务本身读出来（running 任务被取消时，指标落在线程退出那一刻，
    # 而"为什么取消"是调用当时就确定的，所以要看任务快照）
    assert r_cancel_reason == "orphan_reclaimed"
    assert "task_store" in snapshot
    assert "canceled" in snapshot["task_store"]["by_status"]
