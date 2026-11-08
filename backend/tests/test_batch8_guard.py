"""批次8：成本与并发治理的回归测试（全部离线，不调真实 LLM）。

覆盖：
- 令牌桶的补充与拒绝语义（不需要 sleep，直接喂时间戳）
- 分级限流中间件：429、Retry-After、按 IP 隔离、探活豁免
- 规划并发闸门：满则 503、失败后必须归还名额
- 请求合并：相同指纹只跑一次，不同指纹不合并
- 增量修改的 lost update 保护：409 + 占用必须释放
"""

import asyncio
import threading
import time

from fastapi.testclient import TestClient

import app.api.routes.trip as trip_routes
from app.api import guard
from app.api.main import app
from app.api.observability import metrics
from app.models.schemas import TripPlan, TripRequest

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


def _fake_plan(city: str = "北京") -> TripPlan:
    return TripPlan(
        city=city,
        start_date="2026-11-10",
        end_date="2026-11-12",
        days=[],
        overall_suggestions="测试用计划",
    )


# === 令牌桶 ===

def test_bucket_denies_and_reports_wait_time():
    """桶空之后要明确告诉调用方等多久，而不是简单拒绝。"""
    bucket = guard.TokenBucket(capacity=2, refill_per_second=1.0)
    assert bucket.take(2)[0] is True
    allowed, retry_after, remaining = bucket.take(1)
    assert allowed is False
    assert remaining < 0.01  # 距上次取用只有几微秒，回补量可忽略
    # 缺 1 个令牌、每秒补 1 个 -> 约 1 秒
    assert 0.9 < retry_after <= 1.0


def test_bucket_refills_over_time():
    """按时间差补充：喂一个 future 时间戳即可，不用真的 sleep。"""
    bucket = guard.TokenBucket(capacity=2, refill_per_second=0.5)
    now = 1000.0
    bucket._updated = now
    assert bucket.take(2, now=now)[0] is True
    # 3 秒补 1.5 个，够再拿 1 个
    assert bucket.take(1, now=now + 3)[0] is True
    # 再要 1 个就不够了
    assert bucket.take(1, now=now + 3)[0] is False


def test_bucket_never_exceeds_capacity():
    """长时间空闲不能攒出无限令牌，否则等于没有限流。"""
    bucket = guard.TokenBucket(capacity=2, refill_per_second=10.0)
    now = 500.0
    bucket._updated = now
    bucket.take(2, now=now)
    _, _, remaining = bucket.take(1, now=now + 3600)
    assert remaining <= 1.0


# === 档位划分 ===

def test_classify_tiers():
    assert guard.classify("/api/trip/plan") == "plan"
    assert guard.classify("/api/trip/plan/async") == "plan"
    assert guard.classify("/api/trip/plan/abc123/refine") == "plan"
    assert guard.classify("/api/poi/photo") == "default"
    # 探活被限流会被负载均衡直接摘节点，必须豁免
    assert guard.classify("/health") is None
    assert guard.classify("/docs") is None


def test_classify_covers_every_real_health_path():
    """本项目有三个探活入口，只豁免 /health 会漏掉前端和监控真正用的那个。

    这个 bug 是实测发现的：连打 3 次 /api/trip/health，第 3 次返回 429。
    """
    assert guard.classify("/api/trip/health") is None
    assert guard.classify("/api/map/health") is None
    # 但带 health 字样的业务路径不能被误豁免
    assert guard.classify("/api/trip/health-report") == "default"


# === 中间件 ===

def _run_middleware(path: str, ip: str = "1.2.3.4"):
    """直接驱动 ASGI 中间件，绕开 HTTP 栈，快且不依赖具体路由。"""
    sent = []
    hits = {"n": 0}

    async def downstream(scope, receive, send):
        hits["n"] += 1
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "query_string": b"",
        "headers": [],
        "client": (ip, 4321),
        "server": ("testserver", 80),
        "scheme": "http",
        "http_version": "1.1",
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    middleware = guard.RateLimitMiddleware(downstream)
    asyncio.run(middleware(scope, receive, send))
    return sent, hits["n"]


def _status(sent) -> int:
    return sent[0]["status"]


def _header(sent, name: bytes):
    for key, value in sent[0].get("headers", []):
        if key == name:
            return value.decode()
    return None


def test_middleware_returns_429_with_retry_after(monkeypatch):
    monkeypatch.setattr(guard.CONFIG, "enabled", True)
    monkeypatch.setattr(guard.CONFIG, "default_burst", 2)
    monkeypatch.setattr(guard.CONFIG, "default_per_minute", 1)
    guard.reset_limiter()

    assert _status(_run_middleware("/api/poi/photo")[0]) == 200
    assert _status(_run_middleware("/api/poi/photo")[0]) == 200
    sent, hits = _run_middleware("/api/poi/photo")
    assert _status(sent) == 429
    assert hits == 0, "被限流的请求不应该进入下游处理"
    assert _header(sent, b"retry-after") is not None
    assert metrics.rate_limited_total >= 1


def test_middleware_isolates_by_ip(monkeypatch):
    """一个 IP 刷爆不能连累别人。"""
    monkeypatch.setattr(guard.CONFIG, "enabled", True)
    monkeypatch.setattr(guard.CONFIG, "default_burst", 1)
    monkeypatch.setattr(guard.CONFIG, "default_per_minute", 1)
    guard.reset_limiter()

    assert _status(_run_middleware("/api/poi/photo", ip="9.9.9.9")[0]) == 200
    assert _status(_run_middleware("/api/poi/photo", ip="9.9.9.9")[0]) == 429
    assert _status(_run_middleware("/api/poi/photo", ip="8.8.8.8")[0]) == 200


def test_middleware_exempts_health(monkeypatch):
    """探活不计限流：把桶调成 0 容量也不该拦。"""
    monkeypatch.setattr(guard.CONFIG, "enabled", True)
    monkeypatch.setattr(guard.CONFIG, "default_burst", 1)
    monkeypatch.setattr(guard.CONFIG, "default_per_minute", 1)
    guard.reset_limiter()
    for _ in range(5):
        assert _status(_run_middleware("/health")[0]) == 200


def test_middleware_disabled_passes_everything(monkeypatch):
    monkeypatch.setattr(guard.CONFIG, "enabled", False)
    for _ in range(10):
        assert _status(_run_middleware("/api/trip/plan")[0]) == 200


def test_plan_tier_is_charged_double(monkeypatch):
    """规划接口每次扣 2 个令牌，普通接口扣 1 个。"""
    monkeypatch.setattr(guard.CONFIG, "enabled", True)
    monkeypatch.setattr(guard.CONFIG, "plan_burst", 2)
    monkeypatch.setattr(guard.CONFIG, "plan_per_minute", 1)
    guard.reset_limiter()

    assert _status(_run_middleware("/api/trip/plan")[0]) == 200
    assert _status(_run_middleware("/api/trip/plan")[0]) == 429


# === 规划并发闸门 ===

def test_gate_rejects_when_saturated(monkeypatch):
    monkeypatch.setattr(
        trip_routes,
        "_plan_trip_isolated",
        lambda req, cancel_token=None: (_fake_plan(), "llm_success", "ok"),
    )
    guard.plan_gate.reset(limit=1)
    assert guard.plan_gate.try_enter() is True

    r = client.post("/api/trip/plan", json=SAMPLE)
    assert r.status_code == 503, r.text
    assert r.headers.get("retry-after") == "5"
    assert metrics.planning_rejected >= 1

    guard.plan_gate.leave()
    r2 = client.post("/api/trip/plan", json=SAMPLE)
    assert r2.status_code == 200, r2.text


def test_gate_released_even_when_generation_raises(monkeypatch):
    """失败路径必须归还名额，否则一次异常就把容量永久吃掉一格。"""

    def boom(req, cancel_token=None):
        raise RuntimeError("LLM 挂了")

    monkeypatch.setattr(trip_routes, "_plan_trip_isolated", boom)
    guard.plan_gate.reset(limit=1)

    r = client.post("/api/trip/plan", json=SAMPLE)
    assert r.status_code == 500
    assert guard.plan_gate.snapshot()["inflight"] == 0
    # 请求合并的 key 也必须出表，否则后续相同请求会等着一个永不完成的 future
    assert guard.single_flight.pending() == 0


# === 请求合并 ===

def test_duplicate_request_reuses_inflight_result(monkeypatch):
    """相同请求在飞行中时，后来者直接等结果，不再调一次 LLM。"""

    def should_not_run(req, cancel_token=None):
        raise AssertionError("重复请求不应该再触发一次生成")

    monkeypatch.setattr(trip_routes, "_plan_trip_isolated", should_not_run)

    key = trip_routes._plan_request_key(TripRequest(**SAMPLE))
    is_leader, future = guard.single_flight.claim(key)
    assert is_leader is True

    def finish_later():
        time.sleep(0.3)
        guard.single_flight.finish(
            key, future, result=(_fake_plan(), "llm_success", "合并命中")
        )

    thread = threading.Thread(target=finish_later)
    thread.start()
    r = client.post("/api/trip/plan", json=SAMPLE)
    thread.join(timeout=5)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["coalesced"] is True
    assert body["message"] == "合并命中"
    # 每个调用方都拿到自己的 plan_id，后续各自 refine 互不影响
    assert body["plan_id"]
    assert metrics.coalesced_requests >= 1


def test_inflight_failure_propagates_to_waiter(monkeypatch):
    """leader 失败时，等待方应该拿到 502，而不是挂死。"""

    def should_not_run(req, cancel_token=None):
        raise AssertionError("不应该自己跑")

    monkeypatch.setattr(trip_routes, "_plan_trip_isolated", should_not_run)
    key = trip_routes._plan_request_key(TripRequest(**SAMPLE))
    _, future = guard.single_flight.claim(key)

    def fail_later():
        time.sleep(0.3)
        guard.single_flight.finish(key, future, error=RuntimeError("上游超时"))

    thread = threading.Thread(target=fail_later)
    thread.start()
    r = client.post("/api/trip/plan", json=SAMPLE)
    thread.join(timeout=5)

    assert r.status_code == 502, r.text
    assert "上游超时" in r.text


def test_different_payload_is_not_coalesced(monkeypatch):
    """城市不同就是不同请求，绝不能合并。"""
    calls = []

    def fake(req, cancel_token=None):
        calls.append(req.city)
        return _fake_plan(city=req.city), "llm_success", "ok"

    monkeypatch.setattr(trip_routes, "_plan_trip_isolated", fake)

    a = dict(SAMPLE)
    b = dict(SAMPLE, city="上海")
    assert client.post("/api/trip/plan", json=a).status_code == 200
    assert client.post("/api/trip/plan", json=b).status_code == 200
    assert calls == ["北京", "上海"]


# === 增量修改的并发保护 ===

def test_concurrent_refine_returns_409(monkeypatch):
    monkeypatch.setattr(
        trip_routes,
        "_plan_trip_isolated",
        lambda req, cancel_token=None: (_fake_plan(), "llm_success", "ok"),
    )
    plan_id = client.post("/api/trip/plan", json=SAMPLE).json()["plan_id"]

    # 模拟另一轮修改正占着这个计划
    assert trip_routes._claim_plan(plan_id) == "ok"

    r = client.post(f"/api/trip/plan/{plan_id}/refine", json={"instruction": "改第二天"})
    assert r.status_code == 409, r.text
    assert metrics.refine_conflicts >= 1

    # 释放后应恢复正常
    trip_routes._release_plan(plan_id)
    assert trip_routes._claim_plan(plan_id) == "ok"
    trip_routes._release_plan(plan_id)


def test_refine_claim_released_after_failure(monkeypatch):
    """修改失败也不能把计划锁死，否则用户再也改不了这一版。"""
    monkeypatch.setattr(
        trip_routes,
        "_plan_trip_isolated",
        lambda req, cancel_token=None: (_fake_plan(), "llm_success", "ok"),
    )
    plan_id = client.post("/api/trip/plan", json=SAMPLE).json()["plan_id"]

    def boom(req, prev, instruction, cancel_token=None):
        raise RuntimeError("模型返回空")

    monkeypatch.setattr(trip_routes, "_refine_isolated", boom)
    r = client.post(f"/api/trip/plan/{plan_id}/refine", json={"instruction": "改第二天"})
    assert r.status_code == 502

    assert trip_routes._claim_plan(plan_id) == "ok"
    trip_routes._release_plan(plan_id)


def test_unknown_plan_id_still_404(monkeypatch):
    """不存在的 plan_id 仍然是 404，不能被并发保护改成 409。"""
    r = client.post("/api/trip/plan/not-a-real-id/refine", json={"instruction": "改"})
    assert r.status_code == 404


# === 指标暴露 ===

def test_metrics_exposes_cost_guard_counters():
    body = client.get("/metrics").json()
    for key in (
        "rate_limited_total",
        "rate_limited_by_tier",
        "planning_rejected",
        "coalesced_requests",
        "refine_conflicts",
        "planning_gate",
        "single_flight",
    ):
        assert key in body, f"/metrics 缺少 {key}"
    assert body["planning_gate"]["limit"] >= 1


# === 内存表的清理约束 ===

def test_prune_never_drops_running_task():
    """过期但还在跑的任务不能删：删了用户查不到结果，钱却已经花了。"""
    trip_routes._tasks.clear()
    now = time.time()
    trip_routes._tasks["running"] = {"status": "running", "created_at": now - 99999}
    trip_routes._tasks["done"] = {"status": "done", "created_at": now - 99999}

    with trip_routes._tasks_lock:
        trip_routes._prune_tasks_locked()

    assert "running" in trip_routes._tasks
    assert "done" not in trip_routes._tasks
    trip_routes._tasks.clear()


def test_task_table_has_capacity_limit():
    """任务表原本只有 TTL、没有容量上限，一小时内提交多少就堆多少。"""
    trip_routes._tasks.clear()
    for _ in range(trip_routes._TASK_MAXSIZE + 15):
        task_id = trip_routes._new_task()
        trip_routes._update_task(task_id, status="done")

    assert len(trip_routes._tasks) <= trip_routes._TASK_MAXSIZE
    trip_routes._tasks.clear()


def test_plan_eviction_skips_plan_under_refine():
    """容量淘汰要绕开正在被修改的计划，否则那一轮修改白做。"""
    trip_routes._plans.clear()
    now = time.time()
    trip_routes._plans["keep"] = {
        "plan": {},
        "request": {},
        "created_at": now,
        "refining": True,
    }
    for index in range(trip_routes._PLAN_MAXSIZE):
        trip_routes._plans[f"p{index}"] = {
            "plan": {},
            "request": {},
            "created_at": now,
            "refining": False,
        }

    with trip_routes._plans_lock:
        trip_routes._prune_plans_locked()

    assert "keep" in trip_routes._plans
    assert len(trip_routes._plans) <= trip_routes._PLAN_MAXSIZE
    trip_routes._plans.clear()
