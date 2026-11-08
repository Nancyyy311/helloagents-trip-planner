"""批次3改造的回归测试（不调用真实 LLM、不消耗额度）。

覆盖：
- PlannerContext 缓存的 TTL 过期 / LRU 容量上限 / 命中率统计
- /metrics 里新增的缓存与 LLM 指标字段
- LLM 调用的"可重试错误"分类与指数退避重试
- SSE 流式推送能推状态并推最终结果
"""

import time

import pytest
from fastapi.testclient import TestClient

import app.agents.trip_planner_agent as agent_mod
import app.api.routes.trip as trip_routes
from app.api.main import app
from app.models.schemas import TripPlan
from app.planner.context import _TTLCache

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


def _fake_plan() -> TripPlan:
    return TripPlan(
        city="北京",
        start_date="2026-11-10",
        end_date="2026-11-12",
        days=[],
        overall_suggestions="测试用计划",
    )


# ---------------- 缓存：TTL / LRU / 命中率 ----------------

def test_ttl_cache_expires():
    """TTL=0 时写入即过期，get 返回 None 且计入 miss。"""
    cache = _TTLCache(ttl_seconds=0, maxsize=4)
    cache.set("k", {"v": 1})
    assert cache.get("k") is None
    assert cache.stats()["misses"] == 1


def test_ttl_cache_lru_eviction():
    """超过 maxsize 时淘汰最久未使用的 key（访问会刷新位置）。"""
    cache = _TTLCache(ttl_seconds=60, maxsize=2)
    cache.set("a", {"v": "a"})
    cache.set("b", {"v": "b"})
    assert cache.get("a") == {"v": "a"}  # 访问 a，a 变成"最近使用"
    cache.set("c", {"v": "c"})           # 超限 -> 应淘汰最久未使用的 b

    assert cache.get("b") is None
    assert cache.get("a") == {"v": "a"}
    assert cache.get("c") == {"v": "c"}

    stats = cache.stats()
    assert stats["size"] == 2
    assert stats["maxsize"] == 2
    assert stats["evictions"] >= 1
    assert stats["hits"] >= 2


def test_ttl_cache_returns_copy():
    """get 返回深拷贝：调用方改返回值不能污染缓存。"""
    cache = _TTLCache(ttl_seconds=60, maxsize=4)
    cache.set("k", {"nested": {"v": 1}})
    got = cache.get("k")
    got["nested"]["v"] = 999
    assert cache.get("k")["nested"]["v"] == 1


# ---------------- /metrics 新指标 ----------------

def test_metrics_includes_cache_and_llm():
    """/metrics 应同时暴露请求指标、LLM 成功率、缓存命中率。"""
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "llm_success_rate" in body
    assert "llm_calls" in body
    assert "planner_context_cache" in body
    cache_stats = body["planner_context_cache"]
    for key in ("size", "maxsize", "hits", "misses", "hit_rate", "evictions"):
        assert key in cache_stats


# ---------------- LLM 退避重试 ----------------

def _dummy_planner():
    """跳过 __init__ 造一个空外壳，只测重试逻辑（不初始化真实模型）。"""
    return agent_mod.MultiAgentTripPlanner.__new__(agent_mod.MultiAgentTripPlanner)


def test_retryable_error_classification():
    planner = _dummy_planner()
    # 限流 / 5xx / 连接重置 -> 可重试
    assert planner._is_retryable_llm_error(RuntimeError("Error code: 429 rate limit"))
    assert planner._is_retryable_llm_error(RuntimeError("503 service unavailable"))
    assert planner._is_retryable_llm_error(RuntimeError("connection reset by peer"))
    # 内容风控 / 参数错误 -> 重试也没用，不重试
    assert not planner._is_retryable_llm_error(RuntimeError("content exists risk"))
    assert not planner._is_retryable_llm_error(RuntimeError("invalid_request_error 400"))
    assert not planner._is_retryable_llm_error(RuntimeError("something weird"))


def test_backoff_retries_then_succeeds(monkeypatch):
    """前两次 429、第三次成功 -> 应重试并最终返回结果。"""
    monkeypatch.setattr(agent_mod, "PLANNER_LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(agent_mod, "PLANNER_LLM_RETRY_BASE_DELAY", 0.0)
    planner = _dummy_planner()

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 too many requests")
        return "ok"

    assert planner._call_llm_with_backoff("test", flaky) == "ok"
    assert calls["n"] == 3


def test_backoff_gives_up_on_non_retryable(monkeypatch):
    """不可重试错误应立即抛出，不做无意义等待。"""
    monkeypatch.setattr(agent_mod, "PLANNER_LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(agent_mod, "PLANNER_LLM_RETRY_BASE_DELAY", 0.0)
    planner = _dummy_planner()

    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise RuntimeError("content exists risk")

    with pytest.raises(RuntimeError):
        planner._call_llm_with_backoff("test", bad)
    assert calls["n"] == 1


def test_backoff_exhausts_after_max_retries(monkeypatch):
    """一直 429 -> 重试到上限后抛出，调用次数 = 1 + 重试次数。"""
    monkeypatch.setattr(agent_mod, "PLANNER_LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(agent_mod, "PLANNER_LLM_RETRY_BASE_DELAY", 0.0)
    planner = _dummy_planner()

    calls = {"n": 0}

    def always_429():
        calls["n"] += 1
        raise RuntimeError("429 too many requests")

    with pytest.raises(RuntimeError):
        planner._call_llm_with_backoff("test", always_429)
    assert calls["n"] == 3  # 首次 + 2 次重试


# ---------------- SSE 流式推送 ----------------

def _submit_and_wait() -> str:
    submit = client.post("/api/trip/plan/async", json=SAMPLE)
    assert submit.status_code == 200, submit.text
    task_id = submit.json()["task_id"]
    for _ in range(100):
        data = client.get(f"/api/trip/plan/async/{task_id}").json()
        if data["status"] in ("done", "failed"):
            return task_id
        time.sleep(0.05)
    raise AssertionError("任务未在预期时间内完成")


def test_sse_stream_pushes_status_and_result(monkeypatch):
    """SSE 连接应收到 event-stream、状态帧与最终结果帧。"""
    monkeypatch.setattr(
        trip_routes,
        "_plan_trip_isolated",
        lambda req, cancel_token=None: (_fake_plan(), "llm_success", "默认 Planner 生成成功"),
    )
    task_id = _submit_and_wait()

    with client.stream("GET", f"/api/trip/plan/async/{task_id}/stream") as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = "".join(resp.iter_text())

    assert "event: status" in body
    assert "event: result" in body
    assert '"status": "done"' in body


def test_sse_unknown_task_404():
    """不存在的任务订阅 SSE 应直接 404，而不是挂着不返回。"""
    r = client.get("/api/trip/plan/async/does-not-exist/stream")
    assert r.status_code == 404
