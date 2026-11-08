"""批次8：上下文缓存击穿（cache stampede）保护的回归测试。

不联网：把三个取数函数换成计数桩，只验证"同一 key 只取一遍"。
"""

import threading
import time

import pytest

from app.models.schemas import TripRequest
from app.planner import context as ctx

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
def _clear_context_cache():
    ctx._context_cache._store.clear()
    yield
    ctx._context_cache._store.clear()


def _instrument(builder, calls, lock, delay=0.1):
    def make(name):
        def inner(request):
            with lock:
                calls.append(name)
            time.sleep(delay)
            return {"tool_snapshot": {name: []}, "status": "ok"}

        return inner

    # 方法名是单数（_collect_attraction_snapshot / _collect_hotel_snapshot），
    # 不能按数据源名字拼，否则会静默替换不到、真去打高德
    for member, name in (
        ("_collect_attraction_snapshot", "attractions"),
        ("_collect_hotel_snapshot", "hotels"),
        ("_collect_weather_snapshot", "weather"),
    ):
        setattr(builder, member, make(name))


def test_same_key_fetches_only_once():
    """5 个线程同时查同一城市，只应该出现 3 次取数（每个数据源一次）。

    没有保护时是 5 × 3 = 15 次，缓存等于白设。
    """
    builder = ctx.PlannerContextBuilder("fake-key")
    calls: list[str] = []
    lock = threading.Lock()
    _instrument(builder, calls, lock)

    request = TripRequest(**SAMPLE)
    results: list[dict] = []

    def worker():
        results.append(builder.collect(request))

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(results) == 5, "所有线程都必须拿到结果，不能有人卡死"
    assert len(calls) == 3, f"应该只取一遍，实际取了 {len(calls)} 次"
    # 其他线程要么等到锁后复用，要么直接命中缓存，两种都算省下
    stats = ctx.context_cache_stats()
    assert stats["stampede_waits"] + stats["hits"] >= 1


def test_different_city_is_not_serialized_into_one_fetch():
    """不同城市是不同 key，各自的取数都要发生。"""
    builder = ctx.PlannerContextBuilder("fake-key")
    calls: list[str] = []
    lock = threading.Lock()
    _instrument(builder, calls, lock, delay=0.01)

    beijing = TripRequest(**SAMPLE)
    shanghai = TripRequest(**dict(SAMPLE, city="上海"))

    builder.collect(beijing)
    builder.collect(shanghai)

    assert len(calls) == 6, f"两个城市应各取一遍（3+3），实际 {len(calls)} 次"


def test_second_hit_does_not_refetch():
    """缓存命中后不应再取数。"""
    builder = ctx.PlannerContextBuilder("fake-key")
    calls: list[str] = []
    lock = threading.Lock()
    _instrument(builder, calls, lock, delay=0.01)

    request = TripRequest(**SAMPLE)
    builder.collect(request)
    builder.collect(request)

    assert len(calls) == 3
    assert ctx.context_cache_stats()["hits"] >= 1


def test_stats_exposes_stampede_counter():
    assert "stampede_waits" in ctx.context_cache_stats()
