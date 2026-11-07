"""批次2改造的回归测试（不调用真实 LLM）。

用 monkeypatch 把 `_plan_trip_isolated` 替换成假实现，
验证：同步接口成功/降级语义、异步任务提交与轮询流程。
"""

import time

from fastapi.testclient import TestClient

import app.api.routes.trip as trip_routes
from app.api.main import app
from app.models.schemas import TripPlan

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


def test_sync_plan_success(monkeypatch):
    """同步接口：LLM 成功 -> 200 + success=True。"""
    monkeypatch.setattr(
        trip_routes,
        "_plan_trip_isolated",
        lambda req: (_fake_plan(), "llm_success", "默认 Planner 生成成功"),
    )
    r = client.post("/api/trip/plan", json=SAMPLE)
    assert r.status_code == 200, r.text
    assert r.json()["success"] is True


def test_sync_plan_fallback_returns_502(monkeypatch):
    """改造点：LLM 失败走 fallback -> 502 + success=False（不再假成功）。"""
    monkeypatch.setattr(
        trip_routes,
        "_plan_trip_isolated",
        lambda req: (_fake_plan(), "fallback_success", "Planner失败，已返回fallback计划"),
    )
    r = client.post("/api/trip/plan", json=SAMPLE)
    assert r.status_code == 502, r.text
    body = r.json()
    assert body["success"] is False
    assert "fallback" in body["message"]


def test_async_task_flow(monkeypatch):
    """改造点：异步任务提交后能通过 task_id 轮询到 done + 结果。"""
    monkeypatch.setattr(
        trip_routes,
        "_plan_trip_isolated",
        lambda req: (_fake_plan(), "llm_success", "默认 Planner 生成成功"),
    )
    submit = client.post("/api/trip/plan/async", json=SAMPLE)
    assert submit.status_code == 200, submit.text
    task_id = submit.json()["task_id"]

    data = None
    for _ in range(100):
        r = client.get(f"/api/trip/plan/async/{task_id}")
        assert r.status_code == 200
        data = r.json()
        if data["status"] in ("done", "failed"):
            break
        time.sleep(0.05)

    assert data is not None
    assert data["status"] == "done"
    assert data["success"] is True
    assert data["result"]["city"] == "北京"
