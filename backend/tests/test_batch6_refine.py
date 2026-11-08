"""批次6回归测试（下）：多轮会话 / 增量修改。

不烧额度：用 monkeypatch 替换 `_refine_isolated`，只验证接口契约与 diff 逻辑。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import app.api.routes.trip as trip_routes
from app.api.main import app
from app.models.schemas import (
    Attraction,
    Budget,
    DayPlan,
    Hotel,
    Location,
    Meal,
    TripPlan,
)

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


def _trip(names_by_day, budget_total=1000, suggestions="原始建议") -> TripPlan:
    """构造一份可对比的计划。"""
    days = []
    for index, names in enumerate(names_by_day):
        days.append(
            DayPlan(
                date=f"2026-11-{10 + index:02d}",
                day_index=index,
                description=f"第{index + 1}天行程",
                transportation="公共交通",
                accommodation="经济型酒店",
                hotel=Hotel(name="同一家酒店", address="", estimated_cost=300),
                attractions=[
                    Attraction(
                        name=name,
                        address="",
                        location=Location(longitude=116.4, latitude=39.9),
                        visit_duration=120,
                        description="",
                    )
                    for name in names
                ],
                meals=[Meal(type="lunch", name=f"第{index + 1}天午餐", estimated_cost=50)],
            )
        )
    return TripPlan(
        city="北京",
        start_date="2026-11-10",
        end_date="2026-11-12",
        days=days,
        overall_suggestions=suggestions,
        budget=Budget(total=budget_total),
    )


# ------------------------------------------------------------- plan_id 下发
def test_plan_response_contains_plan_id(monkeypatch):
    """生成成功后必须返回 plan_id，否则前端没法发起增量修改。"""
    monkeypatch.setattr(
        trip_routes,
        "_plan_trip_isolated",
        lambda req, cancel_token=None: (_trip([["故宫"], ["天坛"], ["颐和园"]]), "llm_success", "生成成功"),
    )
    r = client.post("/api/trip/plan", json=SAMPLE)
    assert r.status_code == 200, r.text
    plan_id = r.json().get("plan_id")
    assert plan_id and len(plan_id) == 32


# ---------------------------------------------------------------- 增量修改
def test_refine_unknown_plan_returns_404():
    r = client.post("/api/trip/plan/not-a-real-id/refine", json={"instruction": "改一下"})
    assert r.status_code == 404


def test_refine_empty_instruction_rejected():
    """空修改要求应该在 schema 层就被拦掉，不要跑到 LLM 那里浪费额度。"""
    r = client.post("/api/trip/plan/whatever/refine", json={"instruction": ""})
    assert r.status_code == 422


def test_refine_only_changes_target_day(monkeypatch):
    """核心验收：只改第二天，diff 必须显示 1 天变更、2 天未变。"""
    original = _trip([["故宫"], ["天坛"], ["颐和园"]], budget_total=1000)
    refined = _trip([["故宫"], ["国家博物馆"], ["颐和园"]], budget_total=1000,
                    suggestions="本次已调整：第二天改为室内博物馆")

    monkeypatch.setattr(
        trip_routes,
        "_plan_trip_isolated",
        lambda req, cancel_token=None: (original, "llm_success", "生成成功"),
    )
    monkeypatch.setattr(
        trip_routes,
        "_refine_isolated",
        lambda req, prev, instruction, cancel_token=None: (refined, "llm_success", "增量修改成功"),
    )

    created = client.post("/api/trip/plan", json=SAMPLE).json()
    plan_id = created["plan_id"]

    r = client.post(f"/api/trip/plan/{plan_id}/refine", json={"instruction": "第二天改成室内"})
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["success"] is True
    diff = body["diff"]
    assert len(diff["changed_days"]) == 1
    assert diff["unchanged_days"] == 2
    changed = diff["changed_days"][0]
    assert changed["day_index"] == 1
    assert changed["before_attractions"] == ["天坛"]
    assert changed["after_attractions"] == ["国家博物馆"]
    assert diff["suggestions_changed"] is True
    # 新计划里第二天确实被替换了
    assert body["data"]["days"][1]["attractions"][0]["name"] == "国家博物馆"
    # 第一天、第三天保持不变
    assert body["data"]["days"][0]["attractions"][0]["name"] == "故宫"
    assert body["data"]["days"][2]["attractions"][0]["name"] == "颐和园"


def test_refine_reports_budget_delta(monkeypatch):
    original = _trip([["故宫"], ["天坛"], ["颐和园"]], budget_total=1000)
    refined = _trip([["故宫"], ["天坛"], ["颐和园"]], budget_total=1250)

    monkeypatch.setattr(
        trip_routes, "_plan_trip_isolated",
        lambda req, cancel_token=None: (original, "llm_success", "生成成功"),
    )
    monkeypatch.setattr(
        trip_routes, "_refine_isolated",
        lambda req, prev, instruction, cancel_token=None: (refined, "llm_success", "增量修改成功"),
    )
    plan_id = client.post("/api/trip/plan", json=SAMPLE).json()["plan_id"]

    diff = client.post(
        f"/api/trip/plan/{plan_id}/refine", json={"instruction": "预算提到 1250"}
    ).json()["diff"]
    assert diff["budget_before"] == 1000
    assert diff["budget_after"] == 1250
    assert diff["budget_delta"] == 250
    assert diff["changed_days"] == []
    assert diff["unchanged_days"] == 3


def test_refine_failure_returns_502_and_keeps_original(monkeypatch):
    """修改失败要明确 502，并且原计划依然可用（不能把上一版弄丢）。"""
    original = _trip([["故宫"], ["天坛"], ["颐和园"]])
    monkeypatch.setattr(
        trip_routes, "_plan_trip_isolated",
        lambda req, cancel_token=None: (original, "llm_success", "生成成功"),
    )

    def _boom(req, prev, instruction):
        raise RuntimeError("模型输出不合法")

    monkeypatch.setattr(trip_routes, "_refine_isolated", _boom)
    plan_id = client.post("/api/trip/plan", json=SAMPLE).json()["plan_id"]

    r = client.post(f"/api/trip/plan/{plan_id}/refine", json={"instruction": "随便改改"})
    assert r.status_code == 502
    assert "原计划未受影响" in r.json()["detail"]

    # 原 plan_id 仍然可以再改一次（说明存储没被破坏）
    monkeypatch.setattr(
        trip_routes, "_refine_isolated",
        lambda req, prev, instruction, cancel_token=None: (original, "llm_success", "增量修改成功"),
    )
    r2 = client.post(f"/api/trip/plan/{plan_id}/refine", json={"instruction": "再改一次"})
    assert r2.status_code == 200


def test_refine_supports_multiple_rounds(monkeypatch):
    """连续改多轮：返回的新 plan_id 能继续用于下一轮修改。"""
    v1 = _trip([["故宫"], ["天坛"], ["颐和园"]])
    v2 = _trip([["故宫"], ["国家博物馆"], ["颐和园"]])
    v3 = _trip([["故宫"], ["国家博物馆"], ["798艺术区"]])

    monkeypatch.setattr(trip_routes, "_plan_trip_isolated", lambda req, cancel_token=None: (v1, "llm_success", "ok"))
    monkeypatch.setattr(trip_routes, "_refine_isolated", lambda req, prev, i, cancel_token=None: (v2, "llm_success", "ok"))
    first = client.post("/api/trip/plan", json=SAMPLE).json()

    r1 = client.post(f"/api/trip/plan/{first['plan_id']}/refine", json={"instruction": "第二天改室内"})
    second = r1.json()
    assert second["plan_id"] != first["plan_id"]

    monkeypatch.setattr(trip_routes, "_refine_isolated", lambda req, prev, i, cancel_token=None: (v3, "llm_success", "ok"))
    r2 = client.post(f"/api/trip/plan/{second['plan_id']}/refine", json={"instruction": "第三天改艺术区"})
    assert r2.status_code == 200
    assert r2.json()["data"]["days"][2]["attractions"][0]["name"] == "798艺术区"


# ------------------------------------------------------------------- diff 单元
def test_diff_marks_new_days_as_changed():
    """天数变化时（理论上不该发生）diff 不能静默漏掉。"""
    before = _trip([["故宫"]])
    after = _trip([["故宫"], ["天坛"]])
    diff = trip_routes.diff_plans(before, after)
    assert len(diff.changed_days) == 1
    assert diff.changed_days[0].day_index == 1
    assert diff.unchanged_days == 1
