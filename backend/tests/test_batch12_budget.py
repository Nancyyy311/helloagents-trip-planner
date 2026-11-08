"""批次12：预算归一化与软预算惩罚的单测。

这两条是"钱"的防线：
* 总额不能采信模型自报（房间数/人数乘错是实测发生过的）；
* 软预算超支必须在 rerank 里扣分，否则"软约束"等于没约束。
"""

from __future__ import annotations

from app.eval import build_request
from app.models.schemas import Budget, DayPlan, Hotel, Location, Meal, TripPlan, Attraction
from app.planner.budget import normalize_budget
from app.planner.rerank import budget_arithmetic_consistent, score_rerank_metrics


def _request(**overrides):
    payload = {
        "city": "南京",
        "start_date": "2026-11-12",
        "end_date": "2026-11-13",
        "travel_days": 2,
        "transportation": "公共交通",
        "accommodation": "经济型酒店",
        "preferences": ["历史文化"],
        "free_text_input": "",
        "party": {"adults": 2, "total": 2, "companion_type": "couple"},
        "budget_constraint": {"amount": 1000, "budget_level": "standard", "strictness": "soft"},
    }
    payload.update(overrides)
    return build_request({"id": "t", "request": payload})


def _attraction(name: str, price: int = 0) -> Attraction:
    return Attraction(
        name=name,
        address="南京市某路1号",
        location=Location(longitude=118.79, latitude=32.06),
        visit_duration=120,
        description="测试用景点",
        ticket_price=price,
    )


def _plan(days: int = 2, budget_total: int = 1000) -> TripPlan:
    dates = [f"2026-11-{12 + i:02d}" for i in range(days)]
    day_plans = [
        DayPlan(
            date=dates[i],
            day_index=i,
            description="测试行程",
            transportation="公共交通",
            accommodation="经济型酒店",
            attractions=[_attraction("中山陵"), _attraction("明孝陵")],
            meals=[
                Meal(type="lunch", name="鸭血粉丝汤", estimated_cost=0),
                Meal(type="dinner", name="盐水鸭", estimated_cost=0),
            ],
        )
        for i in range(days)
    ]
    return TripPlan(
        city="南京",
        start_date=dates[0],
        end_date=dates[-1],
        days=day_plans,
        weather_info=[],
        overall_suggestions="测试",
        budget=Budget(total_transportation=budget_total, total=budget_total),
    )


class TestBudgetNormalize:
    def _plan_with_hotel(self, days: int = 2, price: int = 500):
        plan = _plan(days=days)
        for day in plan.days:
            day.hotel = Hotel(name="测试酒店", estimated_cost=price)
        plan.budget = Budget(
            total_attractions=0,
            total_hotels=price,  # 模型按 1 间房算的老值
            total_meals=0,
            total_transportation=100,
            total=price + 100,
        )
        return plan

    def test_hotel_total_uses_room_count(self):
        request = _request(
            travel_days=2,
            party={"adults": 3, "total": 3, "companion_type": "family"},
            budget_constraint={"amount": 3000, "budget_level": "standard"},
        )
        plan = self._plan_with_hotel(days=2, price=500)
        note = normalize_budget(plan, request)

        # 3 人 → 2 间房；2 晚 × 500 × 2 = 2000
        assert plan.budget.total_hotels == 2000
        assert plan.budget.total_transportation == 100
        assert budget_arithmetic_consistent(plan) is True
        assert note["changed"] is True
        assert note["after_total"] > note["before_total"]

    def test_total_matches_item_sum(self):
        request = _request(
            travel_days=2,
            party={"adults": 2, "total": 2, "companion_type": "couple"},
        )
        plan = self._plan_with_hotel(days=2, price=400)
        normalize_budget(plan, request)
        budget = plan.budget
        assert budget.total == (
            budget.total_attractions
            + budget.total_hotels
            + budget.total_meals
            + budget.total_transportation
        )

    def test_no_override_when_prices_are_zero(self):
        request = _request()
        plan = _plan()
        plan.budget = Budget(total_transportation=500, total=500)
        note = normalize_budget(plan, request)
        assert note["changed"] is False
        assert plan.budget.total == 500


class TestRerankBudgetPenalty:
    BASE = {
        "budget_hard_constraint_ok": True,
        "recomputed_budget_fit_ok": True,
        "budget_relationship_ok": True,
        "meal_cost_scale_ok": True,
        "meal_diversity_ok": True,
        "attraction_diversity_ok": True,
        "meal_grounding_rate": 1.0,
        "attraction_grounding_rate": 1.0,
        "hotel_grounding_rate": 1.0,
        "budget_arithmetic_consistent": True,
        "budget_fit_closeness": 1.0,
    }

    def test_overshoot_costs_score(self):
        without = score_rerank_metrics(dict(self.BASE))
        with_30 = score_rerank_metrics({**self.BASE, "budget_overshoot_ratio": 0.3})
        assert round(without - with_30, 4) == 30.0

    def test_penalty_is_capped(self):
        capped = score_rerank_metrics({**self.BASE, "budget_overshoot_ratio": 2.0})
        at_cap = score_rerank_metrics({**self.BASE, "budget_overshoot_ratio": 0.3})
        assert capped == at_cap
