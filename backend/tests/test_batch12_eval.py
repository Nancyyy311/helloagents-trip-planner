"""批次12：评测口径与约束检查的单测。

重点不是"覆盖率"，而是**口径不能漂**：
评测说通过/不通过的依据，必须能在离线、无 LLM 的情况下复现。
"""

from __future__ import annotations

from app.eval import build_request, load_cases
from app.eval.metrics import aggregate, evaluate_plan, render_markdown
from app.models.schemas import (
    Attraction,
    Budget,
    DayPlan,
    GroundingSummary,
    Location,
    Meal,
    TripPlan,
)
from app.planner.constraints import expand_terms, find_constraint_violations


def _request(**overrides):
    payload = {
        "city": "南京",
        "start_date": "2026-11-12",
        "end_date": "2026-11-13",
        "travel_days": 2,
        "transportation": "公共交通",
        "accommodation": "经济型酒店",
        "preferences": ["历史文化"],
        "free_text_input": "不要购物点",
        "party": {"adults": 2, "total": 2, "companion_type": "couple"},
        "budget_constraint": {"amount": 1000, "budget_level": "standard", "strictness": "soft"},
    }
    payload.update(overrides)
    return build_request({"id": "t", "request": payload})


def _attraction(name: str, price: int = 0, verified: bool = True) -> Attraction:
    return Attraction(
        name=name,
        address="南京市某路1号",
        location=Location(longitude=118.79, latitude=32.06),
        visit_duration=120,
        description="测试用景点",
        ticket_price=price,
        verified=verified,
    )


def _plan(
    names_per_day=None,
    meal_names=None,
    budget_total: int = 1000,
    dates=("2026-11-12", "2026-11-13"),
    grounding_rate: float = 1.0,
) -> TripPlan:
    names_per_day = names_per_day or [["中山陵", "明孝陵"], ["总统府", "南京博物院"]]
    meal_names = meal_names or [["鸭血粉丝汤", "盐水鸭"], ["小笼包", "牛肉锅贴"]]
    days = []
    for index, date in enumerate(dates):
        days.append(
            DayPlan(
                date=date,
                day_index=index,
                description="测试行程",
                transportation="公共交通",
                accommodation="经济型酒店",
                attractions=[_attraction(name) for name in names_per_day[index]],
                meals=[
                    Meal(type="lunch", name=meal_names[index][0], estimated_cost=0),
                    Meal(type="dinner", name=meal_names[index][1], estimated_cost=0),
                ],
            )
        )
    return TripPlan(
        city="南京",
        start_date=dates[0],
        end_date=dates[-1],
        days=days,
        weather_info=[],
        overall_suggestions="测试",
        budget=Budget(total_transportation=budget_total, total=budget_total),
        grounding=GroundingSummary(
            skipped=False,
            verified_rate=grounding_rate,
            attractions_total=4,
            attractions_verified=round(4 * grounding_rate),
            meals_total=4,
            meals_verified=round(4 * grounding_rate),
        ),
    )


class TestConstraintRules:
    def test_shopping_term_expands(self):
        terms = expand_terms(["购物点"])
        assert "购物中心" in terms and "商业街" in terms

    def test_flags_shopping_mall(self):
        plan = _plan(names_per_day=[["中山陵", "某某购物中心"], ["总统府", "南京博物院"]])
        violations = find_constraint_violations(plan, ["购物点"])
        assert len(violations) == 1
        assert violations[0]["name"] == "某某购物中心"
        assert violations[0]["day"] == 1

    def test_flags_spicy_food(self):
        plan = _plan(meal_names=[["麻辣香锅", "盐水鸭"], ["小笼包", "牛肉锅贴"]])
        violations = find_constraint_violations(plan, ["辣"])
        assert [item["name"] for item in violations] == ["麻辣香锅"]

    def test_no_false_positive_on_museum(self):
        plan = _plan()
        assert find_constraint_violations(plan, ["购物点", "辣"]) == []

    def test_empty_terms_short_circuit(self):
        plan = _plan()
        assert find_constraint_violations(plan, []) == []


class TestMetrics:
    def test_clean_plan_passes(self):
        result = evaluate_plan(_request(), _plan())
        assert result["structure"]["ok"] is True
        assert result["constraints"]["violation_count"] == 0
        assert result["budget"]["over"] is False
        assert result["grounding"]["rate"] == 1.0
        assert result["passed"] is True

    def test_thin_day_fails_structure(self):
        plan = _plan(names_per_day=[["中山陵"], ["总统府", "南京博物院"]])
        result = evaluate_plan(_request(), plan)
        assert result["structure"]["ok"] is False
        assert result["structure"]["thin_days"] == [1]
        assert result["passed"] is False

    def test_date_mismatch_detected(self):
        plan = _plan(dates=("2026-11-20", "2026-11-21"))
        result = evaluate_plan(_request(), plan)
        assert result["structure"]["dates_ok"] is False

    def test_budget_over_tolerance(self):
        plan = _plan(budget_total=1500)
        result = evaluate_plan(_request(), plan)
        assert result["budget"]["deviation"] == 0.5
        assert result["budget"]["over"] is True
        assert result["passed"] is False

    def test_duplicate_attraction_across_days(self):
        plan = _plan(names_per_day=[["中山陵", "中山陵"], ["总统府", "南京博物院"]])
        result = evaluate_plan(_request(), plan)
        assert result["duplicates"]["duplicate_count"] == 1
        assert result["duplicates"]["repeated_attractions"] == {"中山陵": 2}

    def test_grounding_falls_back_to_verified_fields(self):
        plan = _plan()
        plan.grounding = None  # 强制走 verified 字段回退路径
        plan.days[0].attractions[1].verified = False
        result = evaluate_plan(_request(), plan)
        assert result["grounding"]["rate"] < 1.0
        assert "明孝陵" in result["grounding"]["unverified"]

    def test_violation_fails_case(self):
        request = _request()
        plan = _plan(names_per_day=[["中山陵", "某某购物中心"], ["总统府", "南京博物院"]])
        result = evaluate_plan(request, plan)
        assert result["passed"] is False
        assert "购物" in result["constraints"]["messages"][0]


class TestAggregate:
    def test_aggregate_rates(self):
        good = evaluate_plan(_request(), _plan())
        bad = evaluate_plan(
            _request(),
            _plan(names_per_day=[["中山陵", "某某购物中心"], ["总统府", "南京博物院"]], budget_total=2000),
        )
        summary = aggregate([good, bad])
        assert summary["cases"] == 2
        assert summary["pass_rate"] == 0.5
        assert summary["constraint_violation_case_rate"] == 0.5
        assert summary["budget_over_rate"] == 0.5

    def test_markdown_contains_failure_reasons(self):
        bad = evaluate_plan(_request(), _plan(budget_total=2000))
        markdown = render_markdown(aggregate([bad]), [bad])
        assert "评测报告" in markdown
        assert "超预算" in markdown


class TestCaseSet:
    def test_cases_load(self):
        cases = load_cases()
        assert len(cases) >= 12
        ids = [case["id"] for case in cases]
        assert len(ids) == len(set(ids)), "用例 id 不能重复"

    def test_every_case_builds_valid_request(self):
        for case in load_cases():
            request = build_request(case)
            assert request.travel_days >= 1
            assert request.party.total == request.party.adults + request.party.children + request.party.elders

    def test_negative_constraint_cases_exist(self):
        """评测集必须真的包含"有约束"的用例，否则约束维度等于没测。"""
        constrained = 0
        for case in load_cases():
            request = build_request(case)
            if request.free_text_input and any(
                marker in request.free_text_input for marker in ("不要", "不吃", "别", "少")
            ):
                constrained += 1
        assert constrained >= 6
