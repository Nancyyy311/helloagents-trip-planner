"""批次14：预算审计口径的单测——防"同义反复"指标复活。

背景（这是本批次存在的唯一原因）
--------------------------------
归一化上线后，`budget_report_consistent_rate` 恒为 100%：
  normalize_budget() 把 budget.total 改成重算值
  → metrics 再拿 budget.total 和重算值比
  → 必然相等
这不是"模型变准了"，是"这个字段不再由模型决定"。把恒真命题当效果证据，
等于用坏掉的温度计宣布暖气修好了。所以：
  1. 留存 budget_reported_raw（模型原始值），让"纠正了多少"可度量；
  2. 新增 hidden_overrun（上报显示未超、实际超），这是用户会当场失信的错；
  3. 旧数据没有原始值时，报告必须写"无法度量"，不能默认成 0。
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "scripts"))

from audit_budget import audit_row  # noqa: E402
from app.eval import build_request  # noqa: E402
from app.eval.metrics import aggregate, evaluate_plan, render_markdown  # noqa: E402
from app.models.schemas import (  # noqa: E402
    Attraction,
    Budget,
    DayPlan,
    GroundingSummary,
    Location,
    Meal,
    TripPlan,
)


def _request(amount: int = 600):
    payload = {
        "city": "南京",
        "start_date": "2026-11-12",
        "end_date": "2026-11-12",
        "travel_days": 1,
        "transportation": "公共交通",
        "accommodation": "经济型酒店",
        "preferences": ["历史文化"],
        "free_text_input": "",
        "party": {"adults": 2, "total": 2, "companion_type": "couple"},
        "budget_constraint": {"amount": amount, "budget_level": "standard", "strictness": "soft"},
    }
    return build_request({"id": "t", "request": payload})


def _attraction(name: str, price: int) -> Attraction:
    return Attraction(
        name=name,
        address="南京市某路1号",
        location=Location(longitude=118.79, latitude=32.06),
        visit_duration=120,
        description="测试用景点",
        ticket_price=price,
    )


def _plan(shown_total: int, raw: int | None = None) -> TripPlan:
    """构造：交通 100 + 两个景点各 300 × 2 人 = 真实 1300。"""
    day = DayPlan(
        date="2026-11-12",
        day_index=0,
        description="测试行程",
        transportation="公共交通",
        accommodation="经济型酒店",
        attractions=[_attraction("中山陵", 300), _attraction("明孝陵", 300)],
        meals=[
            Meal(type="lunch", name="鸭血粉丝汤", estimated_cost=0),
            Meal(type="dinner", name="盐水鸭", estimated_cost=0),
        ],
    )
    return TripPlan(
        city="南京",
        start_date="2026-11-12",
        end_date="2026-11-12",
        days=[day],
        weather_info=[],
        overall_suggestions="测试",
        budget=Budget(total_attractions=600, total_hotels=0, total_meals=0,
                      total_transportation=100, total=shown_total),
        budget_reported_raw=raw,
        grounding=GroundingSummary(skipped=False, verified_rate=1.0,
                                   attractions_total=2, attractions_verified=2,
                                   meals_total=2, meals_verified=2),
    )


class TestHiddenOverrun:
    def test_detects_overrun_hidden_behind_low_report(self):
        """上报 500、预算 600、真实 1300：用户以为没超，实际超 700。"""
        result = evaluate_plan(_request(amount=600), _plan(shown_total=700, raw=500))
        budget = result["budget"]
        assert budget["recomputed_total"] == 1300
        assert budget["reported_raw"] == 500
        assert budget["raw_available"] is True
        assert budget["hidden_overrun"] is True
        assert budget["hidden_overrun_amount"] == 700

    def test_visible_overrun_is_not_hidden(self):
        """上报就已经超预算，属于"看得见的超支"，不算隐藏。"""
        result = evaluate_plan(_request(amount=600), _plan(shown_total=700, raw=650))
        assert result["budget"]["hidden_overrun"] is False

    def test_no_flag_when_raw_missing(self):
        """旧数据没有原始值：必须判 False，不能凭猜测报隐藏超支。"""
        result = evaluate_plan(_request(amount=600), _plan(shown_total=700, raw=None))
        budget = result["budget"]
        assert budget["raw_available"] is False
        assert budget["normalized"] is False
        assert budget["hidden_overrun"] is False

    def test_normalized_amount_counts_absolute_gap(self):
        result = evaluate_plan(_request(), _plan(shown_total=700, raw=480))
        budget = result["budget"]
        assert budget["normalized"] is True
        assert budget["normalized_delta"] == 220  # 700 - 480


class TestAggregateAndReport:
    def test_aggregate_sums_normalized_and_hidden(self):
        results = [
            evaluate_plan(_request(600), _plan(700, raw=500)),
            evaluate_plan(_request(600), _plan(700, raw=690)),
            evaluate_plan(_request(600), _plan(700, raw=None)),
        ]
        summary = aggregate(results)
        assert summary["budget_raw_available_cases"] == 2
        assert summary["budget_normalized_cases"] == 2
        assert summary["budget_normalized_amount"] == 210  # 200 + 10
        assert summary["budget_hidden_overrun_cases"] == 1
        assert summary["budget_hidden_overrun_max"] == 700

    def test_report_admits_when_unmeasurable(self):
        """全部旧数据时，报告不能把"没记录"写成"0 份被纠正"。"""
        results = [evaluate_plan(_request(600), _plan(700, raw=None))]
        markdown = render_markdown(aggregate(results), results)
        assert "未记录模型原始上报值" in markdown
        assert "恒真" in markdown  # consistency 字段必须自曝局限

    def test_report_shows_audit_when_measurable(self):
        results = [evaluate_plan(_request(600), _plan(700, raw=500))]
        markdown = render_markdown(aggregate(results), results)
        assert "1/1" in markdown
        assert "隐藏超支 **1** 份" in markdown


class TestAuditScript:
    def test_legacy_row_uses_total_as_raw(self):
        """旧格式（无 raw 字段）：budget.total 就是模型原始值，可直接审计。"""
        row = {
            "case_id": "legacy",
            "request": _request(600).model_dump(),
            "plan": _plan(shown_total=500, raw=None).model_dump(),
        }
        item = audit_row(row)
        assert item["raw_available"] is False
        assert item["raw_total"] == 500
        assert item["real_total"] == 1300
        assert item["diff"] == -800
        assert item["hidden_overrun"] is True

    def test_new_row_prefers_recorded_raw(self):
        """新格式：必须用 budget_reported_raw，而不是被覆盖后的 total。"""
        row = {
            "case_id": "new",
            "request": _request(600).model_dump(),
            "plan": _plan(shown_total=700, raw=500).model_dump(),
        }
        item = audit_row(row)
        assert item["raw_available"] is True
        assert item["raw_total"] == 500
        assert item["hidden_overrun"] is True
