"""行程质量的离线度量。

目标：把"感觉这版比上版好"换成可复现的数字。

五个维度，全部可以从**已落盘的一份 TripPlan + 原始请求**算出来，
不调 LLM、不调高德——所以同一批结果可以反复重算、口径永不变。

1. 结构合法：天数/日期/每日景点数/每日两餐/预算算术自洽
2. 真实性：grounding 命中率（LLM 编的 POI 有没有被抓出来）
3. 预算：按 POI 价格重算后的总额 vs 用户预算的偏差
4. 约束：负向约束（不要购物/不吃辣/少走路…）被违反了几条
5. 重复：同一景点跨天重复出现（凑数的最明显信号）
"""

from __future__ import annotations

from datetime import datetime, timedelta
from statistics import median
from typing import Any, Dict, List, Optional, Sequence

from ..models.schemas import TripPlan, TripRequest
from ..planner.constraints import find_constraint_violations, summarize_violations
from ..planner.policy import build_preference_profile
from ..planner.rerank import (
    budget_arithmetic_consistent,
    recompute_budget_from_selected_items,
)

DEFAULT_MIN_ATTRACTIONS_PER_DAY = 2
DEFAULT_BUDGET_OVER_TOLERANCE = 0.15
# 单景点当天游览时长达到这个值，视为"远郊/全天型"（石林、长城、环球影城），
# 只排一个景点是合理设计，不是偷懒。
FULL_DAY_VISIT_MINUTES = 240
# 上报预算与重算预算的允许误差：2% 以内算一致（四舍五入 + 门票取整会有一两块钱差）
BUDGET_REPORT_TOLERANCE = 0.02


def _expected_dates(request: TripRequest) -> List[str]:
    """按 start_date 推导每一天应有的日期字符串。"""
    try:
        start = datetime.strptime(request.start_date, "%Y-%m-%d")
    except (TypeError, ValueError):
        return []
    return [
        (start + timedelta(days=offset)).strftime("%Y-%m-%d")
        for offset in range(request.travel_days)
    ]


def _grounding_stats(trip_plan: TripPlan) -> Dict[str, Any]:
    """优先用响应里自带的 grounding 摘要，缺失时回退到逐条 verified 字段。"""
    summary = trip_plan.grounding
    if summary is not None:
        return {
            "skipped": bool(summary.skipped),
            "rate": float(summary.verified_rate or 0.0),
            "total": summary.attractions_total + summary.meals_total + summary.hotels_total,
            "verified": summary.attractions_verified + summary.meals_verified + summary.hotels_verified,
            "unverified": list(summary.unverified_attractions or []),
        }

    total = verified = 0
    unverified: List[str] = []
    for day in trip_plan.days:
        for attraction in day.attractions:
            if attraction.verified is None:
                continue  # 无法校验 ≠ 编造，不能算进分母
            total += 1
            if attraction.verified:
                verified += 1
            else:
                unverified.append(attraction.name)
        for meal in day.meals:
            if meal.verified is None:
                continue
            total += 1
            if meal.verified:
                verified += 1
        if day.hotel is not None and day.hotel.verified is not None:
            total += 1
            if day.hotel.verified:
                verified += 1
    return {
        "skipped": total == 0,
        "rate": round(verified / total, 4) if total else 0.0,
        "total": total,
        "verified": verified,
        "unverified": unverified,
    }


def _duplicate_stats(trip_plan: TripPlan) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for day in trip_plan.days:
        for attraction in day.attractions:
            name = str(attraction.name or "").strip()
            if name:
                counts[name] = counts.get(name, 0) + 1
    repeated = {name: count for name, count in counts.items() if count > 1}
    return {"repeated_attractions": repeated, "duplicate_count": len(repeated)}


def _thin_days(trip_plan: TripPlan, min_attractions: int) -> List[int]:
    """找出"景点明显不够"的天（返回 1 起的序号）。

    第一版规则是"每天必须 ≥2 个景点"，跑完 18 条才发现这是把尺子量歪了：
    抵达日、返程日、以及石林/长城这种远郊全天行程，本来就只该排 1 个。
    按老规则它们全被判失败（3/18），逼着模型往第一天和最后一天硬塞景点。
    现在的豁免条件写死在这里，评审时能一条条讲清楚：

    * 首日/末日各允许只排 1 个（抵达、返程）；
    * 只有 1 个但建议游览 ≥ 240 分钟，视为远郊/全天型；
    * 其余中间天仍要求 ≥2 个，0 个景点的天一律算不合格。
    """
    last_index = len(trip_plan.days) - 1
    thin: List[int] = []
    for index, day in enumerate(trip_plan.days):
        count = len(day.attractions)
        if count >= min_attractions:
            continue
        if count == 0:
            thin.append(day.day_index + 1)
            continue
        is_edge_day = index in (0, last_index) and len(trip_plan.days) >= 3
        is_full_day = count == 1 and day.attractions[0].visit_duration >= FULL_DAY_VISIT_MINUTES
        if is_edge_day or is_full_day:
            continue
        thin.append(day.day_index + 1)
    return thin


def evaluate_plan(
    request: TripRequest,
    trip_plan: TripPlan,
    case: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """对单份行程打分，返回结构化结果（可直接序列化成 JSON）。"""
    case = case or {}
    expect = case.get("expect") or {}
    min_attractions = int(expect.get("min_attractions_per_day") or DEFAULT_MIN_ATTRACTIONS_PER_DAY)
    over_tolerance = float(expect.get("budget_over_tolerance") or DEFAULT_BUDGET_OVER_TOLERANCE)

    # --- 1. 结构 ---
    expected_dates = _expected_dates(request)
    actual_dates = [day.date for day in trip_plan.days]
    days_count_ok = len(trip_plan.days) == request.travel_days
    day_index_ok = [day.day_index for day in trip_plan.days] == list(range(len(trip_plan.days)))
    dates_ok = bool(expected_dates) and actual_dates == expected_dates

    thin_days = _thin_days(trip_plan, min_attractions)
    missing_meals = [
        day.day_index + 1
        for day in trip_plan.days
        if len({str(meal.type).lower() for meal in day.meals} & {"lunch", "dinner"}) < 2
    ]
    budget_arithmetic_ok = budget_arithmetic_consistent(trip_plan)

    structure_ok = (
        days_count_ok and day_index_ok and dates_ok
        and not thin_days and not missing_meals and budget_arithmetic_ok
    )

    # --- 2. 真实性 ---
    grounding = _grounding_stats(trip_plan)

    # --- 3. 预算 ---
    recomputed = recompute_budget_from_selected_items(trip_plan, request)
    target = int(request.budget_constraint.amount or 0)
    budget_total = int(recomputed.get("total") or 0)
    reported_total = int((trip_plan.budget.total if trip_plan.budget else 0) or 0)
    # 上报预算和按 POI 单价重算的预算必须对得上。"对不上"通常意味着
    # 模型把房间数/人数乘错了——这是最容易被用户当场发现的那种错。
    #
    # 注意：归一化上线后 budget.total 已经是重算值，这里必然相等。
    # 所以这个字段只能当"接线检查"（证明归一化真的挂上了），
    # 不能当"模型准确率提升"的证据。真正的效果证据看下面 normalized_* 与
    # hidden_overrun——它们依赖 budget_reported_raw（模型原始值）。
    report_consistent = abs(reported_total - budget_total) <= max(
        1, int(budget_total * BUDGET_REPORT_TOLERANCE)
    )
    # 模型原始上报值。旧数据没有这个字段（当时未留存），退化为"未干预"，
    # 同时用 raw_available 标出来，避免把"没记录"误读成"没纠正"。
    reported_raw = trip_plan.budget_reported_raw
    raw_available = reported_raw is not None
    raw_baseline = reported_raw if raw_available else reported_total
    normalized = raw_available and abs(raw_baseline - reported_total) > 0
    normalized_delta = (reported_total - raw_baseline) if normalized else 0
    # 修复前的隐性超支：模型上报"没超预算"，实际按单价算超了。
    # 这是用户会当场失信的那类错（到店才发现钱不够），比"偏差金额"更该报。
    hidden_overrun = (
        raw_available
        and target > 0
        and raw_baseline <= target < budget_total
    )
    hidden_overrun_amount = (budget_total - target) if hidden_overrun else 0
    deviation: Optional[float] = None
    budget_over = False
    if target > 0:
        deviation = round((budget_total - target) / target, 4)
        budget_over = deviation > over_tolerance

    # --- 4. 约束 ---
    profile = build_preference_profile(request)
    constraint_terms = list(profile.get("negative_constraints") or [])
    constraint_terms.extend(expect.get("forbidden_terms") or [])
    violations = find_constraint_violations(trip_plan, constraint_terms)

    # --- 5. 重复 ---
    duplicates = _duplicate_stats(trip_plan)

    grounding_ok = grounding["rate"] >= 0.8 or grounding["skipped"]
    passed = bool(structure_ok and grounding_ok and not budget_over and not violations)

    return {
        "case_id": case.get("id", ""),
        "city": request.city,
        "days": request.travel_days,
        "structure": {
            "ok": structure_ok,
            "days_count_ok": days_count_ok,
            "day_index_ok": day_index_ok,
            "dates_ok": dates_ok,
            "thin_days": thin_days,
            "missing_meals_days": missing_meals,
            "budget_arithmetic_ok": budget_arithmetic_ok,
        },
        "grounding": grounding,
        "grounding_ok": grounding_ok,
        "budget": {
            "target": target,
            "reported_total": reported_total,
            "recomputed_total": budget_total,
            "report_consistent": report_consistent,
            "reported_raw": reported_raw,
            "raw_available": raw_available,
            "normalized": normalized,
            "normalized_delta": normalized_delta,
            "hidden_overrun": hidden_overrun,
            "hidden_overrun_amount": hidden_overrun_amount,
            "deviation": deviation,
            "over_tolerance": over_tolerance,
            "over": budget_over,
        },
        "constraints": {
            "terms": constraint_terms,
            "violation_count": len(violations),
            "violations": violations,
            "messages": summarize_violations(violations),
        },
        "duplicates": duplicates,
        "passed": passed,
    }


def _rate(values: Sequence[bool]) -> float:
    if not values:
        return 0.0
    return round(sum(1 for value in values if value) / len(values), 4)


def aggregate(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """把逐条结果汇总成一张可对比的报表。"""
    if not results:
        return {"cases": 0}

    deviations = [
        item["budget"]["deviation"]
        for item in results
        if item["budget"]["deviation"] is not None
    ]
    over_count = sum(1 for item in results if item["budget"]["over"])
    violation_cases = sum(1 for item in results if item["constraints"]["violation_count"] > 0)
    violation_total = sum(item["constraints"]["violation_count"] for item in results)
    grounding_rates = [item["grounding"]["rate"] for item in results]
    duplicate_cases = sum(1 for item in results if item["duplicates"]["duplicate_count"] > 0)
    report_consistent_cases = sum(
        1 for item in results if item["budget"].get("report_consistent")
    )
    budget_cases = [item["budget"] for item in results]
    raw_available_cases = sum(1 for b in budget_cases if b.get("raw_available"))
    normalized_cases = sum(1 for b in budget_cases if b.get("normalized"))
    normalized_amount = sum(abs(int(b.get("normalized_delta") or 0)) for b in budget_cases)
    hidden_overrun_cases = sum(1 for b in budget_cases if b.get("hidden_overrun"))
    hidden_overrun_max = max(
        [int(b.get("hidden_overrun_amount") or 0) for b in budget_cases] or [0]
    )

    return {
        "cases": len(results),
        "passed": sum(1 for item in results if item["passed"]),
        "pass_rate": _rate([bool(item["passed"]) for item in results]),
        "structure_pass_rate": _rate([bool(item["structure"]["ok"]) for item in results]),
        "grounding_mean": round(sum(grounding_rates) / len(grounding_rates), 4),
        "grounding_ok_rate": _rate([bool(item["grounding_ok"]) for item in results]),
        # 这是"报警"不是"指标"：归一化生效时它必然为 True，值本身没有信息量；
        # 唯一有用的是它为 False 的那一刻——说明归一化被摘掉或重算逻辑被改坏。
        # 所以不给百分比，给布尔值，避免一个永远 100% 的数字被误当成质量提升。
        "budget_wiring_ok": report_consistent_cases == len(results),
        # 下面四项依赖 budget_reported_raw，是"代码纠正了多少"的可信度量。
        # 旧数据没有原始值，raw_available_cases 会提醒读者：这一批度量不到。
        "budget_raw_available_cases": raw_available_cases,
        "budget_normalized_cases": normalized_cases,
        "budget_normalized_amount": normalized_amount,
        "budget_hidden_overrun_cases": hidden_overrun_cases,
        "budget_hidden_overrun_max": hidden_overrun_max,
        "budget_report_consistent_rate": round(report_consistent_cases / len(results), 4),
        "budget_over_rate": round(over_count / len(results), 4),
        "budget_deviation_median": round(median(deviations), 4) if deviations else None,
        "budget_deviation_mean": round(sum(deviations) / len(deviations), 4) if deviations else None,
        "constraint_violation_case_rate": round(violation_cases / len(results), 4),
        "constraint_violation_total": violation_total,
        "duplicate_case_rate": round(duplicate_cases / len(results), 4),
    }


def _budget_audit_line(summary: Dict[str, Any]) -> str:
    """预算纠正审计的一行摘要（含接线报警）。

    为什么不再输出"预算上报一致率"这个百分比：归一化生效后它恒为 100%，
    值本身没有信息量。唯一有用的是它为 false 的那一刻——说明归一化被摘掉、
    或重算逻辑被改坏。所以降级成布尔报警，失败时才出现。

    旧数据没有 budget_reported_raw，必须显式说"度量不到"，不能默认成 0。
    """
    parts: List[str] = []
    cases = int(summary.get("cases") or 0)
    raw_available = int(summary.get("budget_raw_available_cases") or 0)
    if summary.get("budget_wiring_ok") is False:
        # 两种可能无法用数据区分：归一化被摘掉，或这批数据本来就早于归一化上线。
        # 所以文案同时说出两种，别把历史数据误判成线上故障。
        if raw_available == 0:
            parts.append(
                "- ⚠️ 预算归一化未生效（也可能本批数据早于归一化上线）：多份计划的上报总额"
                "与按单价重算不符；若这是新跑的数据，检查 `normalize_budget()` 是否被摘掉"
            )
        else:
            parts.append(
                "- ⚠️ **预算归一化未生效**：多份计划的上报总额与按单价重算不符，"
                "检查 `normalize_budget()` 是否被摘掉、重算逻辑是否被改坏"
            )
    if raw_available == 0:
        parts.append(
            "- 预算纠正审计：**本批数据未记录模型原始上报值**（旧格式），"
            "无法度量纠正量；需重跑一次或使用 `scripts/audit_budget.py`"
        )
        return "\n".join(parts)
    normalized = int(summary.get("budget_normalized_cases") or 0)
    amount = int(summary.get("budget_normalized_amount") or 0)
    hidden = int(summary.get("budget_hidden_overrun_cases") or 0)
    hidden_max = int(summary.get("budget_hidden_overrun_max") or 0)
    parts.append(
        f"- 预算纠正审计：**{normalized}/{cases}** 份被代码纠正，累计金额差 **{amount} 元**；"
        f"其中修复前隐藏超支 **{hidden}** 份（最大 **{hidden_max} 元**）"
    )
    return "\n".join(parts)


def render_markdown(summary: Dict[str, Any], results: Sequence[Dict[str, Any]], title: str = "评测报告") -> str:
    """生成人看的 markdown 报告。"""
    lines = [
        f"# {title}",
        "",
        f"- 用例数：**{summary.get('cases', 0)}**",
        f"- 通过：**{summary.get('passed', 0)}**（通过率 **{summary.get('pass_rate', 0):.1%}**）",
        f"- 结构合法率：**{summary.get('structure_pass_rate', 0):.1%}**",
        f"- POI 真实命中率（均值）：**{summary.get('grounding_mean', 0):.1%}**",
        f"- 预算超支率：**{summary.get('budget_over_rate', 0):.1%}**"
        f"（偏差中位数 {_fmt_pct(summary.get('budget_deviation_median'))}）",
        _budget_audit_line(summary),
        f"- 约束违反：**{summary.get('constraint_violation_case_rate', 0):.1%}** 的用例"
        f"（共 {summary.get('constraint_violation_total', 0)} 条）",
        f"- 景点跨天重复：**{summary.get('duplicate_case_rate', 0):.1%}** 的用例",
        "",
        "| # | 城市 | 天 | 通过 | 结构 | 真实率 | 预算 Δ | 约束违反 | 重复 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for index, item in enumerate(results, 1):
        structure = "OK" if item["structure"]["ok"] else "FAIL"
        passed = "✅" if item["passed"] else "❌"
        lines.append(
            "| {i} | {city} | {days} | {passed} | {structure} | {grounding:.0%} | {dev} | {viol} | {dup} |".format(
                i=index,
                city=item["city"],
                days=item["days"],
                passed=passed,
                structure=structure,
                grounding=item["grounding"]["rate"],
                dev=_fmt_pct(item["budget"]["deviation"]),
                viol=item["constraints"]["violation_count"],
                dup=item["duplicates"]["duplicate_count"],
            )
        )

    failing = [item for item in results if not item["passed"]]
    if failing:
        lines.extend(["", "## 未通过用例", ""])
        for item in failing:
            reasons: List[str] = []
            if not item["structure"]["ok"]:
                if item["structure"]["thin_days"]:
                    reasons.append(f"景点不足的天：{item['structure']['thin_days']}")
                if item["structure"]["missing_meals_days"]:
                    reasons.append(f"缺正餐的天：{item['structure']['missing_meals_days']}")
                if not item["structure"]["dates_ok"]:
                    reasons.append("日期与请求不一致")
                if not item["structure"]["budget_arithmetic_ok"]:
                    reasons.append("预算分项加总不自洽")
            if not item["grounding_ok"]:
                reasons.append(f"真实率 {item['grounding']['rate']:.0%} 低于 80%")
            if item["budget"]["over"]:
                reasons.append(f"超预算 {item['budget']['deviation']:.0%}")
            if item["constraints"]["violation_count"]:
                reasons.extend(item["constraints"]["messages"])
            lines.append(f"- **{item['case_id'] or item['city']}**：" + "；".join(reasons))
    return "\n".join(lines) + "\n"


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:+.0%}"
