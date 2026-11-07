"""Planner 输出的真实性校验（Grounding）。

改造背景（原项目的真实情况，不是"完全没有"）：
原项目其实已经算了 grounding，但分散在三个互不相通的地方：

1. `output.warn_plan_grounding()` —— 只在 stdout 打一行 `⚠️` 警告，API 响应里完全看不到；
2. `output.enrich_trip_plan_poi_details()` —— 只回填**餐饮**的地址/坐标，
   景点和酒店从不回填，模型编的坐标会原样进响应；
3. `rerank.compute_metrics()` —— 算 `*_grounding_rate` 只用于打分和打印，
   不进 `/metrics`，线上无法持续观察。

结果是：模型编造的景点照样返回给前端，用户无法分辨哪条是真的。

本模块把这三件事收敛到一处，并且**真正落地**：
- 给每个 POI 标注 `verified` 与 `verification`（写回 TripPlan，前端可见）；
- 命中候选但坐标偏离过大时，用真实坐标覆盖（计入 `coordinates_corrected`）；
- 汇总成 `TripPlan.grounding` 摘要，同时累加 `/metrics` 计数。

设计原则：**只标注、不删除**。高德召回有缺口，硬删会把真实存在的小馆子、
老店误杀；把判断权交给前端（可以显示"未核实"标签），比静默删掉更诚实。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from ..models.schemas import GroundingSummary, TripPlan
from .output import (
    find_candidate_by_name,
    is_lodging_breakfast_meal,
    location_from_candidate,
)

# 坐标容差：同一景区不同入口相差几百米是正常的，超过这个距离视为"坐标不可信"。
GROUNDING_COORD_TOLERANCE_METERS = 800

VERIFIED_CANDIDATE = "candidate"
VERIFIED_LODGING_BREAKFAST = "lodging_breakfast"
UNVERIFIED = "unverified"
SKIPPED = "skipped_no_candidates"


def verify_trip_plan(
    trip_plan: TripPlan,
    planner_context: Optional[Dict[str, Any]],
) -> GroundingSummary:
    """校验 TripPlan 里每个 POI 是否真实来自工具候选，并把结果写回计划。

    Args:
        trip_plan: 已通过格式校验的旅行计划（会就地修改）
        planner_context: PlannerContext；为空时无法校验，返回 skipped 摘要

    Returns:
        GroundingSummary：可直接放进响应，也可累加进 /metrics
    """
    summary = GroundingSummary()

    if not planner_context:
        summary.skipped = True
        trip_plan.grounding = summary
        return summary

    snapshot = planner_context.get("tool_snapshot") or {}
    scenic_pool = _merge_pools(
        snapshot.get("classic_pois"),
        snapshot.get("preference_pois"),
        snapshot.get("scenic_pois"),
        snapshot.get("experience_pois"),
    )
    food_pool = _merge_pools(snapshot.get("food_pois"))
    hotel_pool = _merge_pools(snapshot.get("hotel_pois"))

    # 候选池为空说明工具没抓到数据（或该城市无召回），此时"
    # 未命中"不代表模型编造，标记为 skipped 而不是 unverified，避免误伤。
    summary.scenic_candidates = len(scenic_pool)
    summary.food_candidates = len(food_pool)
    summary.hotel_candidates = len(hotel_pool)

    for day in trip_plan.days:
        for attraction in day.attractions:
            summary.attractions_total += 1
            candidate, verified, corrected = _verify_one(
                name=attraction.name,
                location=attraction.location,
                pool=scenic_pool if scenic_pool else None,
            )
            if verified:
                summary.attractions_verified += 1
                attraction.verified = True
                attraction.verification = VERIFIED_CANDIDATE
                if candidate:
                    _apply_candidate_details(
                        target=attraction,
                        candidate=candidate,
                        corrected_flag=corrected,
                    )
                if corrected:
                    summary.coordinates_corrected += 1
            else:
                attraction.verified = False if scenic_pool else None
                attraction.verification = UNVERIFIED if scenic_pool else SKIPPED
                if scenic_pool:
                    summary.unverified_attractions.append(attraction.name)

        for meal in day.meals:
            summary.meals_total += 1
            # 住宿含早不需要外部 POI 支撑，按已核实处理（与 rerank 口径一致）
            if is_lodging_breakfast_meal(meal.name, meal.type):
                summary.meals_verified += 1
                meal.verified = True
                meal.verification = VERIFIED_LODGING_BREAKFAST
                continue

            candidate, verified, corrected = _verify_one(
                name=meal.name,
                location=meal.location,
                pool=food_pool if food_pool else None,
            )
            if verified:
                summary.meals_verified += 1
                meal.verified = True
                meal.verification = VERIFIED_CANDIDATE
                if candidate:
                    _apply_candidate_details(target=meal, candidate=candidate, corrected_flag=corrected)
                if corrected:
                    summary.coordinates_corrected += 1
            else:
                meal.verified = False if food_pool else None
                meal.verification = UNVERIFIED if food_pool else SKIPPED
                if food_pool:
                    summary.unverified_meals.append(meal.name)

        if day.hotel is not None:
            summary.hotels_total += 1
            candidate, verified, corrected = _verify_one(
                name=day.hotel.name,
                location=day.hotel.location,
                pool=hotel_pool if hotel_pool else None,
            )
            if verified:
                summary.hotels_verified += 1
                day.hotel.verified = True
                day.hotel.verification = VERIFIED_CANDIDATE
                if candidate:
                    _apply_candidate_details(
                        target=day.hotel,
                        candidate=candidate,
                        corrected_flag=corrected,
                    )
                if corrected:
                    summary.coordinates_corrected += 1
            else:
                day.hotel.verified = False if hotel_pool else None
                day.hotel.verification = UNVERIFIED if hotel_pool else SKIPPED
                if hotel_pool:
                    summary.unverified_hotels.append(day.hotel.name)

    total_items = summary.attractions_total + summary.meals_total + summary.hotels_total
    verified_items = (
        summary.attractions_verified + summary.meals_verified + summary.hotels_verified
    )
    summary.verified_rate = round(verified_items / total_items, 4) if total_items else 0.0

    trip_plan.grounding = summary
    _print_summary(summary)
    return summary


def _merge_pools(*buckets: Any) -> List[Dict[str, Any]]:
    """合并多个候选桶并按名称去重（同一 POI 可能在多个关键词下被召回）。"""
    merged: List[Dict[str, Any]] = []
    seen = set()
    for bucket in buckets:
        for row in bucket or []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            merged.append(row)
    return merged


def _verify_one(
    name: str,
    location: Any,
    pool: Optional[List[Dict[str, Any]]],
) -> Tuple[Optional[Dict[str, Any]], bool, bool]:
    """校验单个 POI。

    Returns:
        (命中的候选, 是否命中, 坐标是否被纠正)
    """
    if not pool:
        return None, False, False

    candidate = find_candidate_by_name(str(name or ""), pool)
    if candidate is None:
        return None, False, False

    corrected = False
    real_location = location_from_candidate(candidate)
    if real_location is not None:
        if location is None:
            corrected = True
        else:
            try:
                distance = _distance_meters(
                    float(location.longitude),
                    float(location.latitude),
                    real_location.longitude,
                    real_location.latitude,
                )
            except (TypeError, ValueError):
                distance = None
            if distance is None or distance > GROUNDING_COORD_TOLERANCE_METERS:
                corrected = True
    return candidate, True, corrected


def _apply_candidate_details(target: Any, candidate: Dict[str, Any], corrected_flag: bool) -> None:
    """用真实候选补全地址/坐标。

    地址：命中候选时**以真实地址为准**（真实数据 > 模型生成），
    只有候选没地址时才保留模型写的。这是刻意的取舍：模型写的地址常常
    是"XX路附近"这类模糊描述，可靠性不如高德。
    坐标：偏离超过容差才覆盖，避免把同一景区不同入口的正常差异改掉。
    """
    real_address = str(candidate.get("address") or "").strip()
    if real_address:
        target.address = real_address

    real_location = location_from_candidate(candidate)
    if real_location is not None and corrected_flag:
        target.location = real_location

    empty_poi_id = not getattr(target, "poi_id", "")
    if empty_poi_id and candidate.get("id"):
        try:
            target.poi_id = str(candidate.get("id") or "")
        except (AttributeError, ValueError):
            pass


def _distance_meters(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    """Haversine 距离（米）。不引第三方依赖，够用即可。"""
    radius = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))


def _print_summary(summary: GroundingSummary) -> None:
    """打印一行紧凑摘要，便于看日志时立刻发现幻觉。"""
    print(
        "🔍 Grounding: "
        f"景点 {summary.attractions_verified}/{summary.attractions_total}, "
        f"餐饮 {summary.meals_verified}/{summary.meals_total}, "
        f"酒店 {summary.hotels_verified}/{summary.hotels_total}"
        f"（坐标纠正 {summary.coordinates_corrected}）",
        flush=True,
    )
    for label, names in (
        ("景点", summary.unverified_attractions),
        ("餐饮", summary.unverified_meals),
        ("酒店", summary.unverified_hotels),
    ):
        if names:
            print(f"   ⚠️ 未命中候选的{label}: {', '.join(names[:5])}", flush=True)
