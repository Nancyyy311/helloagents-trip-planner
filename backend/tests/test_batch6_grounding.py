"""批次6回归测试（上）：Grounding —— 对抗模型编造景点/餐厅/坐标。

不烧额度：全部离线构造 TripPlan + PlannerContext，不调用 LLM 和高德。
"""

from __future__ import annotations

from app.api.observability import metrics
from app.models.schemas import (
    Attraction,
    DayPlan,
    Hotel,
    Location,
    Meal,
    TripPlan,
)
from app.planner.grounding import (
    SKIPPED,
    UNVERIFIED,
    VERIFIED_CANDIDATE,
    VERIFIED_LODGING_BREAKFAST,
    verify_trip_plan,
)


def _loc(lng: float, lat: float) -> Location:
    return Location(longitude=lng, latitude=lat)


def _context(scenic=None, food=None, hotel=None):
    """构造最小 PlannerContext，只保留 grounding 用到的候选桶。"""
    return {
        "tool_snapshot": {
            "classic_pois": scenic or [],
            "preference_pois": [],
            "scenic_pois": [],
            "experience_pois": [],
            "food_pois": food or [],
            "hotel_pois": hotel or [],
        }
    }


def _plan(attractions=None, meals=None, hotel=None) -> TripPlan:
    return TripPlan(
        city="杭州",
        start_date="2027-03-10",
        end_date="2027-03-10",
        days=[
            DayPlan(
                date="2027-03-10",
                day_index=0,
                description="第一天",
                transportation="公共交通",
                accommodation="经济型酒店",
                hotel=hotel,
                attractions=attractions or [],
                meals=meals or [],
            )
        ],
        overall_suggestions="建议错峰出行",
    )


def _attraction(name: str, lng: float, lat: float) -> Attraction:
    return Attraction(
        name=name,
        address="模型写的地址",
        location=_loc(lng, lat),
        visit_duration=120,
        description="模型写的描述",
    )


SCENIC_POOL = [
    {
        "name": "西湖风景名胜区",
        "address": "浙江省杭州市西湖区龙井路1号",
        "location": {"longitude": 120.150000, "latitude": 30.250000},
        "id": "B0FFH3XXXX",
    }
]


# --------------------------------------------------------------- 命中与编造
def test_verified_attraction_is_marked():
    """名字命中候选池 -> verified=True，并回填真实地址与 poi_id。"""
    plan = _plan(attractions=[_attraction("西湖风景名胜区", 120.1501, 30.2501)])
    summary = verify_trip_plan(plan, _context(scenic=SCENIC_POOL))

    attraction = plan.days[0].attractions[0]
    assert attraction.verified is True
    assert attraction.verification == VERIFIED_CANDIDATE
    assert summary.attractions_verified == 1
    assert summary.unverified_attractions == []
    # 原项目的 enrich 只回填餐饮，景点地址一直是模型编的；这里补上
    assert attraction.address == "浙江省杭州市西湖区龙井路1号"
    assert attraction.poi_id == "B0FFH3XXXX"


def test_fabricated_attraction_is_marked_unverified():
    """候选池里没有 -> verified=False，且名字进 unverified 列表。"""
    plan = _plan(attractions=[_attraction("不存在的天空之城", 120.1, 30.2)])
    summary = verify_trip_plan(plan, _context(scenic=SCENIC_POOL))

    attraction = plan.days[0].attractions[0]
    assert attraction.verified is False
    assert attraction.verification == UNVERIFIED
    assert summary.attractions_verified == 0
    assert summary.unverified_attractions == ["不存在的天空之城"]


def test_alias_name_counts_as_verified():
    """模型写"西湖"，候选是"西湖风景名胜区" —— 别名匹配应算命中。"""
    plan = _plan(attractions=[_attraction("西湖", 120.15, 30.25)])
    summary = verify_trip_plan(plan, _context(scenic=SCENIC_POOL))

    assert plan.days[0].attractions[0].verified is True
    assert summary.attractions_verified == 1


# ------------------------------------------------------------------ 坐标纠正
def test_fabricated_coordinates_are_overwritten():
    """名字命中但坐标差了十几度（编的）-> 用真实坐标覆盖并计数。"""
    plan = _plan(attractions=[_attraction("西湖风景名胜区", 100.0000, 40.0000)])
    summary = verify_trip_plan(plan, _context(scenic=SCENIC_POOL))

    location = plan.days[0].attractions[0].location
    assert location.longitude == 120.15
    assert location.latitude == 30.25
    assert summary.coordinates_corrected == 1


def test_close_coordinates_are_kept():
    """坐标只差约 100 米（同一景区不同入口）-> 不应覆盖。"""
    plan = _plan(attractions=[_attraction("西湖风景名胜区", 120.1508, 30.2505)])
    summary = verify_trip_plan(plan, _context(scenic=SCENIC_POOL))

    location = plan.days[0].attractions[0].location
    assert location.longitude == 120.1508
    assert summary.coordinates_corrected == 0


# ---------------------------------------------------------------- 池为空不误伤
def test_empty_candidate_pool_is_skipped():
    """工具没抓到数据时不能判"编造"，否则会把真实地点误标成假的。"""
    plan = _plan(attractions=[_attraction("某某古镇", 120.1, 30.2)])
    summary = verify_trip_plan(plan, _context(scenic=[]))

    attraction = plan.days[0].attractions[0]
    assert attraction.verified is None
    assert attraction.verification == SKIPPED
    assert summary.unverified_attractions == []


def test_missing_context_is_skipped():
    """完全没有 PlannerContext（如兜底计划）时，整体标记 skipped。"""
    plan = _plan(attractions=[_attraction("西湖", 120.15, 30.25)])
    summary = verify_trip_plan(plan, None)

    assert summary.skipped is True
    assert plan.grounding is summary


# ------------------------------------------------------------------- 餐饮/酒店
def test_lodging_breakfast_is_verified():
    """住宿含早不是 food_pois，但不算编造。"""
    plan = _plan(meals=[Meal(type="breakfast", name="酒店早餐", estimated_cost=0)])
    summary = verify_trip_plan(plan, _context(scenic=SCENIC_POOL))

    meal = plan.days[0].meals[0]
    assert meal.verified is True
    assert meal.verification == VERIFIED_LODGING_BREAKFAST
    assert summary.meals_verified == 1


def test_fabricated_meal_is_marked():
    plan = _plan(meals=[Meal(type="lunch", name="不存在的小馆", estimated_cost=50)])
    summary = verify_trip_plan(
        plan,
        _context(food=[{"name": "外婆家", "address": "某路1号", "location": {"longitude": 120.16, "latitude": 30.26}}]),
    )

    assert plan.days[0].meals[0].verified is False
    assert summary.unverified_meals == ["不存在的小馆"]


def test_hotel_verified_and_address_filled():
    plan = _plan(
        hotel=Hotel(name="杭州某某酒店", address="", location=None, estimated_cost=300)
    )
    summary = verify_trip_plan(
        plan,
        _context(hotel=[{"name": "杭州某某酒店", "address": "上城区某路8号", "location": {"longitude": 120.17, "latitude": 30.27}}]),
    )

    hotel = plan.days[0].hotel
    assert hotel.verified is True
    assert hotel.address == "上城区某路8号"
    assert hotel.location is not None
    assert summary.hotels_verified == 1
    # 原本没有坐标 -> 回填算一次纠正
    assert summary.coordinates_corrected == 1


# -------------------------------------------------------------------- 汇总率
def test_verified_rate_and_counts():
    plan = _plan(
        attractions=[
            _attraction("西湖风景名胜区", 120.15, 30.25),
            _attraction("编造景点A", 120.11, 30.21),
        ],
        meals=[Meal(type="lunch", name="编造餐厅B", estimated_cost=40)],
    )
    summary = verify_trip_plan(plan, _context(scenic=SCENIC_POOL, food=[]))

    assert summary.attractions_total == 2
    assert summary.attractions_verified == 1
    assert summary.meals_total == 1
    # food 池为空 -> 餐饮标记 skipped，但计数仍算 1 条未命中
    assert summary.verified_rate == round(1 / 3, 4)
    assert summary.scenic_candidates == 1


def test_metrics_accumulate_grounding():
    before = metrics.snapshot()
    plan = _plan(
        attractions=[
            _attraction("西湖风景名胜区", 120.15, 30.25),
            _attraction("编造景点C", 120.11, 30.21),
        ]
    )
    summary = verify_trip_plan(plan, _context(scenic=SCENIC_POOL))
    metrics.record_grounding(summary)

    after = metrics.snapshot()
    assert after["grounding_checked"] - before["grounding_checked"] == 2
    assert after["grounding_verified"] - before["grounding_verified"] == 1
    assert after["grounding_unverified"] - before["grounding_unverified"] == 1
    assert "grounding_rate" in after
