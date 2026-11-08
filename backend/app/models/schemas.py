"""数据模型定义"""

from typing import List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ============ 请求模型 ============

class PartyInfo(BaseModel):
    """同行人数信息"""
    adults: int = Field(..., description="成人数量", ge=0, le=20, json_schema_extra={"example": 2})
    children: int = Field(default=0, description="儿童数量", ge=0, le=20, json_schema_extra={"example": 0})
    elders: int = Field(default=0, description="老人数量", ge=0, le=20, json_schema_extra={"example": 0})
    total: int = Field(..., description="同行总人数", ge=1, le=30, json_schema_extra={"example": 2})
    companion_type: str = Field(default="other", description="同行类型", json_schema_extra={"example": "couple"})

    @model_validator(mode="after")
    def validate_total(self):
        """total 必须显式等于分项人数之和。"""
        expected_total = self.adults + self.children + self.elders
        if self.total != expected_total:
            raise ValueError(f"total必须等于adults+children+elders: expected={expected_total}, got={self.total}")
        if expected_total <= 0:
            raise ValueError("同行总人数必须大于0")
        return self


class BudgetConstraint(BaseModel):
    """预算约束"""
    amount: Optional[int] = Field(default=None, description="整趟旅行总预算；无明确预算时为null", ge=0, json_schema_extra={"example": 1200})
    scope: str = Field(default="total", description="预算口径，固定为total", json_schema_extra={"example": "total"})
    currency: str = Field(default="CNY", description="币种", json_schema_extra={"example": "CNY"})
    budget_level: str = Field(default="standard", description="预算档位", json_schema_extra={"example": "limited"})
    strictness: str = Field(default="none", description="预算严格程度: hard/soft/none", json_schema_extra={"example": "soft"})


class TripRequest(BaseModel):
    """旅行规划请求"""
    city: str = Field(..., description="目的地城市", json_schema_extra={"example": "北京"})
    start_date: str = Field(..., description="开始日期 YYYY-MM-DD", json_schema_extra={"example": "2025-06-01"})
    end_date: str = Field(..., description="结束日期 YYYY-MM-DD", json_schema_extra={"example": "2025-06-03"})
    travel_days: int = Field(..., description="旅行天数", ge=1, le=30, json_schema_extra={"example": 3})
    transportation: str = Field(..., description="交通方式", json_schema_extra={"example": "公共交通"})
    accommodation: str = Field(..., description="住宿偏好", json_schema_extra={"example": "经济型酒店"})
    preferences: List[str] = Field(default_factory=list, description="旅行偏好标签", json_schema_extra={"example": ["历史文化", "美食"]})
    free_text_input: Optional[str] = Field(default="", description="额外要求", json_schema_extra={"example": "希望多安排一些博物馆"})
    party: PartyInfo = Field(..., description="同行人数")
    budget_constraint: BudgetConstraint = Field(..., description="预算约束")

    # 改造点：Pydantic V2 已弃用 class Config，改用 model_config；
    # 否则每次启动都会刷 DeprecatedSince20 警告，属于"跑着能跑但已在过时写法上"。
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "city": "北京",
                "start_date": "2025-06-01",
                "end_date": "2025-06-03",
                "travel_days": 3,
                "transportation": "公共交通",
                "accommodation": "经济型酒店",
                "preferences": ["历史文化", "美食"],
                "free_text_input": "希望多安排一些博物馆",
                "party": {
                    "adults": 2,
                    "children": 0,
                    "elders": 0,
                    "total": 2,
                    "companion_type": "couple"
                },
                "budget_constraint": {
                    "amount": 1200,
                    "scope": "total",
                    "currency": "CNY",
                    "budget_level": "limited",
                    "strictness": "soft"
                }
            }
        }
    )


class POISearchRequest(BaseModel):
    """POI搜索请求"""
    keywords: str = Field(..., description="搜索关键词", json_schema_extra={"example": "故宫"})
    city: str = Field(..., description="城市", json_schema_extra={"example": "北京"})
    citylimit: bool = Field(default=True, description="是否限制在城市范围内")


class RouteRequest(BaseModel):
    """路线规划请求"""
    origin_address: str = Field(..., description="起点地址", json_schema_extra={"example": "北京市朝阳区阜通东大街6号"})
    destination_address: str = Field(..., description="终点地址", json_schema_extra={"example": "北京市海淀区上地十街10号"})
    origin_city: Optional[str] = Field(default=None, description="起点城市")
    destination_city: Optional[str] = Field(default=None, description="终点城市")
    route_type: str = Field(default="walking", description="路线类型: walking/driving/transit")


# ============ 响应模型 ============

class Location(BaseModel):
    """地理位置"""
    longitude: float = Field(..., description="经度")
    latitude: float = Field(..., description="纬度")


class Attraction(BaseModel):
    """景点信息"""
    name: str = Field(..., description="景点名称")
    address: str = Field(..., description="地址")
    location: Location = Field(..., description="经纬度坐标")
    visit_duration: int = Field(..., description="建议游览时间(分钟)")
    description: str = Field(..., description="景点描述")
    category: Optional[str] = Field(default="景点", description="景点类别")
    rating: Optional[float] = Field(default=None, description="评分")
    photos: Optional[List[str]] = Field(default_factory=list, description="景点图片URL列表")
    poi_id: Optional[str] = Field(default="", description="POI ID")
    image_url: Optional[str] = Field(default=None, description="图片URL")
    ticket_price: int = Field(default=0, description="门票价格(元)")
    # 改造点：真实性校验结果（None=无法校验，False=候选池里找不到）
    verified: Optional[bool] = Field(default=None, description="是否命中高德候选池")
    verification: Optional[str] = Field(default=None, description="校验结果: candidate/unverified/skipped_no_candidates")


class Meal(BaseModel):
    """餐饮信息"""
    type: str = Field(..., description="餐饮类型: breakfast/lunch/dinner/snack")
    name: str = Field(..., description="餐饮名称")
    address: Optional[str] = Field(default=None, description="地址")
    location: Optional[Location] = Field(default=None, description="经纬度坐标")
    description: Optional[str] = Field(default=None, description="描述")
    estimated_cost: int = Field(default=0, description="预估费用(元)")
    verified: Optional[bool] = Field(default=None, description="是否命中高德候选池")
    verification: Optional[str] = Field(default=None, description="校验结果: candidate/lodging_breakfast/unverified/skipped_no_candidates")


class Hotel(BaseModel):
    """酒店信息"""
    name: str = Field(..., description="酒店名称")
    address: str = Field(default="", description="酒店地址")
    location: Optional[Location] = Field(default=None, description="酒店位置")
    price_range: str = Field(default="", description="价格范围")
    rating: str = Field(default="", description="评分")
    distance: str = Field(default="", description="距离景点距离")
    type: str = Field(default="", description="酒店类型")
    estimated_cost: int = Field(default=0, description="预估费用(元/晚)")
    verified: Optional[bool] = Field(default=None, description="是否命中高德候选池")
    verification: Optional[str] = Field(default=None, description="校验结果: candidate/unverified/skipped_no_candidates")


class DayPlan(BaseModel):
    """单日行程"""
    date: str = Field(..., description="日期 YYYY-MM-DD")
    day_index: int = Field(..., description="第几天(从0开始)")
    description: str = Field(..., description="当日行程描述")
    transportation: str = Field(..., description="交通方式")
    accommodation: str = Field(..., description="住宿")
    hotel: Optional[Hotel] = Field(default=None, description="推荐酒店")
    attractions: List[Attraction] = Field(default_factory=list, description="景点列表")
    meals: List[Meal] = Field(default_factory=list, description="餐饮列表")


class WeatherInfo(BaseModel):
    """天气信息"""
    date: str = Field(..., description="日期 YYYY-MM-DD")
    day_weather: str = Field(default="", description="白天天气")
    night_weather: str = Field(default="", description="夜间天气")
    day_temp: Union[int, str] = Field(default=0, description="白天温度")
    night_temp: Union[int, str] = Field(default=0, description="夜间温度")
    wind_direction: str = Field(default="", description="风向")
    wind_power: str = Field(default="", description="风力")

    @field_validator('day_temp', 'night_temp', mode='before')
    @classmethod
    def parse_temperature(cls, v):
        """解析温度,移除°C等单位；远期天气未知时保留原文。"""
        if isinstance(v, str):
            # 移除°C, ℃等单位符号
            v = v.replace('°C', '').replace('℃', '').replace('°', '').strip()
            try:
                return int(v)
            except ValueError:
                return v or "未知"
        return v


class Budget(BaseModel):
    """预算信息"""
    total_attractions: int = Field(default=0, description="景点门票总费用")
    total_hotels: int = Field(default=0, description="酒店总费用")
    total_meals: int = Field(default=0, description="餐饮总费用")
    total_transportation: int = Field(default=0, description="交通总费用")
    total: int = Field(default=0, description="总费用")


class GroundingSummary(BaseModel):
    """POI 真实性校验摘要（对抗模型编造景点/餐厅/坐标）。

    原项目把校验结果只 print 到日志，API 响应里看不到，因此前端和用户
    无法分辨哪条是真实存在的。这里把它变成响应的一部分。
    """
    skipped: bool = Field(default=False, description="候选池为空或缺少上下文，未做校验")
    attractions_total: int = Field(default=0, description="景点总数")
    attractions_verified: int = Field(default=0, description="命中候选池的景点数")
    meals_total: int = Field(default=0, description="餐饮总数")
    meals_verified: int = Field(default=0, description="命中候选池的餐饮数")
    hotels_total: int = Field(default=0, description="酒店总数")
    hotels_verified: int = Field(default=0, description="命中候选池的酒店数")
    verified_rate: float = Field(default=0.0, description="整体命中率 0~1")
    coordinates_corrected: int = Field(default=0, description="被真实坐标纠正的 POI 数")
    unverified_attractions: List[str] = Field(default_factory=list, description="未命中的景点名")
    unverified_meals: List[str] = Field(default_factory=list, description="未命中的餐饮名")
    unverified_hotels: List[str] = Field(default_factory=list, description="未命中的酒店名")
    scenic_candidates: int = Field(default=0, description="景点候选池大小")
    food_candidates: int = Field(default=0, description="餐饮候选池大小")
    hotel_candidates: int = Field(default=0, description="酒店候选池大小")


class TripPlan(BaseModel):
    """旅行计划"""
    city: str = Field(..., description="目的地城市")
    start_date: str = Field(..., description="开始日期")
    end_date: str = Field(..., description="结束日期")
    days: List[DayPlan] = Field(..., description="每日行程")
    weather_info: List[WeatherInfo] = Field(default_factory=list, description="天气信息")
    overall_suggestions: str = Field(..., description="总体建议")
    budget: Optional[Budget] = Field(default=None, description="预算信息")
    grounding: Optional[GroundingSummary] = Field(default=None, description="POI真实性校验摘要")
    # 改造点：留存模型"自己说"的预算总额，归一化前写入。
    # 没有它，"代码纠正了多少钱"就无法离线度量——因为 budget.total 已被覆盖，
    # 拿它和重算值比是恒真命题（同义反复），不能当效果证据。
    # None 表示模型上报值与重算值一致（或未触发归一化），不参与展示。
    budget_reported_raw: Optional[int] = Field(
        default=None,
        description="模型原始上报的预算总额（归一化前），仅用于审计与评测",
    )


class TripPlanResponse(BaseModel):
    """旅行计划响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(default="", description="消息")
    data: Optional[TripPlan] = Field(default=None, description="旅行计划数据")
    # 改造点：返回 plan_id，前端才能拿着它做"增量修改"（改一句话不用整趟重生成）
    plan_id: Optional[str] = Field(default=None, description="计划ID，用于后续增量修改")
    # 改造点：告诉调用方这份结果是不是复用了别的在飞请求。
    # 前端可以据此提示"已有相同请求在生成"，也方便演示时证明合并真的生效。
    coalesced: bool = Field(
        default=False,
        description="是否复用了另一个相同请求的结果（请求合并命中）",
    )


class RefineRequest(BaseModel):
    """增量修改请求"""
    instruction: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="自然语言修改要求，例如：把第二天改成室内活动",
        json_schema_extra={"example": "把第二天下午改成室内活动，可能下雨"},
    )


class DayDiff(BaseModel):
    """单日变更详情"""
    day_index: int = Field(..., description="第几天(从0开始)")
    date: str = Field(..., description="日期")
    changed: bool = Field(..., description="该天是否发生变化")
    before_attractions: List[str] = Field(default_factory=list, description="修改前景点名")
    after_attractions: List[str] = Field(default_factory=list, description="修改后景点名")
    before_meals: List[str] = Field(default_factory=list, description="修改前餐饮名")
    after_meals: List[str] = Field(default_factory=list, description="修改后餐饮名")


class PlanDiff(BaseModel):
    """增量修改的变更摘要。

    增量修改最大的价值是"只改该改的"，所以必须能证明这一点 —— 光说"我做了增量修改"
    没用，要能给出"只动了 1 天，另外 2 天一字未变"的证据。
    """
    changed_days: List[DayDiff] = Field(default_factory=list, description="发生变化的日期")
    unchanged_days: int = Field(default=0, description="未变化的日期数")
    budget_before: int = Field(default=0, description="修改前总预算")
    budget_after: int = Field(default=0, description="修改后总预算")
    budget_delta: int = Field(default=0, description="预算变化")
    suggestions_changed: bool = Field(default=False, description="总体建议是否变化")


class RefineResponse(BaseModel):
    """增量修改响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(default="", description="消息")
    plan_id: Optional[str] = Field(default=None, description="修改后的新计划ID")
    data: Optional[TripPlan] = Field(default=None, description="修改后的计划")
    diff: Optional[PlanDiff] = Field(default=None, description="变更摘要")


class POIInfo(BaseModel):
    """POI信息"""
    id: str = Field(..., description="POI ID")
    name: str = Field(..., description="名称")
    type: str = Field(..., description="类型")
    address: str = Field(..., description="地址")
    location: Location = Field(..., description="经纬度坐标")
    tel: Optional[str] = Field(default=None, description="电话")


class POISearchResponse(BaseModel):
    """POI搜索响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(default="", description="消息")
    data: List[POIInfo] = Field(default_factory=list, description="POI列表")


class RouteInfo(BaseModel):
    """路线信息"""
    distance: float = Field(..., description="距离(米)")
    duration: int = Field(..., description="时间(秒)")
    route_type: str = Field(..., description="路线类型")
    description: str = Field(..., description="路线描述")


class RouteResponse(BaseModel):
    """路线规划响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(default="", description="消息")
    data: Optional[RouteInfo] = Field(default=None, description="路线信息")


class WeatherResponse(BaseModel):
    """天气查询响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(default="", description="消息")
    data: List[WeatherInfo] = Field(default_factory=list, description="天气信息")


# ============ 错误响应 ============

class ErrorResponse(BaseModel):
    """错误响应"""
    success: bool = Field(default=False, description="是否成功")
    message: str = Field(..., description="错误消息")
    error_code: Optional[str] = Field(default=None, description="错误代码")
