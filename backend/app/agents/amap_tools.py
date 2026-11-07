"""高德工具：让 Planner Agent 真正具备工具调用能力。

改造依据（原项目的矛盾点）:
- ``trip_planner_agent.py`` 创建 ``SimpleAgent`` 时从未传入 ``tool_registry``，
  而框架里 ``enable_tool_calling = enable_tool_calling and tool_registry is not None``，
  所以它恒为 False：所谓"多智能体/工具调用"实际是"预抓数据 + 一次性生成"。
- 本模块把高德查询封装成标准 ``Tool`` 注册给 Agent，模型在上下文不足时可自行补查，
  补上"模型自主决策调用工具"这一 Agent 的核心特征。

设计取舍（面试可讲）:
1. 复用 ``AmapPlannerClient``：它已内置磁盘缓存 + 进程级 QPS 限速，不重复实现；
2. 每次规划请求共享一份调用预算（默认 6 次），防止 3 个候选各自狂调工具，
   把延迟和额度同时打爆；
3. 工具失败不抛异常，返回可读的错误字符串——Agent 能继续生成，不至于整轮报废；
4. 工具返回内容刻意带上"不要编造未出现的地点"，把 grounding 要求写进观察结果里。
"""

from __future__ import annotations

import os
from threading import Lock
from typing import Any, Dict, List, Optional

from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.registry import ToolRegistry

from ..api.observability import metrics
from ..planner.amap import AmapPlannerClient


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# 单次规划请求的工具调用总预算（3 个候选共享）
PLANNER_TOOL_MAX_CALLS = _env_int("PLANNER_TOOL_MAX_CALLS", 6)
# 单条工具结果最多返回多少条 POI
PLANNER_TOOL_RESULT_LIMIT = _env_int("PLANNER_TOOL_RESULT_LIMIT", 8)


def _clamp(value: Any, low: int, high: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


class ToolBudget:
    """单次规划请求的工具调用总预算，线程安全（并行候选之间共享）。"""

    def __init__(self, max_calls: int = PLANNER_TOOL_MAX_CALLS) -> None:
        self.max_calls = max(0, max_calls)
        self._used = 0
        self._lock = Lock()

    def acquire(self) -> bool:
        """申请一次调用额度；额度用尽返回 False。"""
        with self._lock:
            if self._used >= self.max_calls:
                return False
            self._used += 1
            return True

    @property
    def used(self) -> int:
        with self._lock:
            return self._used


class _BudgetedAmapTool(Tool):
    """带预算控制与指标埋点的高德工具基类。"""

    def __init__(
        self,
        name: str,
        description: str,
        client: AmapPlannerClient,
        budget: ToolBudget,
    ) -> None:
        super().__init__(name=name, description=description, expandable=False)
        self._client = client
        self._budget = budget

    # --- 统一入口：预算 -> 执行 -> 指标 -> 兜底 ---
    def run(self, parameters: Dict[str, Any]) -> str:
        if not self._budget.acquire():
            metrics.record_tool_budget_exhausted()
            return (
                f"工具调用预算已用完（单次规划上限 {self._budget.max_calls} 次）。"
                "请立刻基于已给出的 PlannerContext 与工具结果直接输出 JSON，不要再调用工具。"
            )
        try:
            text = self._execute(parameters or {})
            metrics.record_tool(self.name, True)
            return text
        except Exception as error:  # noqa: BLE001 - 工具失败不应打断整轮生成
            metrics.record_tool(self.name, False)
            return f"工具执行失败：{error}（请基于已有信息继续生成，不要重复调用同一工具）"

    def _execute(self, parameters: Dict[str, Any]) -> str:  # pragma: no cover - 子类实现
        raise NotImplementedError


class AmapPOISearchTool(_BudgetedAmapTool):
    """高德地点检索：确认某个地点是否真实存在，并拿到真实地址与坐标。"""

    def __init__(self, client: AmapPlannerClient, budget: ToolBudget) -> None:
        super().__init__(
            name="amap_search_poi",
            description=(
                "检索某个城市里的真实地点（景点/餐厅/酒店/商场）。"
                "当你不确定某个地点是否存在、需要补全真实地址或经纬度时调用。"
                "参数：keywords（关键词，如 西湖）、city（城市，如 杭州）、limit（返回条数，可选）。"
            ),
            client=client,
            budget=budget,
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="keywords", type="string", description="搜索关键词，例如 西湖"),
            ToolParameter(name="city", type="string", description="城市名称，例如 杭州"),
            ToolParameter(
                name="limit",
                type="integer",
                description="返回条数，1-10",
                required=False,
                default=5,
            ),
        ]

    def _execute(self, parameters: Dict[str, Any]) -> str:
        keywords = str(parameters.get("keywords") or "").strip()
        city = str(parameters.get("city") or "").strip()
        if not keywords or not city:
            raise ValueError("keywords 与 city 均为必填")
        limit = _clamp(parameters.get("limit"), 1, PLANNER_TOOL_RESULT_LIMIT, 5)

        raw = self._client.get(
            "/place/text",
            {
                "keywords": keywords,
                "city": city,
                "citylimit": "true",
                "extensions": "all",
                "offset": limit,
                "page": 1,
            },
        )
        pois = raw.get("pois") or []
        if not pois:
            return (
                f"高德未检索到与「{keywords}」匹配的真实地点（城市={city}）。"
                "请不要编造该地点，改用 PlannerContext 中已存在的真实地点。"
            )

        lines = [f"高德真实检索结果（关键词={keywords}，城市={city}，共{len(pois)}条）："]
        for index, poi in enumerate(pois[:limit], start=1):
            address = poi.get("address")
            address_text = address if isinstance(address, str) and address else "无"
            lines.append(
                f"{index}. {poi.get('name', '')} | 类型={poi.get('type', '')} "
                f"| 地址={address_text} | 坐标={poi.get('location') or '无'}"
            )
        lines.append("只允许使用上述真实名称与坐标，禁止编造未出现在结果里的地点。")
        return "\n".join(lines)


class AmapWeatherTool(_BudgetedAmapTool):
    """高德天气预报：补齐超出 PlanningContext 覆盖范围的日期天气。"""

    def __init__(self, client: AmapPlannerClient, budget: ToolBudget) -> None:
        super().__init__(
            name="amap_weather",
            description=(
                "查询城市未来几天的真实天气预报。"
                "当 PlannerContext 中缺少某天天气、或你需要确认是否下雨以安排室内备选时调用。"
                "参数：city（城市，如 杭州）。"
            ),
            client=client,
            budget=budget,
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="city", type="string", description="城市名称，例如 杭州"),
        ]

    def _execute(self, parameters: Dict[str, Any]) -> str:
        city = str(parameters.get("city") or "").strip()
        if not city:
            raise ValueError("city 为必填")

        raw = self._client.get("/weather/weatherInfo", {"city": city, "extensions": "all"})
        forecasts = raw.get("forecasts") or []
        if not forecasts:
            return f"高德未返回 {city} 的天气预报（该日期可能超出预报范围），请按未知天气处理。"

        casts = (forecasts[0] or {}).get("casts") or []
        if not casts:
            return f"高德未返回 {city} 的逐日预报，请按未知天气处理。"

        lines = [f"高德真实天气预报（城市={city}，共{len(casts)}天）："]
        for cast in casts:
            lines.append(
                f"{cast.get('date', '')} 白天={cast.get('dayweather', '')} "
                f"夜间={cast.get('nightweather', '')} "
                f"气温={cast.get('nighttemp', '')}~{cast.get('daytemp', '')}℃ "
                f"风={cast.get('daywind', '')}{cast.get('daypower', '')}"
            )
        lines.append("若某天不在上面的预报范围内，说明超出预报窗口，按未知处理，不要臆造天气。")
        return "\n".join(lines)


def build_amap_tool_registry(
    api_key: Optional[str],
    max_calls: Optional[int] = None,
) -> tuple[ToolRegistry, ToolBudget]:
    """为一次规划请求构建工具注册表。

    返回 ``(registry, budget)``；没有配置高德 Key 时返回空注册表，
    Agent 会自动退回"纯上下文生成"模式。
    """
    budget = ToolBudget(PLANNER_TOOL_MAX_CALLS if max_calls is None else max_calls)
    registry = ToolRegistry()
    if not api_key:
        return registry, budget

    client = AmapPlannerClient(api_key)
    registry.register_tool(AmapPOISearchTool(client, budget), auto_expand=False)
    registry.register_tool(AmapWeatherTool(client, budget), auto_expand=False)
    return registry, budget
