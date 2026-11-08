"""预算归一化：钱不能交给 LLM 算。

事故现场
--------
评测 18 条行程（最终计划级，容差 2%），"预算上报总额 == 按 POI 单价重算的总额"
只有 **7 条**成立，**11 条不符**，累计差 **5553 元**；其中 **2 条**更糟——
上报显示在预算内，按单价算实际超支（最大 +1362 元），用户到店才会发现。
典型例子（北京 5 天 / 3 人 / 预算 7000）：

    模型上报 total_hotels = 2000      # 5 晚 × 500 × 1 间
    代码重算          = 4000      # 5 晚 × 500 × ceil(3/2)=2 间
    上报 total        = 6690
    真实 total        = 8362          # 少了 1672 元

prompt 第 8 条明明写了 `rooms = ceil(party.total/2)`，模型这一轮就是没照做。
这不是"模型笨"，是**任务分错了**：选 POI 需要判断力，乘法不需要。LLM 做乘法
的失败率永远不是 0，而用户对账单一秒钟就能发现——属于"错一次就失信"的字段。

做法
----
模型只负责"选了哪些 POI、单价是多少"，总额由代码按同一套规则重算并覆盖：
    total_hotels      = Σ hotel.estimated_cost × rooms
    total_attractions = Σ attraction.ticket_price × party.total
    total_meals       = Σ meal.estimated_cost × party.total
    total_transportation = 模型上报值（没有 POI 可依，保留）
    total             = 四项之和（算术自洽由结构保证，不再依赖模型）

副作用：rerank 拿到的是"真实价格"而不是"模型自称的价格"，
软预算超支才可能被选中之前就被扣分。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..models.schemas import Budget, TripPlan, TripRequest
from .rerank import recompute_budget_from_selected_items


def normalize_budget(trip_plan: TripPlan, request: TripRequest) -> Dict[str, Any]:
    """用 POI 单价重算预算并覆盖上报值。

    Returns:
        {"changed": bool, "before_total": int, "after_total": int, "delta": int}
        —— 调用方用它打日志/打点，避免"悄悄改数字"。
    """
    recomputed = recompute_budget_from_selected_items(trip_plan, request)
    total = int(recomputed.get("total") or 0)
    before_total = int((trip_plan.budget.total if trip_plan.budget else 0) or 0)

    # 全是 0（例如没有价格的兜底计划）时不做覆盖：此时"重算"没有信息量，
    # 硬覆盖会把用户看到的预算抹成 0，比不修更糟。
    if total <= 0:
        return {
            "changed": False,
            "before_total": before_total,
            "after_total": before_total,
            "delta": 0,
        }

    trip_plan.budget = Budget(
        total_attractions=int(recomputed["total_attractions"]),
        total_hotels=int(recomputed["total_hotels"]),
        total_meals=int(recomputed["total_meals"]),
        total_transportation=int(recomputed["total_transportation"]),
        total=total,
    )
    return {
        "changed": before_total != total,
        "before_total": before_total,
        "after_total": total,
        "delta": total - before_total,
    }


def describe_budget_fix(note: Optional[Dict[str, Any]]) -> str:
    """一行日志，别把"改了钱"这件事藏起来。"""
    if not note or not note.get("changed"):
        return ""
    return (
        f"💰 预算按单价重算: {note['before_total']} → {note['after_total']} "
        f"({note['delta']:+d})"
    )
