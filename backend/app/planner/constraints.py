"""负向约束的落地检查（不是只写进 prompt 让模型自觉）。

背景
----
`policy.build_preference_profile()` 早就把"不要购物点""不吃辣"抽成了
`negative_constraints`，但它**只出现在 prompt 里**：全仓库 grep 确认没有任何
消费方，rerank 不打分、结果也不校验。后果是——模型这轮听话就没事，
这轮不听话就悄悄违反，用户只能自己逐条核对。

这里把约束变成**可计算的事实**：给定一份 TripPlan 和一组约束词，
返回具体违反了哪一条、第几天、哪个 POI。rerank 用它扣分（在 3 个候选里
挑最干净的那个），评测用它统计违反率。

设计取舍
--------
* 只做"名称级"匹配，不做语义理解：约束词 → 同义扩展 → 名称包含。
  好处是可解释、零成本、可单测；坏处是有假阳性（比如约束"爬山"，
  景点叫"香山"会被判违反——这其实是对的；但"景山公园"也算）。
  宁可严一点：漏判会让用户踩坑，误判只是排序时更保守。
* 扩展表集中在这里维护，评测和 rerank 共用同一份规则（单一事实源），
  否则"评测说没违反、排序却扣分"这种自相矛盾迟早发生。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

from ..models.schemas import TripPlan


# 约束词 → 名称里出现即视为违反的同义/上位词。
# 键是抽取出来的约束短语里可能包含的核心词。
TERM_EXPANSIONS: Dict[str, Tuple[str, ...]] = {
    "购物": (
        "购物", "商城", "商厦", "商业街", "步行街", "免税", "奥特莱斯",
        "百货", "购物中心", "商业广场",
    ),
    "商场": ("商场", "商城", "百货", "购物中心"),
    "辣": ("火锅", "麻辣", "川菜", "湘菜", "水煮", "香锅", "串串", "冒菜", "辣"),
    "爬山": ("登山", "徒步", "山顶", "山道"),
    # 注意：不要加"索道"。评测第一版加了，结果把重庆"长江索道"判成爬山——
    # 那是过江交通工具，不是登山。约束规则宁可漏判也不要误伤，
    # 否则 rerank 会为了躲开假阳性把合理候选排到最后。
    "登山": ("登山", "徒步", "山顶"),
    # 注意：不要加单字"宫"。它会命中"故宫"——中国最重要的景点之一，
    # 用户说"不想去寺庙"时故宫被判违规，rerank 会把它排到最后，属于
    # 灾难性误伤（同"索道"那条）。代价是漏判"雍和宫"，可接受。
    "寺庙": ("寺", "庙", "道观", "教堂"),
    "博物馆": ("博物馆", "博物院", "纪念馆", "展览馆"),
    "游乐园": ("乐园", "欢乐谷", "迪士尼", "游乐场"),
    "海边": ("海滩", "海滨", "浴场", "海岛"),
    "夜生活": ("酒吧", "夜店", "KTV", "夜总会", "livehouse"),
    "长途": ("长途", "包车", "一日游"),
    "排队": ("网红", "排队", "打卡"),
    "商业化": ("商业街", "步行街", "商城", "购物中心"),
}


def expand_terms(terms: Iterable[str]) -> List[str]:
    """把约束短语展开成名称匹配词，保留原始短语本身。"""
    expanded: List[str] = []
    for term in terms:
        clean = str(term or "").strip()
        if not clean:
            continue
        expanded.append(clean)
        for key, synonyms in TERM_EXPANSIONS.items():
            if key in clean:
                expanded.extend(synonyms)
    # 去重但保持顺序，方便测试断言
    seen = set()
    result = []
    for item in expanded:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _hit(name: str, terms: Sequence[str]) -> str:
    """返回命中的约束词，未命中返回空串。"""
    text = str(name or "")
    if not text:
        return ""
    for term in terms:
        if term and term in text:
            return term
    return ""


def find_constraint_violations(
    trip_plan: TripPlan,
    negative_constraints: Sequence[str],
) -> List[Dict[str, Any]]:
    """逐条检查行程是否违反负向约束。

    Returns:
        [{day, date, field, name, matched, constraint}]；无违反返回空列表。
    """
    terms = expand_terms(negative_constraints)
    if not terms:
        return []

    violations: List[Dict[str, Any]] = []
    for day in trip_plan.days:
        for attraction in day.attractions:
            matched = _hit(attraction.name, terms)
            if matched:
                violations.append(
                    {
                        "day": day.day_index + 1,
                        "date": day.date,
                        "field": "attraction",
                        "name": attraction.name,
                        "matched": matched,
                    }
                )
        for meal in day.meals:
            matched = _hit(meal.name, terms)
            if matched:
                violations.append(
                    {
                        "day": day.day_index + 1,
                        "date": day.date,
                        "field": "meal",
                        "name": meal.name,
                        "matched": matched,
                    }
                )
        if day.hotel is not None:
            matched = _hit(day.hotel.name, terms)
            if matched:
                violations.append(
                    {
                        "day": day.day_index + 1,
                        "date": day.date,
                        "field": "hotel",
                        "name": day.hotel.name,
                        "matched": matched,
                    }
                )
    return violations


def summarize_violations(violations: Sequence[Dict[str, Any]]) -> List[str]:
    """压成人类可读的一行行文本（日志/报告/接口提示都能用）。"""
    return [
        f"D{item['day']} {item['field']}「{item['name']}」命中约束「{item['matched']}」"
        for item in violations
    ]
