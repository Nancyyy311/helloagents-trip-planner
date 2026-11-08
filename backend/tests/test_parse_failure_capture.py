"""回归测试：并行候选失败时必须保留模型的原始输出。

背景（这是我自己引入的回归）：
串行路径的 `except` 里能直接读到 `planner_response` 变量，所以失败样本是完整的；
我把它改成线程池并行之后，`response` 变成 `_one()` 的局部变量，
`future.result()` 抛异常时它已经出了作用域 —— 于是失败记录里 `rejected` 恒为空字符串，
`planner_feedback` 那套"用失败样本做 DPO 训练数据"的机制等于停摆，
而且线上出现 `candidate_shortfall` 时也没法回答"模型到底输出了什么"。

修法：让 `_one()` 把原始输出包进异常一起抛回主线程。
"""

from types import SimpleNamespace

import app.agents.trip_planner_agent as agent_mod
from app.agents.trip_planner_agent import PlannerCandidateError
from app.models.schemas import TripRequest

SAMPLE = {
    "city": "北京",
    "start_date": "2026-11-10",
    "end_date": "2026-11-12",
    "travel_days": 3,
    "transportation": "公共交通",
    "accommodation": "经济型酒店",
    "preferences": ["文化历史"],
    "free_text_input": "",
    "party": {"adults": 1, "children": 0, "elders": 0, "total": 1, "companion_type": "solo"},
    "budget_constraint": {
        "amount": 3000,
        "scope": "total",
        "currency": "CNY",
        "budget_level": "standard",
        "strictness": "none",
    },
}

# 模拟模型真实会吐出来的烂输出：带一句开场白 + 代码围栏 + 被截断的 JSON
BAD_OUTPUT = (
    '好的，以下是为你规划的行程：\n```json\n{"city": "北京", "start_date": "2026-11-10", '
    '"end_date": "2026-11-12", "days": [{"date": "2026-11-10", "day_index": 0'
)


def test_candidate_error_carries_raw_response():
    """异常对象必须能把原文带出去，否则上层永远拿不回来。"""
    err = PlannerCandidateError("解析失败", raw_response=BAD_OUTPUT)
    assert err.raw_response == BAD_OUTPUT
    assert str(err) == "解析失败"


def test_parallel_failure_keeps_raw_response(monkeypatch):
    """并行路径失败时，失败记录里的 rejected 必须是模型的真实输出。"""
    planner = object.__new__(agent_mod.MultiAgentTripPlanner)
    planner._tool_registry = None

    # 不需要真的建 Agent，也不需要真的调模型
    monkeypatch.setattr(agent_mod, "SimpleAgent", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(planner, "_call_llm_with_backoff", lambda label, fn: BAD_OUTPUT)
    monkeypatch.setattr(planner, "_log_timer", lambda *a, **k: None)

    written = []
    monkeypatch.setattr(agent_mod, "append_jsonl", lambda path, row: written.append(row))

    request = TripRequest(**SAMPLE)
    valid, failures = planner._parallel_candidates(
        agent=SimpleNamespace(llm=object()),
        planner_query="请规划北京三日游",
        request=request,
        label="测试Planner",
        planner_context={"city": "北京"},
        target_candidate_count=3,
    )

    assert valid == [], "坏输出不应该产出有效候选"
    assert len(failures) == 3
    assert len(written) == 3, "每条失败都要落一条 jsonl"

    for row in failures:
        assert row["rejected"] == BAD_OUTPUT, "失败样本丢了原文，这套机制就废了"
        # 归因要看根因类型，不能是被包装后的 PlannerCandidateError
        assert row["error_type"] == "ValueError"
        assert "TripPlan" in row["error"] or "JSON" in row["error"]


def test_parallel_llm_call_failure_does_not_fake_raw_response(monkeypatch):
    """LLM 调用本身就失败时没有原文，不能编一个空壳当原文。"""
    planner = object.__new__(agent_mod.MultiAgentTripPlanner)
    planner._tool_registry = None

    monkeypatch.setattr(agent_mod, "SimpleAgent", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(planner, "_log_timer", lambda *a, **k: None)

    def _boom(label, fn):
        raise RuntimeError("429 rate limit")

    monkeypatch.setattr(planner, "_call_llm_with_backoff", _boom)
    written = []
    monkeypatch.setattr(agent_mod, "append_jsonl", lambda path, row: written.append(row))

    _, failures = planner._parallel_candidates(
        agent=SimpleNamespace(llm=object()),
        planner_query="请规划北京三日游",
        request=TripRequest(**SAMPLE),
        label="测试Planner",
        planner_context={"city": "北京"},
        target_candidate_count=2,
    )

    assert len(failures) == 2
    for row in failures:
        assert row["rejected"] == ""
        assert row["error_type"] == "RuntimeError"
