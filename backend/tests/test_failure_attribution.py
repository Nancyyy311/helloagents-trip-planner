"""失败归因与思考模式回归测试。

覆盖这轮排查出来的三件事：
1. 失败要能按病因分开（否则会把"截断"误判成"prompt 写得不好"）；
2. 并行路径失败时要带回 finish_reason（否则无法区分截断与 schema 错误）；
3. 思考模式必须能显式开关，且开启时输出预算要自动留出推理空间。

不烧额度：LLM 客户端用假对象替换。
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from app.agents.llm_tracing import LLMCallMeta, TracedLLM
from app.agents.planner_query import (
    PLANNER_OUTPUT_BASE_TOKENS,
    PLANNER_OUTPUT_TOKENS_PER_DAY,
    planner_max_output_tokens,
)
from app.api.observability import MetricsRegistry
from app.models.schemas import TripRequest
from app.planner import failure_kind as fk


def _request(days: int = 3) -> TripRequest:
    return TripRequest.model_validate(
        {
            "city": "北京",
            "start_date": "2026-11-10",
            "end_date": "2026-11-12",
            "travel_days": days,
            "transportation": "公共交通",
            "accommodation": "经济型酒店",
            "preferences": ["历史文化"],
            "free_text_input": "",
            "party": {
                "adults": 2,
                "children": 0,
                "elders": 0,
                "total": 2,
                "companion_type": "couple",
            },
            "budget_constraint": {
                "amount": 1500,
                "scope": "total",
                "currency": "CNY",
                "budget_level": "limited",
                "strictness": "soft",
            },
        }
    )


# --------------------------------------------------------------------------
# 病因分类
# --------------------------------------------------------------------------


def test_classify_empty_response():
    assert fk.classify_failure("") == fk.EMPTY
    assert fk.classify_failure("   \n  ") == fk.EMPTY


def test_classify_truncated_inside_string():
    """真实样本：结尾停在 '世界' 这种半句话上。"""
    raw = '{"city":"北京","days":[{"description":"明清两代皇宫，世界'
    assert fk.classify_failure(raw) == fk.TRUNCATED_IN_STRING


def test_classify_truncated_inside_number():
    """真实样本：结尾停在坐标数字上（"longitude": 116）。"""
    raw = '{"city":"北京","days":[{"hotel":{"location":{"longitude": 116'
    assert fk.classify_failure(raw) == fk.TRUNCATED_IN_NUMBER


def test_classify_truncated_after_comma_counts_as_number():
    """结尾是 `116.362578,` 这种"数值写完但值没给"的形态，也算值中断。"""
    raw = '{"city":"北京","days":[{"location":{"longitude": 116.362578,'
    assert fk.classify_failure(raw) == fk.TRUNCATED_IN_NUMBER


def test_classify_trailing_content_is_recoverable():
    """真实样本：完整 JSON 之后又重复输出了 `,"budget":{...}}`。

    之前会被判成 broken_structure（听起来像原文废了），
    实际上前面的对象完全可用，解析器扫描一下就能救回。
    """
    raw = (
        '{"city":"北京","start_date":"2026-11-10","end_date":"2026-11-12",'
        '"days":[],"overall_suggestions":"x"}'
        ',"budget":{"total":1511}}'
    )
    assert fk.classify_failure(raw) == fk.TRAILING_CONTENT
    assert fk.is_recoverable(fk.TRAILING_CONTENT)
    assert not fk.is_truncation(fk.TRAILING_CONTENT)


def test_finish_reason_length_wins_over_structure():
    """API 说被截断了就采信 API，不再靠结构猜。"""
    raw = '{"city":"北京'  # 引号未闭合，结构上属于截断在字符串里
    assert fk.classify_failure(raw) == fk.TRUNCATED_IN_STRING
    assert fk.classify_failure(raw, "length") == fk.TRUNCATED_LENGTH


def test_classify_schema_invalid_when_json_already_parsed():
    """结构没问题、只是不合 schema 的，不能算截断。"""
    raw = '{"city":"北京","days":[]}'
    assert fk.classify_failure(raw, "stop", json_parsed=True) == fk.SCHEMA_INVALID


def test_classify_plain_text_is_not_json():
    assert fk.classify_failure("抱歉，我无法完成这个请求。") == fk.NOT_JSON


def test_describe_returns_human_readable():
    for kind in (fk.EMPTY, fk.TRUNCATED_LENGTH, fk.TRAILING_CONTENT, fk.SCHEMA_INVALID):
        assert fk.describe(kind)
        assert fk.describe(kind) != kind


# --------------------------------------------------------------------------
# TracedLLM：元信息捕获 + 思考开关
# --------------------------------------------------------------------------


class _FakeCompletions:
    def __init__(self, content="{}", finish_reason="stop", reasoning_tokens=0):
        self.content = content
        self.finish_reason = finish_reason
        self.reasoning_tokens = reasoning_tokens
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=self.reasoning_tokens),
        )
        message = SimpleNamespace(content=self.content)
        choice = SimpleNamespace(message=message, finish_reason=self.finish_reason)
        return SimpleNamespace(choices=[choice], usage=usage, model=kwargs.get("model"))


def _make_llm(thinking, completions):
    """绕过父类 __init__（它要求真实 API key），只装配 invoke 需要的属性。"""
    llm = object.__new__(TracedLLM)
    llm.model = "deepseek-v4-flash"
    llm.temperature = 0.2
    llm.max_tokens = 100
    llm.thinking = thinking
    llm._call_meta = threading.local()
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return llm


def test_traced_llm_records_meta():
    completions = _FakeCompletions(content='{"ok":1}', finish_reason="stop", reasoning_tokens=7)
    llm = _make_llm(False, completions)
    out = llm.invoke([{"role": "user", "content": "hi"}])
    meta = llm.last_call

    assert out == '{"ok":1}'
    assert meta is not None
    assert meta.finish_reason == "stop"
    assert meta.completion_tokens == 50
    assert meta.reasoning_tokens == 7
    assert meta.content_chars == len('{"ok":1}')
    assert meta.reasoning_share == pytest.approx(0.14)


def test_traced_llm_injects_thinking_disabled():
    completions = _FakeCompletions()
    llm = _make_llm(False, completions)
    llm.invoke([{"role": "user", "content": "hi"}])

    body = completions.calls[0]["extra_body"]
    assert body["thinking"] == {"type": "disabled"}


def test_traced_llm_injects_thinking_enabled():
    completions = _FakeCompletions()
    llm = _make_llm(True, completions)
    llm.invoke([{"role": "user", "content": "hi"}])

    assert completions.calls[0]["extra_body"]["thinking"] == {"type": "enabled"}


def test_traced_llm_leaves_thinking_untouched_when_none():
    """不指定时不干预，避免把"服务端默认"变成硬编码。"""
    completions = _FakeCompletions()
    llm = _make_llm(None, completions)
    llm.invoke([{"role": "user", "content": "hi"}])

    assert completions.calls[0]["extra_body"] is None


def test_traced_llm_meta_is_thread_local():
    """3 路候选并行共享一个 LLM 客户端，元信息不能串。"""
    completions = _FakeCompletions()
    llm = _make_llm(False, completions)
    seen = {}

    def worker(name):
        llm.invoke([{"role": "user", "content": name}])
        seen[name] = llm.last_call

    threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert set(seen) == {"t0", "t1", "t2"}
    assert all(seen[name] is not None for name in seen)


# --------------------------------------------------------------------------
# 输出预算：思考模式要额外留推理空间
# --------------------------------------------------------------------------


def test_output_budget_without_thinking(monkeypatch):
    monkeypatch.setenv("PLANNER_THINKING", "0")
    budget = planner_max_output_tokens(_request(3))
    assert budget == PLANNER_OUTPUT_BASE_TOKENS + 3 * PLANNER_OUTPUT_TOKENS_PER_DAY


def test_output_budget_adds_reasoning_reserve_when_thinking_on(monkeypatch):
    """开思考却不加预算，必然复现"正文被切断"的老问题。"""
    monkeypatch.setenv("PLANNER_THINKING", "1")
    monkeypatch.setenv("PLANNER_REASONING_RESERVE", "8000")
    budget = planner_max_output_tokens(_request(3))
    assert budget == PLANNER_OUTPUT_BASE_TOKENS + 3 * PLANNER_OUTPUT_TOKENS_PER_DAY + 8000


# --------------------------------------------------------------------------
# 指标：推理 token 要能被看见
# --------------------------------------------------------------------------


def test_metrics_track_reasoning_share():
    registry = MetricsRegistry()
    registry.record_llm_usage(
        LLMCallMeta(
            finish_reason="length",
            prompt_tokens=100,
            completion_tokens=8400,
            content_chars=480,
            model="deepseek-v4-flash",
            reasoning_tokens=7900,
        )
    )
    snapshot = registry.snapshot()

    assert snapshot["llm_completion_tokens"] == 8400
    assert snapshot["llm_reasoning_tokens"] == 7900
    assert snapshot["llm_reasoning_share"] == pytest.approx(0.9405, abs=1e-3)
    assert snapshot["llm_truncated"] == 1


def test_metrics_ignore_missing_meta():
    """拿不到元信息时不能把 null 当成 0 累加错误数据。"""
    registry = MetricsRegistry()
    registry.record_llm_usage(None)
    assert registry.snapshot()["llm_completion_tokens"] == 0
