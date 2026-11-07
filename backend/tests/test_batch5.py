"""批次5回归测试：真实工具调用 + 空壳服务修复。

不烧额度：所有外部调用（高德/LLM）都用假数据替换。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.agents.amap_tools import (
    AmapPOISearchTool,
    ToolBudget,
    build_amap_tool_registry,
)
from app.agents.prompts import PLANNER_AGENT_PROMPT, PLANNER_AGENT_PROMPT_WITH_TOOLS
from app.api.main import app
from app.api.observability import metrics
from app.services import amap_service as amap_service_module
from app.services.amap_service import AmapService

client = TestClient(app)


# --------------------------------------------------------------------- 提示词
def test_tool_prompt_removes_tool_ban():
    """原提示词禁止工具调用，工具版必须放开这条，否则挂了工具也用不了。"""
    assert "、前言、后记、工具调用或" in PLANNER_AGENT_PROMPT
    assert "、前言、后记、工具调用或" not in PLANNER_AGENT_PROMPT_WITH_TOOLS
    assert "工具使用规则" in PLANNER_AGENT_PROMPT_WITH_TOOLS


def test_tool_prompt_keeps_json_contract():
    """放开工具调用不能把 JSON 硬性要求弄丢。"""
    for keyword in ("只能输出一个 JSON 对象本身", "字段结构示例", "budget.total"):
        assert keyword in PLANNER_AGENT_PROMPT_WITH_TOOLS
    assert len(PLANNER_AGENT_PROMPT_WITH_TOOLS) > len(PLANNER_AGENT_PROMPT)


# ------------------------------------------------------------------- 工具预算
def test_tool_budget_blocks_after_limit():
    budget = ToolBudget(2)
    assert budget.acquire() is True
    assert budget.acquire() is True
    assert budget.acquire() is False
    assert budget.used == 2


class _FakeClient:
    """假高德客户端：按 path 返回固定 JSON。"""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, path, params):
        self.calls.append((path, params))
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def test_poi_tool_returns_real_names_and_warns_against_fabrication():
    fake = _FakeClient(
        {
            "pois": [
                {
                    "name": "西湖",
                    "type": "风景名胜",
                    "address": "西湖区龙井路1号",
                    "location": "120.148,30.248",
                }
            ]
        }
    )
    tool = AmapPOISearchTool(fake, ToolBudget(3))
    text = tool.run({"keywords": "西湖", "city": "杭州", "limit": 3})

    assert "西湖" in text
    assert "120.148,30.248" in text
    assert "禁止编造" in text
    assert fake.calls[0][0] == "/place/text"


def test_poi_tool_empty_result_tells_model_not_to_fabricate():
    tool = AmapPOISearchTool(_FakeClient({"pois": []}), ToolBudget(3))
    text = tool.run({"keywords": "不存在的地方", "city": "杭州"})
    assert "未检索到" in text
    assert "请不要编造" in text


def test_tool_failure_returns_string_instead_of_raising():
    """工具炸了不能让整轮生成报废，必须返回可读错误交给模型继续。"""
    tool = AmapPOISearchTool(_FakeClient(RuntimeError("上游超时")), ToolBudget(3))
    before = metrics.snapshot()
    text = tool.run({"keywords": "西湖", "city": "杭州"})
    after = metrics.snapshot()

    assert "工具执行失败" in text
    assert after["tool_calls"] - before["tool_calls"] == 1
    assert after["tool_failures"] - before["tool_failures"] == 1


def test_tool_budget_exhausted_is_counted():
    tool = AmapPOISearchTool(_FakeClient({"pois": []}), ToolBudget(1))
    tool.run({"keywords": "西湖", "city": "杭州"})
    before = metrics.snapshot()
    text = tool.run({"keywords": "灵隐寺", "city": "杭州"})
    after = metrics.snapshot()

    assert "预算已用完" in text
    assert after["tool_budget_exhausted"] - before["tool_budget_exhausted"] == 1


def test_registry_empty_without_api_key():
    """没配 Key 时不能假装有工具，否则提示词会诱导模型调用不存在的工具。"""
    registry, budget = build_amap_tool_registry("")
    assert registry.list_tools() == []
    assert budget.max_calls >= 0


# ---------------------------------------------------------------- 候选池指标
def test_candidate_shortfall_is_counted():
    before = metrics.snapshot()
    metrics.record_candidates(3, 2)
    after = metrics.snapshot()

    assert after["candidate_target"] - before["candidate_target"] == 3
    assert after["candidate_valid"] - before["candidate_valid"] == 2
    assert after["candidate_shortfall"] - before["candidate_shortfall"] == 1


# ---------------------------------------------------- AmapService 空壳修复
def test_search_poi_actually_returns_data(monkeypatch):
    """原实现恒返回 []；修复后必须能解析出真实 POI。"""
    fake = _FakeClient(
        {
            "pois": [
                {
                    "id": "B000A8UIN8",
                    "name": "故宫博物院",
                    "type": "风景名胜",
                    "address": "景山前街4号",
                    "location": "116.397,39.918",
                    "tel": "010-85007421",
                },
                {"id": "X", "name": "没有坐标的记录", "type": "", "address": ""},
            ]
        }
    )
    monkeypatch.setattr(amap_service_module, "_get_client", lambda: fake)

    pois = AmapService().search_poi("故宫", "北京")
    assert len(pois) == 1  # 无坐标的记录被丢弃
    assert pois[0].name == "故宫博物院"
    assert pois[0].location.longitude == pytest.approx(116.397)
    assert pois[0].tel == "010-85007421"


def test_get_weather_actually_returns_data(monkeypatch):
    fake = _FakeClient(
        {
            "forecasts": [
                {
                    "casts": [
                        {
                            "date": "2026-11-20",
                            "dayweather": "晴",
                            "nightweather": "多云",
                            "daytemp": "18",
                            "nighttemp": "9",
                            "daywind": "北风",
                            "daypower": "3级",
                        }
                    ]
                }
            ]
        }
    )
    monkeypatch.setattr(amap_service_module, "_get_client", lambda: fake)

    weather = AmapService().get_weather("北京")
    assert len(weather) == 1
    assert weather[0].day_weather == "晴"
    assert weather[0].day_temp == 18


def test_plan_route_parses_distance_and_duration(monkeypatch):
    fake = _FakeClient(
        {"route": {"paths": [{"distance": "1200", "duration": "900", "steps": [1, 2, 3]}]}}
    )
    monkeypatch.setattr(amap_service_module, "_get_client", lambda: fake)

    route = AmapService().plan_route("A", "B", route_type="walking")
    assert route["distance"] == pytest.approx(1200.0)
    assert route["duration"] == 900
    assert "3 段" in route["description"]


def test_map_health_is_503_when_upstream_degraded(monkeypatch):
    """原实现永远返回 healthy；现在上游不通必须 503，让网关能摘流量。"""
    monkeypatch.setattr(
        AmapService,
        "probe",
        lambda self: {"status": "degraded", "reason": "高德地理编码返回空结果"},
    )
    response = client.get("/api/map/health")
    assert response.status_code == 503


def test_map_health_is_200_when_upstream_ok(monkeypatch):
    monkeypatch.setattr(
        AmapService,
        "probe",
        lambda self: {"status": "healthy", "upstream": "amap-http"},
    )
    response = client.get("/api/map/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_metrics_expose_tool_fields():
    snapshot = client.get("/metrics").json()
    for field in (
        "tool_calls",
        "tool_failures",
        "tool_failure_rate",
        "tool_calls_by_name",
        "tool_budget_exhausted",
        "candidate_target",
        "candidate_valid",
        "candidate_valid_rate",
        "candidate_shortfall",
    ):
        assert field in snapshot, f"/metrics 缺少字段: {field}"
    assert isinstance(snapshot["tool_calls_by_name"], dict)


def test_openapi_still_buildable():
    """改了 Query 的 examples 写法后，OpenAPI 文档必须还能正常生成。"""
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    assert json.dumps(schema.json())[:1] == "{"
