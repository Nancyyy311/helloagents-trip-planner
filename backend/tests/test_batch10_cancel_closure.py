"""批次10：把取消能力闭环（前端取消入口的后端契约 + refine 取消）。

批次9 做出了取消能力，但留了两个缺口：
1. 前端没有"取消"按钮，`cancelPlanTask` 写了却没人调用——用户想主动放弃
   只能关掉整个页面（走 pagehide）。这需要一个可被调用的后端契约，
   本轮补的是"提交响应里能拿到 task_id"这一环（前端已在 onSubmitted 里接住）。
2. refine 是同步接口，没有 task_id 可寻址，于是完全无法取消——
   同样是四十秒的 LLM 调用，关页面照样跑完照扣钱。

这里的用例只覆盖后端契约（前端 UI 只能靠真浏览器验证）：
- DELETE /plan/{plan_id}/refine：没有 refine 在跑时 409，不是 200 假成功；
- refine 进行中被取消：返回 499 且**原计划不被改动**；
- 取消后令牌一定被注销（否则同一 plan 再也 cancel 不动，还会泄漏内存）。
"""

import threading
import time
import types

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routes import trip as trip_routes
from app.models.schemas import (
    Attraction,
    Budget,
    DayPlan,
    Hotel,
    Location,
    Meal,
    TripPlan,
)
from app.planner.cancel import CancelToken, PlanCancelled

client = TestClient(app)

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


def _trip(names_by_day, budget_total=1000):
    """构造一份合法 TripPlan（字段必须齐全，否则 TripPlan(**data) 会校验失败）。"""
    days = []
    for index, names in enumerate(names_by_day):
        days.append(
            DayPlan(
                date=f"2026-11-{10 + index:02d}",
                day_index=index,
                description=f"第{index + 1}天行程",
                transportation="公共交通",
                accommodation="经济型酒店",
                hotel=Hotel(name="同一家酒店", address="", estimated_cost=300),
                attractions=[
                    Attraction(
                        name=name,
                        address="",
                        location=Location(longitude=116.4, latitude=39.9),
                        visit_duration=120,
                        description="",
                    )
                    for name in names
                ],
                meals=[Meal(type="lunch", name=f"第{index + 1}天午餐", estimated_cost=50)],
            )
        )
    return TripPlan(
        city="北京",
        start_date="2026-11-10",
        end_date="2026-11-12",
        days=days,
        overall_suggestions="",
        budget=Budget(total=budget_total),
    )


@pytest.fixture(autouse=True)
def _clean_stores():
    with trip_routes._tasks_lock:
        trip_routes._tasks.clear()
    with trip_routes._plans_lock:
        trip_routes._plans.clear()
        trip_routes._refine_tokens.clear()
    yield
    with trip_routes._tasks_lock:
        trip_routes._tasks.clear()
    with trip_routes._plans_lock:
        trip_routes._plans.clear()
        trip_routes._refine_tokens.clear()


def _seed_plan(monkeypatch, plan_id="p" * 32):
    """造一个可直接 refine 的计划（绕过生成，省额度）。"""
    monkeypatch.setattr(
        trip_routes,
        "_plan_trip_isolated",
        lambda req, cancel_token=None: (_trip([["故宫"], ["天坛"], ["颐和园"]]), "llm_success", "ok"),
    )
    r = client.post("/api/trip/plan", json=SAMPLE)
    assert r.status_code == 200, r.text
    return r.json()["plan_id"]


# ------------------------------------------------------- refine 取消：反面契约
def test_cancel_refine_when_idle_returns_409(monkeypatch):
    """没有 refine 在跑时取消必须 409，不能回 200 假装成功。

    这条是防"接口存在但永远有效"的假象：调用方以为停住了，
    实际下一次改一句还是照样跑满四十秒。
    """
    plan_id = _seed_plan(monkeypatch)
    r = client.delete(f"/api/trip/plan/{plan_id}/refine")
    assert r.status_code == 409, r.text
    assert "没有正在进行的修改" in r.json()["detail"]


def test_cancel_refine_unknown_plan_returns_409_not_500():
    """不存在的 plan 也走 idle 分支（409），不该 500 或 404 泄露内部状态。"""
    r = client.delete("/api/trip/plan/does-not-exist/refine")
    assert r.status_code == 409, r.text


# ------------------------------------------------------- refine 取消：正面能力
def test_refine_cancel_aborts_before_llm_and_returns_499(monkeypatch):
    """refine 跑到一半被取消：返回 499，且原计划不被替换。

    真实场景：用户点了"改成室内"，等了 10 秒觉得不对，点取消。
    此时必须停下，且**上一版计划完好**——不能把半成品的计划写回去。
    """
    plan_id = _seed_plan(monkeypatch)
    original = client.get(f"/api/trip/plan/{plan_id}/refine").status_code  # 405，仅确认路由存在
    assert original in (405, 404)

    started = threading.Event()

    def slow_refine(req, prev, instruction, cancel_token=None):
        # 模拟"已经进入 LLM 调用"：先告诉主线程我开始了，再等取消信号
        started.set()
        # 真实实现里这里是阻塞的 LLM 调用；测试里用可打断的等待替代
        for _ in range(100):
            if cancel_token is not None and cancel_token.cancelled:
                cancel_token.checkpoint("before_llm:增量修改")
            time.sleep(0.05)
        return _trip([["故宫"], ["国家博物馆"], ["颐和园"]])

    monkeypatch.setattr(trip_routes, "_refine_isolated", slow_refine)

    result = {}

    def do_refine():
        result["response"] = client.post(
            f"/api/trip/plan/{plan_id}/refine", json={"instruction": "第二天换成室内"}
        )

    worker = threading.Thread(target=do_refine)
    worker.start()
    assert started.wait(5.0), "refine 没能进入执行阶段"
    time.sleep(0.1)

    r = client.delete(f"/api/trip/plan/{plan_id}/refine")
    assert r.status_code == 200, r.text
    assert r.json()["cancel"] in ("canceling", "already_canceling")

    worker.join(timeout=10.0)
    assert not worker.is_alive(), "取消之后 refine 没有停下"

    assert result["response"].status_code == 499, result["response"].text
    # 原计划必须完好：取消不是失败，更不是"用半成品覆盖"
    reloaded, _request = trip_routes._load_plan(plan_id)
    assert [a.name for a in reloaded.days[1].attractions] == ["天坛"]


def test_refine_cancel_after_llm_returns_499_not_200(monkeypatch):
    """LLM 已经在飞时取消：调用跑完后必须停下，不能照常交付结果。

    这是实测抓到的一个漏洞：`_call_llm_with_backoff` 只在**发请求之前**检查，
    调用本身拦不住；如果返回后不再查一次，refine 会无视取消照常返回 200——
    用户明明点了取消，却收到一份"只改了一半"的计划，`tasks_canceled` 还是 0。

    这里直接测 agent 的检查点，而不是打桩 `_refine_isolated`：
    打桩会绕过要验的那行代码，等于什么都没测。
    """
    from app.agents import trip_planner_agent as agent_module

    # 绕过 __init__（它需要真实 API key），只给 refine_trip 必要的属性
    planner = object.__new__(agent_module.MultiAgentTripPlanner)
    planner.amap_api_key = ""
    planner.planner_llm = None
    planner.tool_llm = None
    planner._cancel_token = None
    planner.planner_context_builder = types.SimpleNamespace(collect=lambda req: {})

    token = CancelToken()

    # 模块级函数替换掉，避免真的去打高德
    monkeypatch.setattr(agent_module, "build_refine_query", lambda *a, **k: "query")
    monkeypatch.setattr(
        agent_module,
        "build_amap_tool_registry",
        lambda key: (None, types.SimpleNamespace(max_calls=0)),
    )

    # 关键：模拟"调用已经在飞、期间被取消、然后正常返回"
    def fake_call(label, call):
        token.cancel("user_canceled")
        return "{}"

    monkeypatch.setattr(planner, "_call_llm_with_backoff", fake_call)
    # 如果检查点缺失，这行会被执行并返回一份"成功"的计划
    monkeypatch.setattr(
        planner,
        "_parse_response",
        lambda *a, **k: pytest.fail("取消之后不该再解析并交付结果"),
    )

    with pytest.raises(PlanCancelled) as info:
        planner.refine_trip(
            _sample_request(), _trip([["故宫"]]), "改一下", cancel_token=token
        )
    assert info.value.stage == "after_refine_llm"


def _sample_request():
    from app.models.schemas import TripRequest

    return TripRequest(**SAMPLE)


def test_refine_token_is_unregistered_after_finish(monkeypatch):
    """跑完（无论成功还是取消）令牌都必须注销。

    不注销有两个后果：同一 plan 之后 cancel 会打到上一个死令牌上（回
    already_canceling 但什么也没停），以及字典随 plan 数无限增长。
    """
    plan_id = _seed_plan(monkeypatch)
    monkeypatch.setattr(
        trip_routes,
        "_refine_isolated",
        lambda req, prev, i, cancel_token=None: (
            _trip([["故宫"], ["天坛"], ["颐和园"]]),
            "llm_success",
            "ok",
        ),
    )

    r = client.post(f"/api/trip/plan/{plan_id}/refine", json={"instruction": "微调"})
    assert r.status_code == 200, r.text
    assert plan_id not in trip_routes._refine_tokens


def test_cancel_refine_releases_gate_and_plan_lock(monkeypatch):
    """取消后并发闸门和 plan 占用都必须放掉。

    漏放闸门 = 取消几次就把并发位漏光（后面所有请求 503）；
    漏放 plan 占用 = 这个计划再也改不了（永远 409 busy）。
    """
    plan_id = _seed_plan(monkeypatch)
    started = threading.Event()

    def slow_refine(req, prev, instruction, cancel_token=None):
        started.set()
        while not (cancel_token and cancel_token.cancelled):
            time.sleep(0.02)
        cancel_token.checkpoint("before_llm:增量修改")

    monkeypatch.setattr(trip_routes, "_refine_isolated", slow_refine)

    worker = threading.Thread(
        target=lambda: client.post(
            f"/api/trip/plan/{plan_id}/refine", json={"instruction": "换酒店"}
        )
    )
    worker.start()
    assert started.wait(5.0)

    client.delete(f"/api/trip/plan/{plan_id}/refine")
    worker.join(timeout=10.0)

    assert trip_routes.plan_gate.snapshot()["inflight"] == 0
    with trip_routes._plans_lock:
        assert trip_routes._plans[plan_id]["refining"] is False


# --------------------------------------------- 提交响应：取消按钮需要 task_id
def test_async_submit_returns_task_id_that_can_be_canceled(monkeypatch):
    """提交任务必须立刻给出 task_id，且用它就能取消。

    这是前端"取消按钮"能工作的前提：拿不到 id，按钮就无从下手，
    后端的 DELETE 接口只能被 pagehide 这一个场景触发。

    注意：TestClient 会在 `post` 返回前把 BackgroundTasks 同步跑完，真实服务
    里它是在响应之后由线程池执行的。为了验证"提交返回的 id 可以直接用于取消"
    这个契约，这里把后台函数打桩成空操作，让任务停在 pending。
    """
    monkeypatch.setattr(trip_routes, "_run_plan_task", lambda task_id, request: None)

    sub = client.post("/api/trip/plan/async", json=SAMPLE)
    assert sub.status_code == 200, sub.text
    body = sub.json()
    task_id = body["task_id"]
    assert len(task_id) == 32
    # 响应里还必须给出轮询地址，前端不该自己拼路径
    assert body["poll"] == f"/api/trip/plan/async/{task_id}"

    r = client.delete(f"/api/trip/plan/async/{task_id}")
    assert r.status_code == 200, r.text
    assert r.json()["cancel"] == "canceled"
    # 取消后状态必须落到终态，否则前端会一直轮询下去
    assert r.json()["status"] == "canceled"
