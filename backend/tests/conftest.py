"""pytest 全局夹具：让接口用例不受限流与并发闸门干扰。

限流是有状态的（令牌桶按客户端 IP 累计），如果不重置，一个进程里跑几十个
接口用例，跑到一半就开始返回 429，症状是"单跑能过、整包跑就挂"。
专门验证限流的用例会自己在用例内部打开开关，跑完由这里兜底恢复。
"""

import pytest

from app.api import guard


@pytest.fixture(autouse=True)
def _isolate_cost_guard(monkeypatch):
    """每个用例开始前重置限流/合并/闸门状态，并默认关闭限流。"""
    guard.reset_limiter()
    guard.single_flight.reset()
    guard.plan_gate.reset(limit=2)
    monkeypatch.setattr(guard.CONFIG, "enabled", False)
    yield
    guard.reset_limiter()
    guard.single_flight.reset()
    guard.plan_gate.reset()
