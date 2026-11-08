"""pytest 全局夹具：让接口用例不受限流与并发闸门干扰。

限流是有状态的（令牌桶按客户端 IP 累计），如果不重置，一个进程里跑几十个
接口用例，跑到一半就开始返回 429，症状是"单跑能过、整包跑就挂"。
专门验证限流的用例会自己在用例内部打开开关，跑完由这里兜底恢复。

另外两条纪律写在文件顶部，因为必须在 import app 之前生效：
1. 测试不允许依赖本地 .env（CI 和新克隆的仓库都没有这个文件）；
2. 测试不允许拿真实密钥跑网络请求。
"""

import os

# 批次16：注入占位配置，让测试在没有 .env 的环境里也能跑。
#
# 为什么需要：app/config.py 的 validate_config() 把"高德 key 缺失"当成硬错误
# 并 raise，启动钩子捕获后让 /api/trip/health、/readyz 返回 503。
# 于是 test_batch1 的"探活不该 503"和 test_batch9 的"探针绕过限流"会假失败——
# 失败原因不是代码错了，是跑测试的机器没配 key。
#
# 用 setdefault 而非直接赋值：开发机若已在 shell 里导出真实值就沿用，
# 但本地 .env 里的真实密钥不会进入测试进程以外的地方。
# 这两个值只为满足"非空"判断，任何真实网络调用都由用例自身 monkeypatch。
os.environ.setdefault("AMAP_API_KEY", "test-amap-key-not-used")
os.environ.setdefault("LLM_API_KEY", "test-llm-key-not-used")
# hello_agents 的 HelloAgentsLLM 自己读 LLM_BASE_URL / LLM_MODEL_ID，
# 两个都必须非空，否则构造 agent 时抛异常，健康检查连锁变 503。
os.environ.setdefault("LLM_BASE_URL", "https://llm.invalid/v1")
os.environ.setdefault("LLM_MODEL_ID", "test-model")

import pytest  # noqa: E402  （必须在设置环境变量之后导入）

from app.api import guard  # noqa: E402


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
