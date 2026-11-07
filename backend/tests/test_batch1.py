"""批次1改造的回归测试：健康检查、错误语义、可观测性。

运行方式（在 backend 目录下）：
    venv\\Scripts\\python.exe -m pytest -q
"""

from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_app_health_ok():
    """应用级健康检查始终应为 200。"""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_trip_health_not_503():
    """改造点：trips 子服务健康检查不再因读不到 .agent 而恒 503。"""
    r = client.get("/api/trip/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "healthy"
    assert "agent_name" in body


def test_request_id_header():
    """改造点：每个响应都应带上 X-Request-ID，用于链路追踪。"""
    r = client.get("/health")
    assert "X-Request-ID" in r.headers
    assert r.headers["X-Request-ID"]


def test_metrics_endpoint():
    """改造点：/metrics 应能看到请求计数。"""
    client.get("/health")  # 先制造至少一次请求
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "total_requests" in data
    assert data["total_requests"] >= 1


def test_unknown_async_task_404():
    """改造点：异步任务查询，未知 task_id 应返回 404（不触发真实生成）。"""
    r = client.get("/api/trip/plan/async/does-not-exist")
    assert r.status_code == 404


def test_latency_by_path_isolated():
    """改造点：耗时按路由分别统计，探活请求不会稀释规划接口的真实耗时。

    只暴露一个全局平均值时，5 次请求里 4 次是毫秒级探活，
    40s 的规划会被算成 10s，看板数字直接失真。
    """
    from app.api.observability import MetricsRegistry

    reg = MetricsRegistry()
    reg.record("/health", 200, 2.0)
    reg.record("/health", 200, 4.0)
    reg.record("/api/trip/plan", 200, 40000.0)

    snap = reg.snapshot()
    by_path = snap["latency_by_path"]
    assert by_path["/health"]["count"] == 2
    assert by_path["/health"]["avg_ms"] == 3.0
    assert by_path["/api/trip/plan"]["avg_ms"] == 40000.0
    # 全局平均仍会被稀释，这正是要按路由看的原因
    assert snap["avg_latency_ms"] < 14000


def test_route_template_key_not_raw_url():
    """改造点：指标 key 用路由模板，避免 plan_id 这类路径参数撑爆指标基数。"""
    client.get("/api/trip/plan/abc123/refine")  # 不存在 plan_id，走 404 也没关系
    data = client.get("/metrics").json()
    keys = list(data["latency_by_path"].keys())
    assert not any("abc123" in k for k in keys), keys

