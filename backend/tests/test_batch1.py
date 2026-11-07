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
