"""FastAPI主应用"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ..config import get_settings, validate_config, print_config
from ..planner.context import context_cache_stats
from ..planner.output import output_parse_stats
from .observability import RequestContextMiddleware, metrics
from .routes import trip, poi, map as map_routes

# 获取配置
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期（替代已弃用的 @app.on_event("startup"/"shutdown")）。"""
    # ---- 启动 ----
    print("\n" + "=" * 60)
    print(f"🚀 {settings.app_name} v{settings.app_version}")
    print("=" * 60)

    print_config()

    try:
        validate_config()
        print("\n✅ 配置验证通过")
    except ValueError as e:
        print(f"\n❌ 配置验证失败:\n{e}")
        print("\n请检查.env文件并确保所有必要的配置项都已设置")
        raise

    print("\n" + "=" * 60)
    print(f"📚 API文档: http://localhost:{settings.port}/docs")
    print(f"📖 ReDoc文档: http://localhost:{settings.port}/redoc")
    print("=" * 60 + "\n")

    yield

    # ---- 关闭 ----
    print("\n" + "=" * 60)
    print("👋 应用正在关闭...")
    print("=" * 60 + "\n")


# 创建FastAPI应用
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="基于HelloAgents框架的智能旅行规划助手API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求级 traceId + 内存指标（自研轻量版；生产可换 Prometheus / OpenTelemetry）
app.add_middleware(RequestContextMiddleware)

# 注册路由
app.include_router(trip.router, prefix="/api")
app.include_router(poi.router, prefix="/api")
app.include_router(map_routes.router, prefix="/api")


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version
    }


@app.get("/metrics")
async def metrics_endpoint():
    """轻量指标快照：请求量/耗时 + LLM 成功率 + 缓存命中率 + JSON 解析成功率。"""
    snapshot = metrics.snapshot()
    # 缓存是"省钱又省时"的关键，但只有命中率能证明它真的有用
    snapshot["planner_context_cache"] = context_cache_stats()
    # 解析失败看不见，但每次失败都要重跑一次几十秒的 LLM 调用，必须量化
    snapshot["planner_json_parse"] = output_parse_stats()
    return snapshot


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
