"""FastAPI主应用"""

import asyncio
import contextlib
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ..config import get_settings, validate_config, print_config
from ..planner.context import context_cache_stats
from ..planner.output import output_parse_stats
from .guard import RateLimitMiddleware, plan_gate, single_flight
from .observability import RequestContextMiddleware, metrics
from .routes import trip, poi, map as map_routes
from .routes.trip import sweep_task_store

# 获取配置
settings = get_settings()


async def _task_janitor_loop() -> None:
    """后台清扫协程：定期清理过期任务/计划，并回收"没人要了"的孤儿任务。

    为什么需要一个常驻循环，而不是在接口里顺手清：

    原实现的清理挂在 `_new_task()` 里，也就是**只有提交新任务时才会清理**。
    用户提交一个任务、看着它跑完、然后关掉页面再也不来了——这个任务就会一直
    躺到进程重启为止（TTL 写了 3600 秒，但没有任何人触发检查）。
    计划表 `_plans` 也是同一个毛病。

    另一个职责是孤儿回收：任务在跑，但既没有轮询也没有 SSE 订阅（超过宽限期），
    说明结果已经没人要了，继续跑只是白烧 LLM 额度。判定放在这里而不是请求里，
    是因为"没人来问"这件事本身就是没有请求——只能由后台发现。
    """
    while True:
        try:
            stats = sweep_task_store()
            if stats["reaped"] or stats["abandoned"]:
                print(
                    f"🧹 清扫: 清理过期任务 {stats['reaped']} 个, "
                    f"回收孤儿任务 {stats['abandoned']} 个",
                    flush=True,
                )
        except Exception as exc:  # 清扫失败绝不能拖垮主进程
            print(f"⚠️  后台清扫异常（忽略，下轮重试）: {exc}", flush=True)
        await asyncio.sleep(trip._TASK_JANITOR_INTERVAL)


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

    # === 改造点：后台清扫（过期回收 + 孤儿任务）===
    janitor = asyncio.create_task(_task_janitor_loop())
    print(
        f"🧹 后台清扫已启动：每 {trip._TASK_JANITOR_INTERVAL:.0f}s 一次，"
        f"孤儿任务宽限 {trip._TASK_ORPHAN_GRACE_SECONDS:.0f}s"
        f"（设为 0 可关闭回收）\n"
    )

    yield

    # ---- 关闭 ----
    janitor.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await janitor

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
# 顺序说明：先加的是内层。限流放在 RequestContext 内层，
# 是为了让被 429 拦下的请求也能拿到 X-Request-ID 并计入耗时统计——
# 否则"限流拦了多少"在请求日志里是查不到的。
app.add_middleware(RateLimitMiddleware)
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


@app.get("/health", tags=["运维"])
async def health():
    """健康检查（保留原路径，向后兼容）。"""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version
    }


# === 改造点：分级探活 ===
# 原来只有一个 /health，且它回答的是"进程还在吗"。这在 K8s 下是不够的：
#   - 存活探针（liveness）要的恰恰是"别管我能不能干活，进程没死就别重启我"，
#     用业务健康度当存活探针，一次依赖抖动就会把好好的进程反复重启；
#   - 就绪探针（readiness）要的是"现在能不能接活"，依赖没配齐就该把流量摘走。
# 两者混在一起的结果是：要么该摘流量时没摘（用户打到必然失败的节点），
# 要么不该重启时被重启（正在跑的规划全丢）。
@app.get("/healthz", tags=["运维"], summary="存活探针")
async def liveness():
    """存活探针：只回答"进程还在不在"，永远 200。

    刻意不做任何依赖检查——数据库/第三方挂了不代表这个进程该被重启。
    """
    return {"status": "alive", "service": settings.app_name}


@app.get("/readyz", tags=["运维"], summary="就绪探针")
async def readiness():
    """就绪探针：现在这个节点能不能真的接一个规划请求。

    三档语义（而不是简单的 200/500）：
    - `ready`：依赖齐全，可以接活 -> 200
    - `degraded`：能接，但有降级项（比如没配高德 Key，只能靠模型自身知识）-> 200
      **仍然返回 200 是刻意的**：能把计划生成出来就不该摘流量，
      降级信息给监控看，不参与摘流决策；
    - `not_ready`：缺关键依赖（没有 LLM Key），接了也必然失败 -> 503
    """
    checks: dict[str, object] = {}
    degraded: list[str] = []
    blocking: list[str] = []

    # 1. 主 LLM：没有它连兜底都跑不了，属于硬依赖
    has_llm = bool((settings.openai_api_key or "").strip())
    checks["llm_key"] = has_llm
    if not has_llm:
        blocking.append("llm_key")

    # 2. 个性化 Planner：配置了但 key 是空的，会静默退回默认模型，属于降级
    if settings.use_personalized_planner:
        has_personal = bool((settings.personalized_llm_api_key or "").strip())
        checks["personalized_llm"] = has_personal
        if not has_personal:
            degraded.append("personalized_llm")

    # 3. 高德 Key：缺了只是拿不到真实 POI/天气，计划仍能生成
    has_amap = bool((settings.amap_api_key or "").strip())
    checks["amap_key"] = has_amap
    if not has_amap:
        degraded.append("amap_key")

    # 4. 任务表水位：接近容量上限时新任务会淘汰旧任务，提前告警
    task_load = len(trip._tasks)
    task_ratio = round(task_load / trip._TASK_MAXSIZE, 3) if trip._TASK_MAXSIZE else 0.0
    checks["tasks"] = {"count": task_load, "max": trip._TASK_MAXSIZE, "ratio": task_ratio}
    if task_ratio >= 0.9:
        degraded.append("task_store_near_full")

    # 5. 并发水位：满了不代表不健康（那说明正在忙），但要让人看得见
    gate = plan_gate.snapshot()
    checks["planning_gate"] = gate
    if gate["inflight"] >= gate["limit"]:
        checks["planning_gate"]["saturated"] = True

    if blocking:
        status = "not_ready"
    elif degraded:
        status = "degraded"
    else:
        status = "ready"

    payload = {
        "status": status,
        "service": settings.app_name,
        "version": settings.app_version,
        "checks": checks,
        "degraded": degraded,
        "blocking": blocking,
    }
    if status == "not_ready":
        return JSONResponse(status_code=503, content=payload)
    return payload


@app.get("/metrics")
async def metrics_endpoint():
    """轻量指标快照：请求量/耗时 + LLM 成功率 + 缓存命中率 + JSON 解析成功率。"""
    snapshot = metrics.snapshot()
    # 缓存是"省钱又省时"的关键，但只有命中率能证明它真的有用
    snapshot["planner_context_cache"] = context_cache_stats()
    # 解析失败看不见，但每次失败都要重跑一次几十秒的 LLM 调用，必须量化
    snapshot["planner_json_parse"] = output_parse_stats()
    # 并发水位：当前有几个规划在跑、上限多少、拒了几个。
    # 这是"服务是不是快被压垮"最直接的信号。
    snapshot["planning_gate"] = plan_gate.snapshot()
    # 请求合并：pending 表示正在飞行的去重 key 数
    snapshot["single_flight"] = {
        "pending": single_flight.pending(),
        "joined": single_flight.joined,
    }
    # 任务表：有多少任务在跑、多少订阅者、多少已经跑完没人取。
    # "done 但没 watched" 的任务数是"白跑"最直接的证据。
    with trip._tasks_lock:
        tasks = list(trip._tasks.values())
    snapshot["task_store"] = {
        "total": len(tasks),
        "max": trip._TASK_MAXSIZE,
        "by_status": {
            status: sum(1 for t in tasks if t.get("status") == status)
            for status in ("pending", "running", "canceling", "done", "failed", "canceled")
        },
        "never_watched": sum(
            1 for t in tasks if not t.get("watched") and t.get("status") != "pending"
        ),
        "sse_subscribers": sum(int(t.get("sse_subscribers", 0)) for t in tasks),
    }
    return snapshot


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
