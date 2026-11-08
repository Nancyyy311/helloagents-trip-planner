"""成本与并发治理：分级限流 + 规划并发闸门 + 请求合并。

为什么需要这一层（面试可直接讲）：
原项目对 `POST /api/trip/plan` 没有任何限额。每调一次就产生
3 个候选 × 各 1 次 LLM 调用 + 十余次高德查询 + 若干秒 CPU。也就是说
**任何一个前端死循环、任何一个脚本、任何一个好奇的路人都能把额度刷光**，
而且单进程的线程池会先被打满，连 `/health` 都探不通。

设计取舍：
1. 不引入 slowapi / redis / nginx 限流，零新依赖，纯内存实现，单机够用。
   要横向扩展时把 TokenBucket 换成 Redis 的 INCR + EXPIRE 即可，接口不变。
2. 用**令牌桶**而不是固定窗口：固定窗口在窗口边界会瞬时放行 2 倍流量
   （59 秒放 6 个 + 61 秒再放 6 个），桶的放行速率才是真的平滑。
3. 分两档：规划类接口贵（默认 6/分钟），其余接口便宜（默认 120/分钟）。
   不分档的话，要么把正常页面请求限死，要么对最贵的接口毫无约束。
4. 并发闸门用 threading 的锁而不是 asyncio.Semaphore：
   后台任务跑在 worker 线程里，事件循环里的同步接口和后台线程必须
   共用同一个闸门，否则两边的额度互相看不见。
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Optional


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


@dataclass
class LimitConfig:
    """限流配置。

    字段保持可写：测试要能把窗口收紧到几个请求来验证行为，
    不应该为此重新导入模块（模块级常量改不动，只能 monkeypatch 名字）。
    """

    enabled: bool = True
    # 规划类：一次请求贵在 3 个候选的 LLM 调用，桶给得很小
    plan_per_minute: int = 6
    plan_burst: int = 3
    # 其他接口：图片、POI 搜索、健康检查这类，只防刷不防正常使用
    default_per_minute: int = 120
    default_burst: int = 60
    # 只有明确部署在反向代理后面时才信任 X-Forwarded-For。
    # 默认不信任：否则任何人换个请求头就能绕过按 IP 限流。
    trust_proxy: bool = False


CONFIG = LimitConfig(
    enabled=_env_flag("RATE_LIMIT_ENABLED", True),
    plan_per_minute=_env_int("RATE_LIMIT_PLAN_PER_MINUTE", 6),
    plan_burst=_env_int("RATE_LIMIT_PLAN_BURST", 3),
    default_per_minute=_env_int("RATE_LIMIT_DEFAULT_PER_MINUTE", 120),
    default_burst=_env_int("RATE_LIMIT_DEFAULT_BURST", 60),
    trust_proxy=_env_flag("RATE_LIMIT_TRUST_PROXY", False),
)

# 单次请求要消耗几个令牌：规划接口确实贵得多
PLAN_PATHS = frozenset({
    "/api/trip/plan",
    "/api/trip/plan/async",
})
PLAN_PREFIX = "/api/trip/plan/"  # 含 /plan/{id}/refine

# 探活与文档不参与限流：负载均衡健康检查被限流会直接摘掉节点。
# 注意这里必须用后缀匹配：本项目真正的探活路径是 /api/trip/health，
# 只写死 "/health" 会让上线后每 3 次探活就有 1 次 429，节点被误判为不可用。
EXEMPT_PATH_SUFFIXES = ("/health",)
EXEMPT_PATHS = frozenset({"/", "/docs", "/redoc", "/openapi.json"})


class TokenBucket:
    """线程安全的令牌桶。

    每次请求按时间差补充令牌，再尝试扣除；扣不动就返回需要等待多久。
    这比"每 60 秒清零计数"更平滑，也不会在窗口边界双倍放行。
    """

    __slots__ = ("capacity", "refill_per_second", "_tokens", "_updated", "_lock")

    def __init__(self, capacity: float, refill_per_second: float) -> None:
        self.capacity = float(max(1.0, capacity))
        self.refill_per_second = float(max(refill_per_second, 1e-6))
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def take(self, amount: float = 1.0, now: Optional[float] = None):
        """尝试取走 amount 个令牌，返回 (是否放行, 需等待秒数, 剩余令牌)。"""
        moment = time.monotonic() if now is None else now
        with self._lock:
            elapsed = max(0.0, moment - self._updated)
            self._updated = moment
            self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_second)
            if self._tokens >= amount:
                self._tokens -= amount
                return True, 0.0, self._tokens
            deficit = amount - self._tokens
            return False, deficit / self.refill_per_second, self._tokens


# (client_ip, tier) -> 桶。key 里带 tier，规划额度和普通额度互不挤占。
_buckets: dict[tuple[str, str], TokenBucket] = {}
_buckets_lock = threading.Lock()
# 防止被大量伪造来源刷出无限多个桶（每个桶几十字节，一天足以堆到几十 MB）
_MAX_BUCKETS = 4096


def _bucket_for(client_ip: str, tier: str, now: float) -> TokenBucket:
    key = (client_ip, tier)
    with _buckets_lock:
        bucket = _buckets.get(key)
        if bucket is None:
            if len(_buckets) >= _MAX_BUCKETS:
                _prune_buckets_locked(now)
            if tier == "plan":
                bucket = TokenBucket(CONFIG.plan_burst, CONFIG.plan_per_minute / 60.0)
            else:
                bucket = TokenBucket(CONFIG.default_burst, CONFIG.default_per_minute / 60.0)
            _buckets[key] = bucket
        return bucket


def _prune_buckets_locked(now: float) -> None:
    """丢掉已经回满的桶：它们等价于新建一个，留着只占内存。"""
    stale = [
        key
        for key, bucket in _buckets.items()
        if now - bucket._updated > 900  # 15 分钟没动过
    ]
    for key in stale:
        _buckets.pop(key, None)


def classify(path: str) -> Optional[str]:
    """返回该路径的限流档位；None 表示不计限流。"""
    if path in EXEMPT_PATHS:
        return None
    if path.endswith(EXEMPT_PATH_SUFFIXES):
        # /health、/api/trip/health、/api/map/health 都算探活
        return None
    if path in PLAN_PATHS or path.startswith(PLAN_PREFIX):
        return "plan"
    return "default"


def check(client_ip: str, tier: str, amount: float = 1.0):
    """按 IP + 档位取令牌，返回 (是否放行, 需等待秒数, 剩余)。"""
    bucket = _bucket_for(client_ip, tier, time.monotonic())
    return bucket.take(amount)


def reset_limiter() -> None:
    """清空所有桶（测试用；也可用于运维时手动解封）。"""
    with _buckets_lock:
        _buckets.clear()


class RateLimitMiddleware:
    """纯 ASGI 限流中间件。

    没有用 BaseHTTPMiddleware：后者会为每个请求再包一层任务，
    在高并发下多一次不必要的 await 往返；纯 ASGI 只做一次字典查找。
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http" or not CONFIG.enabled:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        tier = classify(path)
        if tier is None:
            await self.app(scope, receive, send)
            return

        ip = client_ip_from_scope(scope)
        # 规划接口一次请求就值 3 次 LLM 调用，扣 2 个令牌；
        # 普通接口扣 1 个。用令牌数而不是再加一档配置，少一个旋钮。
        amount = 2.0 if tier == "plan" else 1.0
        allowed, retry_after, remaining = check(ip, tier, amount)

        if not allowed:
            # 延迟导入，避免 guard <-> observability 循环引用
            from .observability import metrics

            metrics.record_rate_limited(tier)
            body = json.dumps(
                {
                    "success": False,
                    "message": f"请求过于频繁（{tier} 档限流），请 {retry_after:.0f} 秒后重试",
                    "detail": {"tier": tier, "retry_after_seconds": round(retry_after, 1)},
                },
                ensure_ascii=False,
            ).encode("utf-8")
            headers = [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"content-length", str(len(body)).encode()),
                (b"retry-after", str(max(1, int(round(retry_after)))).encode()),
                (b"x-ratelimit-remaining", b"0"),
                (b"x-ratelimit-tier", tier.encode()),
            ]
            await send({"type": "http.response.start", "status": 429, "headers": headers})
            await send({"type": "http.response.body", "body": body})
            return

        await self.app(scope, receive, send)


def client_ip_from_scope(scope) -> str:
    """取客户端 IP；只有显式开启 trust_proxy 时才读 X-Forwarded-For。"""
    if CONFIG.trust_proxy:
        for name, value in scope.get("headers") or []:
            if name == b"x-forwarded-for":
                # 第一跳是真实客户端，后面的都是代理链
                first = value.decode("latin-1").split(",")[0].strip()
                if first:
                    return first
    client = scope.get("client")
    return client[0] if client else "unknown"


# === 规划并发闸门 ===
# 限流管的是"单位时间能进多少"，闸门管的是"同一时刻在做几个"。
# 两者缺一不可：1 分钟 6 次听着不多，但如果 6 个请求在第 1 秒同时进来，
# 每个都会开 3 个候选线程，进程照样被压垮。
class PlanningBusy(RuntimeError):
    """规划并发已满。"""


class PlanGate:
    """计数式并发闸门（可读、可观测，比信号量多一个 inflight 数字）。"""

    def __init__(self, limit: int) -> None:
        self._limit = max(1, limit)
        self._inflight = 0
        self._lock = threading.Lock()
        self._rejected = 0

    @property
    def limit(self) -> int:
        return self._limit

    def try_enter(self) -> bool:
        with self._lock:
            if self._inflight >= self._limit:
                self._rejected += 1
                return False
            self._inflight += 1
            return True

    def leave(self) -> None:
        with self._lock:
            if self._inflight > 0:
                self._inflight -= 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "limit": self._limit,
                "inflight": self._inflight,
                "rejected": self._rejected,
            }

    def reset(self, limit: Optional[int] = None) -> None:
        """测试用：清空计数，可顺带改上限。"""
        with self._lock:
            self._inflight = 0
            self._rejected = 0
            if limit is not None:
                self._limit = max(1, limit)


plan_gate = PlanGate(_env_int("PLANNER_MAX_CONCURRENCY", 2))


# === 请求合并（single flight）===
# 用户连点两次"生成"、或者前端重试，会发出两个完全相同的请求。
# 没有合并的话，两次各自跑 3 个候选，钱和时间都翻倍，结果还几乎一样。
# 做法：同一指纹的请求里，第一个当 leader 真跑，其余挂在这个 Future 上等结果。
class SingleFlight:
    """同一 key 的并发请求只执行一次，其余等待并复用结果。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inflight: dict[str, Future] = {}
        self.joined = 0

    def claim(self, key: str):
        """返回 (是否 leader, Future)。leader 负责执行并 finish。"""
        with self._lock:
            existing = self._inflight.get(key)
            if existing is not None:
                self.joined += 1
                return False, existing
            future: Future = Future()
            self._inflight[key] = future
            return True, future

    def finish(self, key: str, future: Future, *, result=None, error: Optional[BaseException] = None) -> None:
        """把结果交给等待者，并让出 key。必须先出表再 set："""
        with self._lock:
            self._inflight.pop(key, None)
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(result)

    def pending(self) -> int:
        with self._lock:
            return len(self._inflight)

    def reset(self) -> None:
        with self._lock:
            self._inflight.clear()
            self.joined = 0


single_flight = SingleFlight()
