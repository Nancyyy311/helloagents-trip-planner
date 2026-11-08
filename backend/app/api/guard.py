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
    # 规划类：一次请求贵在 3 个候选的 LLM 调用，桶给得很小。
    # burst=4 是算出来的，不是拍的：一次自然的会话是"生成(2) + 改两句(1+1)"=4。
    # 之前是 3，配上"refine 也扣 2"的旧逻辑，用户改第二下必然 429（实测）。
    plan_per_minute: int = 6
    plan_burst: int = 4
    # 其他接口：图片、POI 搜索、健康检查这类，只防刷不防正常使用
    default_per_minute: int = 120
    default_burst: int = 60
    # 任务轮询：前端每 1.2 秒问一次进度，50 次/分钟只是"读一个内存字典"，
    # 成本约等于零。它和真正花钱的 /plan 必须分档，否则前端一改轮询，
    # 就会被自己的限流打死（实测：plan 档 6/分钟，第二次轮询就 429）。
    poll_per_minute: int = 240
    poll_burst: int = 120
    # 只有明确部署在反向代理后面时才信任 X-Forwarded-For。
    # 默认不信任：否则任何人换个请求头就能绕过按 IP 限流。
    trust_proxy: bool = False


CONFIG = LimitConfig(
    enabled=_env_flag("RATE_LIMIT_ENABLED", True),
    plan_per_minute=_env_int("RATE_LIMIT_PLAN_PER_MINUTE", 6),
    plan_burst=_env_int("RATE_LIMIT_PLAN_BURST", 4),
    default_per_minute=_env_int("RATE_LIMIT_DEFAULT_PER_MINUTE", 120),
    default_burst=_env_int("RATE_LIMIT_DEFAULT_BURST", 60),
    poll_per_minute=_env_int("RATE_LIMIT_POLL_PER_MINUTE", 240),
    poll_burst=_env_int("RATE_LIMIT_POLL_BURST", 120),
    trust_proxy=_env_flag("RATE_LIMIT_TRUST_PROXY", False),
)

# 单次请求要消耗几个令牌：规划接口确实贵得多
PLAN_PATHS = frozenset({
    "/api/trip/plan",
    "/api/trip/plan/async",
})
PLAN_PREFIX = "/api/trip/plan/"  # 含 /plan/{id}/refine
# 任务读接口（轮询 / SSE / 取消）：不触发任何 LLM 调用，单独一档
TASK_READ_PREFIX = "/api/trip/plan/async"

# 探活与文档不参与限流：负载均衡健康检查被限流会直接摘掉节点。
# 注意这里必须用后缀匹配：本项目真正的探活路径是 /api/trip/health，
# 只写死 "/health" 会让上线后每 3 次探活就有 1 次 429，节点被误判为不可用。
# `/healthz` `/readyz` 是批次9 新加的分级探针，同样必须豁免——
# 这两个是 15 秒一次的存活/就绪探测，被限流等于给自己制造故障。
EXEMPT_PATH_SUFFIXES = ("/health", "/healthz", "/readyz")
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
            elif tier == "poll":
                bucket = TokenBucket(CONFIG.poll_burst, CONFIG.poll_per_minute / 60.0)
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


def classify(path: str, method: str = "GET") -> Optional[str]:
    """返回该路径的限流档位；None 表示不计限流。

    改造点（批次9）：**按成本分档，而不是按路径前缀分档。**

    原来只要是 `/api/trip/plan/` 开头就算规划档（6 次/分钟）。这在同步接口时代
    没问题——那条路径下只有一个请求。加了任务模型之后，`GET /plan/async/{id}`
    （轮询进度）、`DELETE /plan/async/{id}`（取消）全被算进了规划档，
    而前端为了显示进度每 1.2 秒就要轮询一次：**每次生成第 2 次轮询就被 429**，
    用户看到的是"请求过于频繁"而不是"正在生成"。实测复现见 README 4.9。

    改造点（批次10）：**取消 refine 的 DELETE 也被误判成规划档。**
    同一类错误换了个位置——`DELETE /plan/{plan_id}/refine`（取消增量修改）
    不产生任何 LLM 调用，却被扣了 2 个令牌。实测：生成一次（扣 2）后
    再发一次取消，直接 429，取消请求根本到不了业务代码。

    判断依据改成"这次请求会不会触发 LLM 调用"：
    - POST /plan、POST /plan/async、POST /plan/{id}/refine -> plan（严格）
    - GET /plan/async/{id}、GET .../stream、DELETE /plan/async/{id}、
      DELETE /plan/{id}/refine -> poll（宽松）
    """
    if path in EXEMPT_PATHS:
        return None
    if path.endswith(EXEMPT_PATH_SUFFIXES):
        # /health、/api/trip/health、/api/map/health 都算探活
        return None
    # 任务读接口：轮询 1.2s 一次、SSE 长连接、取消请求，都不产生 LLM 调用。
    # 注意要多带一个 "/"：`/api/trip/plan/async` 本身是"提交任务"（POST），
    # 属于最贵的那一档，只有它下面的 /{task_id} 才是纯读。
    if method.upper() != "POST":
        if path.startswith(TASK_READ_PREFIX + "/"):
            return "poll"
        # 取消增量修改：同样只是给正在跑的线程置一个标志，零 LLM 成本
        if path.startswith(PLAN_PREFIX) and path.endswith("/refine"):
            return "poll"
    if path in PLAN_PATHS or path.startswith(PLAN_PREFIX):
        return "plan"
    return "default"


def is_refine_path(path: str) -> bool:
    """是不是"增量修改"（单次 LLM 调用），用于区分扣费额度。"""
    return path.startswith(PLAN_PREFIX) and path.endswith("/refine")


def charge_amount(path: str, tier: str) -> float:
    """这次请求要扣几个令牌 —— 按**真实 LLM 成本**折算，而不是按档位一刀切。

    - 首次生成：3 个候选各 1 次调用（rerank 模式），扣 2 个令牌；
    - 增量修改：只跑 1 次调用，扣 1 个。

    为什么必须区分：默认 `plan_burst=3`，如果两者都扣 2，
    "生成一次 + 改一句话"= 4 > 3 —— 用户改第二下就被自己的限流挡住。
    实测复现：生成成功后立刻 refine，返回 429，一次正常的"生成完微调"
    流程走不通。扣费对齐成本之后，"生成 + 改两次"正好 2+1+1=4，
    配合把 burst 提到 4 就能覆盖一次自然的会话。
    """
    if tier == "plan":
        return 1.0 if is_refine_path(path) else 2.0
    return 1.0


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
        tier = classify(path, scope.get("method", "GET"))
        if tier is None:
            await self.app(scope, receive, send)
            return

        ip = client_ip_from_scope(scope)
        # 按真实成本扣令牌：首次生成 2 个（3 候选），增量修改 1 个（单次调用），
        # 其余 1 个。见 charge_amount() 的注释——一刀切会让"生成完微调"直接 429。
        amount = charge_amount(path, tier)
        allowed, retry_after, remaining = check(ip, tier, amount)

        if not allowed:
            # 延迟导入，避免 guard <-> observability 循环引用
            from .observability import metrics

            metrics.record_rate_limited(tier)
            tier_label = {"plan": "规划", "poll": "进度查询", "default": "普通"}.get(
                tier, tier
            )
            body = json.dumps(
                {
                    "success": False,
                    "message": f"请求过于频繁（{tier_label}档限流），请 {retry_after:.0f} 秒后重试",
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
        # 每个 key 上挂了多少个等待者。原实现只记 joined 总数，
        # 没法回答"这个 key 现在还有人等吗"——而断连取消恰恰需要这个答案：
        # 发起人走了但别人还在等，就不能把这份结果作废。
        self._waiters: dict[str, int] = {}
        self.joined = 0

    def claim(self, key: str):
        """返回 (是否 leader, Future)。leader 负责执行并 finish。"""
        with self._lock:
            existing = self._inflight.get(key)
            if existing is not None:
                self.joined += 1
                self._waiters[key] = self._waiters.get(key, 0) + 1
                return False, existing
            future: Future = Future()
            self._inflight[key] = future
            self._waiters[key] = 0
            return True, future

    def has_waiters(self, key: str) -> bool:
        """这个 key 上是否还有别的请求在等结果。"""
        with self._lock:
            return self._waiters.get(key, 0) > 0

    def finish(self, key: str, future: Future, *, result=None, error: Optional[BaseException] = None) -> None:
        """把结果交给等待者，并让出 key。必须先出表再 set："""
        with self._lock:
            self._inflight.pop(key, None)
            self._waiters.pop(key, None)
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
            self._waiters.clear()
            self.joined = 0


single_flight = SingleFlight()
