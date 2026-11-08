"""轻量可观测性：请求级 traceId + 内存指标。

设计取舍（面试可讲）：
- 不引入 Prometheus / OpenTelemetry，避免为一个单机项目增重；
- 用中间件给每个请求生成 request_id（贯穿日志 + 响应头），解决"出问题定位不到是哪次请求"；
- 用线程安全的内存计数器暴露 /metrics（请求总数、状态码分布、平均/最大耗时），
  生产规模上来后，替换成 prometheus-fastapi-instrumentator / OTel 即可，接口不变。
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from threading import Lock

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class MetricsRegistry:
    """线程安全的内存指标注册表。"""

    def __init__(self) -> None:
        self._lock = Lock()
        self.total_requests = 0
        self.status_counts: dict[str, int] = defaultdict(int)
        self.path_counts: dict[str, int] = defaultdict(int)
        # 改造点：按路由分别统计耗时。
        # 只留一个全局平均值会被健康检查（毫秒级）稀释：5 次请求里 4 次是探活，
        # 真实规划 40s 会被算成 10s，看板上的数字直接失真。
        # key 用「路由模板」（/api/trip/plan）而不是真实 URL（带 plan_id），
        # 否则每个 plan_id 都会新开一个 key，指标基数随请求量爆炸。
        self.path_latency: dict[str, dict[str, float]] = {}
        self.total_latency_ms = 0.0
        self.max_latency_ms = 0.0
        # LLM 调用指标：一次规划会发多个候选请求，"成功率"是最关键的运行指标之一
        self.llm_calls = 0
        self.llm_failures = 0
        # 工具调用指标：Agent 真正调工具的频次与失败率
        self.tool_calls = 0
        self.tool_failures = 0
        self.tool_calls_by_name: dict[str, int] = defaultdict(int)
        self.tool_budget_exhausted = 0
        # 输出预算指标：推理模型会把思考 token 算进 completion_tokens，
        # 也就占用 max_tokens。不单独看这一项，就无法解释"输出为什么没写完"。
        self.llm_completion_tokens = 0
        self.llm_reasoning_tokens = 0
        self.llm_truncated = 0
        # 候选质量指标：目标生成 3 个候选，实际有效几个
        self.candidate_target = 0
        self.candidate_valid = 0
        self.candidate_shortfall = 0
        # 解析失败归因：按病因分开计数。
        # 只记"失败了几次"没用——截断要调输出预算，schema 不合法才该改 prompt，
        # 混在一起统计会一直把优化方向带偏。
        self.planner_failures_by_kind: dict[str, int] = defaultdict(int)
        # Grounding 指标：模型编造了多少 POI / 坐标
        self.grounding_checked = 0
        self.grounding_verified = 0
        self.grounding_unverified = 0
        self.grounding_coordinates_corrected = 0

    def record(self, path: str, status: int, latency_ms: float) -> None:
        with self._lock:
            self.total_requests += 1
            self.status_counts[str(status)] += 1
            self.path_counts[path] += 1
            self.total_latency_ms += latency_ms
            if latency_ms > self.max_latency_ms:
                self.max_latency_ms = latency_ms
            slot = self.path_latency.get(path)
            if slot is None:
                self.path_latency[path] = {
                    "count": 1.0,
                    "total_ms": latency_ms,
                    "max_ms": latency_ms,
                }
            else:
                slot["count"] += 1
                slot["total_ms"] += latency_ms
                if latency_ms > slot["max_ms"]:
                    slot["max_ms"] = latency_ms

    def record_llm(self, ok: bool) -> None:
        """记录一次 LLM 调用结果（含重试后的最终结果）。"""
        with self._lock:
            self.llm_calls += 1
            if not ok:
                self.llm_failures += 1

    def record_llm_usage(self, meta) -> None:
        """记录一次调用的 token 用量与截断情况。

        `reasoning_share` 是这轮排障的关键指标：它一旦接近 1，说明额度全被
        思考吃掉了，正文必然写不完，此时该关思考或加大 max_tokens，
        而不是去改 prompt。
        """
        if meta is None:
            return
        with self._lock:
            self.llm_completion_tokens += getattr(meta, "completion_tokens", 0) or 0
            self.llm_reasoning_tokens += getattr(meta, "reasoning_tokens", 0) or 0
            if getattr(meta, "finish_reason", None) == "length":
                self.llm_truncated += 1

    def record_tool(self, name: str, ok: bool) -> None:
        """记录一次工具调用结果。"""
        with self._lock:
            self.tool_calls += 1
            self.tool_calls_by_name[name] += 1
            if not ok:
                self.tool_failures += 1

    def record_tool_budget_exhausted(self) -> None:
        """记录一次"工具预算用完被拦下"。这个数不为 0，说明模型在过度调工具。"""
        with self._lock:
            self.tool_budget_exhausted += 1

    def record_candidates(self, target: int, valid: int) -> None:
        """记录候选池情况：目标几个、实际有效几个。

        原项目把这层信息只打印在日志里，接口照样返回成功；
        候选不足会直接降低 rerank 的择优质量，必须变成指标才看得见。
        """
        with self._lock:
            self.candidate_target += target
            self.candidate_valid += valid
            if valid < target:
                self.candidate_shortfall += 1

    def record_planner_failure(self, kind: str) -> None:
        """记录一次 Planner 解析失败的病因（见 planner/failure_kind.py）。"""
        with self._lock:
            self.planner_failures_by_kind[kind or "unknown"] += 1

    def record_grounding(self, summary) -> None:
        """记录一次 Grounding 校验结果（模型编造率）。

        原项目算过 grounding，但结果只打印在日志里，线上无法持续观察；
        编造率不进指标，就等于没有对抗幻觉的手段。
        """
        if summary is None:
            return
        total = summary.attractions_total + summary.meals_total + summary.hotels_total
        verified = (
            summary.attractions_verified
            + summary.meals_verified
            + summary.hotels_verified
        )
        with self._lock:
            self.grounding_checked += total
            self.grounding_verified += verified
            self.grounding_unverified += max(0, total - verified)
            self.grounding_coordinates_corrected += summary.coordinates_corrected

    def snapshot(self) -> dict:
        with self._lock:
            avg = (
                self.total_latency_ms / self.total_requests
                if self.total_requests
                else 0.0
            )
            llm_success_rate = (
                (self.llm_calls - self.llm_failures) / self.llm_calls
                if self.llm_calls
                else 0.0
            )
            tool_failure_rate = (
                self.tool_failures / self.tool_calls if self.tool_calls else 0.0
            )
            candidate_valid_rate = (
                self.candidate_valid / self.candidate_target
                if self.candidate_target
                else 0.0
            )
            grounding_rate = (
                self.grounding_verified / self.grounding_checked
                if self.grounding_checked
                else 0.0
            )
            return {
                "total_requests": self.total_requests,
                "status_counts": dict(self.status_counts),
                "path_counts": dict(self.path_counts),
                "avg_latency_ms": round(avg, 2),
                "max_latency_ms": round(self.max_latency_ms, 2),
                # 改造点：按路由的耗时分布——全局 avg 会被探活请求稀释，这里才是真实口径
                "latency_by_path": {
                    path: {
                        "count": int(slot["count"]),
                        "avg_ms": round(slot["total_ms"] / slot["count"], 2),
                        "max_ms": round(slot["max_ms"], 2),
                    }
                    for path, slot in sorted(self.path_latency.items())
                },
                # 改造点：LLM 调用成功率——单看 HTTP 200 看不出模型在退化和限流
                "llm_calls": self.llm_calls,
                "llm_failures": self.llm_failures,
                "llm_success_rate": round(llm_success_rate, 4),
                # 改造点：输出预算是否被"思考"吃掉。推理 token 计入
                # completion_tokens 后，正文就容易半路被切断。
                "llm_completion_tokens": self.llm_completion_tokens,
                "llm_reasoning_tokens": self.llm_reasoning_tokens,
                "llm_reasoning_share": round(
                    self.llm_reasoning_tokens / self.llm_completion_tokens
                    if self.llm_completion_tokens
                    else 0.0,
                    4,
                ),
                "llm_truncated": self.llm_truncated,
                # 改造点：工具调用指标——证明 Agent 真的在调工具，而不是"伪 Agent"
                "tool_calls": self.tool_calls,
                "tool_failures": self.tool_failures,
                "tool_failure_rate": round(tool_failure_rate, 4),
                "tool_calls_by_name": dict(self.tool_calls_by_name),
                "tool_budget_exhausted": self.tool_budget_exhausted,
                # 改造点：候选有效率——候选不足会静默拉低 rerank 质量
                "candidate_target": self.candidate_target,
                "candidate_valid": self.candidate_valid,
                "candidate_valid_rate": round(candidate_valid_rate, 4),
                "candidate_shortfall": self.candidate_shortfall,
                # 改造点：把解析失败按病因拆开。`truncated_*` 说明输出没写完
                # （该调 max_tokens 或压缩输出），`schema_invalid` 才是 prompt 问题。
                "planner_failures_by_kind": dict(self.planner_failures_by_kind),
                # 改造点：Grounding——模型编造 POI 的比例，是 Agent 落地最关键的质检指标
                "grounding_checked": self.grounding_checked,
                "grounding_verified": self.grounding_verified,
                "grounding_unverified": self.grounding_unverified,
                "grounding_rate": round(grounding_rate, 4),
                "grounding_coordinates_corrected": self.grounding_coordinates_corrected,
            }


# 进程级单例
metrics = MetricsRegistry()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """为每个请求生成/透传 X-Request-ID，并记录耗时与状态码。"""

    async def dispatch(self, request: Request, call_next):
        # 允许上游（网关/前端）透传 request id，没有则生成一个短 id
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            latency_ms = (time.perf_counter() - started) * 1000
            metrics.record(_route_key(request), 500, latency_ms)
            print(f"[REQ] {request_id} {request.method} {request.url.path} -> 500 {latency_ms:.0f}ms")
            raise

        latency_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        metrics.record(_route_key(request), response.status_code, latency_ms)
        print(
            f"[REQ] {request_id} {request.method} {request.url.path} "
            f"-> {response.status_code} {latency_ms:.0f}ms"
        )
        return response


def _route_key(request: Request) -> str:
    """取路由模板作为指标 key，未命中路由（404）时退回真实路径。"""
    route = request.scope.get("route")
    return getattr(route, "path", None) or request.url.path
