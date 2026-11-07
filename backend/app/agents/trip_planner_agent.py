"""多智能体旅行规划系统"""

import os
import random
import time
from typing import Callable, Dict, Any, List, Optional

from hello_agents import SimpleAgent
from ..services.llm_service import get_llm, get_planner_llm
from ..models.schemas import TripRequest, TripPlan
from ..config import get_settings
from ..api.observability import metrics
from .planner_feedback import (
    PLANNER_FAILURE_LOG,
    PlannerGenerationError,
    append_jsonl,
    build_failure_row,
    is_non_retryable_llm_error,
    log_preference_candidates,
)
from ..planner.context import PlannerContextBuilder
from ..planner.grounding import verify_trip_plan
from ..planner.output import (
    create_fallback_plan,
    enrich_trip_plan_poi_details,
    extract_json_object,
    validate_trip_plan_shape,
)
from ..planner.rerank import RerankCandidate, rerank_trip_plan_candidates
from .planner_query import build_planner_query, build_refine_query, planner_max_output_tokens
from .amap_tools import build_amap_tool_registry
from .prompts import (
    PLANNER_AGENT_PROMPT,
    PLANNER_AGENT_PROMPT_WITH_TOOLS,
    PLANNER_REFINE_PROMPT,
)


PLANNER_MAX_ATTEMPTS = 5
PLANNER_REQUEST_TIMEOUT = int(os.getenv("PLANNER_REQUEST_TIMEOUT", "600"))
PLANNER_TEMPERATURE = float(os.getenv("PLANNER_TEMPERATURE", "0.2"))
PLANNER_PRIMARY_CALL_FAILURE_LIMIT = int(os.getenv("PLANNER_PRIMARY_CALL_FAILURE_LIMIT", "3"))
PLANNER_FALLBACK_CALL_FAILURE_LIMIT = int(os.getenv("PLANNER_FALLBACK_CALL_FAILURE_LIMIT", "3"))
PLANNER_ENABLE_RERANK = os.getenv("PLANNER_ENABLE_RERANK", "1") == "1"
PLANNER_RERANK_CANDIDATE_COUNT = max(1, int(os.getenv("PLANNER_RERANK_CANDIDATE_COUNT", "3")))
PLANNER_RERANK_TEMPERATURE_STEP = float(os.getenv("PLANNER_RERANK_TEMPERATURE_STEP", "0.08"))

# === 改造点：增量修改（多轮会话）===
# 增量修改要的是"稳定、别发散"，所以温度比首次生成更低（0.1）。
# 首次生成有 3 个候选 + rerank 兜住质量，增量修改只有一次机会，必须稳。
PLANNER_REFINE_TEMPERATURE = float(os.getenv("PLANNER_REFINE_TEMPERATURE", "0.1"))

# === 改造点：真实工具调用 ===
# 原项目创建 SimpleAgent 时没传 tool_registry，框架里 enable_tool_calling 恒为 False，
# 所以"多智能体/工具调用"只是名义上的。这里给 Agent 挂上真实高德工具，
# 并限制工具轮次上限：每多一轮工具调用 = 多一次 LLM 往返（约 40s），必须设上限。
PLANNER_TOOL_MAX_ITERATIONS = max(1, int(os.getenv("PLANNER_TOOL_MAX_ITERATIONS", "2")))

# === 改造点：LLM 调用的限流退避重试 ===
# 实测现象：3 个候选并发打 DeepSeek 时，偶尔会有 1 个失败（日志里出现"有效 2/3"），
# 原因是瞬时并发触发了供应商限流。这类错误是"等一会儿就好"的，直接失败太亏。
# 策略：只对可重试错误做指数退避（2s、4s + 抖动），最多重试 2 次；
#       超时不重试——单次超时已等了几百秒，再重试只会让用户等更久，交给降级更合理。
PLANNER_LLM_MAX_RETRIES = max(0, int(os.getenv("PLANNER_LLM_MAX_RETRIES", "2")))
PLANNER_LLM_RETRY_BASE_DELAY = float(os.getenv("PLANNER_LLM_RETRY_BASE_DELAY", "2.0"))

# 命中这些关键字说明是"临时性故障/被限流"，值得重试
_RETRYABLE_LLM_HINTS = (
    "429",
    "rate limit",
    "too many requests",
    "500",
    "502",
    "503",
    "504",
    "internal server error",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
    "connection reset",
    "connection aborted",
    "temporarily unavailable",
    "overloaded",
)


class MultiAgentTripPlanner:
    """多智能体旅行规划系统"""

    def __init__(self):
        """初始化多智能体系统"""
        print("🔄 开始初始化多智能体旅行规划系统...")
        self.last_generation_status = "idle"
        self.last_generation_message = ""

        try:
            settings = get_settings()
            self.settings = settings
            self.amap_api_key = settings.amap_api_key
            self.planner_context_builder = PlannerContextBuilder(self.amap_api_key)
            self.tool_llm = get_llm()
            self.planner_llm = get_planner_llm()
            self.llm = self.tool_llm

            # 后端不再让工具 Agent 先写自然语言摘要。
            # 这里直接调用高德 HTTP API，拿到结构化快照后交给 Planner。
            print("  - 使用高德HTTP API获取结构化工具快照...")

            # 改造点：工具注册表在每次 plan_trip 时按请求构建（调用预算必须按请求隔离，
            # 否则并发请求会互相消耗对方的额度）。这里只做初始化占位。
            self._tool_registry = None
            self._tool_budget = None

            # 创建行程规划Agent（工具在 plan_trip 里按请求挂载）
            print("  - 创建行程规划Agent...")
            self.planner_agent = SimpleAgent(
                name="行程规划专家",
                llm=self.planner_llm,
                system_prompt=PLANNER_AGENT_PROMPT
            )
            self.fallback_planner_agent = SimpleAgent(
                name="默认行程规划专家",
                llm=self.tool_llm,
                system_prompt=PLANNER_AGENT_PROMPT
            )

            print("✅ 多智能体系统初始化成功")
            print("   工具查询: 高德HTTP API(景点/天气/酒店并行)")
            print(f"   Planner模型: {'个性化模型' if self.planner_llm is not self.tool_llm else '默认模型'}")

        except Exception as e:
            print(f"❌ 多智能体系统初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def plan_trip(self, request: TripRequest) -> TripPlan:
        """
        使用多智能体协作生成旅行计划

        Args:
            request: 旅行请求

        Returns:
            旅行计划
        """
        total_started_at = time.perf_counter()
        try:
            print(f"\n{'='*60}")
            print("🚀 开始多智能体协作规划旅行...")
            print(f"目的地: {request.city}")
            print(f"日期: {request.start_date} 至 {request.end_date}")
            print(f"天数: {request.travel_days}天")
            print(f"人数: {request.party.total}人 ({request.party.companion_type})")
            print(f"预算: {request.budget_constraint.amount or '未指定'} {request.budget_constraint.currency}")
            print(f"偏好: {', '.join(request.preferences) if request.preferences else '无'}")
            print(f"{'='*60}\n")

            # 步骤1: 并行获取结构化工具快照
            print("📍 步骤1: 并行获取景点/天气/酒店工具快照...")
            step_started_at = time.perf_counter()
            planner_context = self.planner_context_builder.collect(request)
            self._log_timer("步骤1 工具快照获取", step_started_at)
            self.planner_context_builder.print_summary(planner_context)
            self.planner_context_builder.print_visualization(planner_context)

            # 步骤2: 行程规划Agent整合信息生成计划
            print("📋 步骤2: 生成行程计划...")
            step_started_at = time.perf_counter()
            planner_query = build_planner_query(self.planner_context_builder, request, planner_context)
            self._log_timer("步骤2 Planner输入构造", step_started_at)
            print(f"Planner输入长度: {len(planner_query)} 字符")

            # === 改造点：挂载真实高德工具（把"伪 Agent"补成真 Agent）===
            # 上下文已经预抓好，所以工具只在"信息不足时补查"，不会每次请求都多花一轮 LLM。
            # 预算按请求隔离，避免并发请求互相吃掉对方额度。
            self._tool_registry, self._tool_budget = build_amap_tool_registry(self.amap_api_key)
            available_tools = self._tool_registry.list_tools() if self._tool_registry else []
            if available_tools:
                for bound_agent in (self.planner_agent, self.fallback_planner_agent):
                    bound_agent.tool_registry = self._tool_registry
                    bound_agent.enable_tool_calling = True
                    bound_agent.system_prompt = PLANNER_AGENT_PROMPT_WITH_TOOLS
                print(
                    f"🔧 已挂载真实工具: {available_tools}"
                    f"（单次预算 {self._tool_budget.max_calls} 次，最多 {PLANNER_TOOL_MAX_ITERATIONS} 轮）",
                    flush=True,
                )
            else:
                print("ℹ️  未配置高德 Key，跳过工具挂载（退回纯上下文生成）", flush=True)

            self.last_generation_status = "running"
            self.last_generation_message = "Planner生成中"
            primary_label = "个性化 Planner" if self.planner_llm is not self.tool_llm else "默认 Planner"
            used_planner_label = primary_label

            # 解析最终计划
            try:
                trip_plan = self._generate_trip_plan_with_retries(
                    agent=self.planner_agent,
                    planner_query=planner_query,
                    request=request,
                    label=primary_label,
                    max_attempts=PLANNER_MAX_ATTEMPTS,
                    planner_context=planner_context,
                    max_call_failures=PLANNER_PRIMARY_CALL_FAILURE_LIMIT,
                )
            except PlannerGenerationError as parse_error:
                if self.planner_llm is self.tool_llm:
                    raise
                print(f"⚠️  个性化 Planner 失败，将回退到默认 LLM: {parse_error}")
                used_planner_label = "默认 Planner"
                trip_plan = self._generate_trip_plan_with_retries(
                    agent=self.fallback_planner_agent,
                    planner_query=planner_query,
                    request=request,
                    label="默认 Planner",
                    max_attempts=PLANNER_MAX_ATTEMPTS,
                    planner_context=planner_context,
                    max_call_failures=PLANNER_FALLBACK_CALL_FAILURE_LIMIT,
                )
                log_preference_candidates(planner_query, request, trip_plan, parse_error.failures)
            except Exception as planner_error:
                if self.planner_llm is self.tool_llm:
                    raise
                print(f"⚠️  个性化 Planner 调用失败，将回退到默认 LLM: {planner_error}")
                used_planner_label = "默认 Planner"
                append_jsonl(
                    PLANNER_FAILURE_LOG,
                    build_failure_row(
                        label="个性化 Planner",
                        attempt=1,
                        max_attempts=PLANNER_MAX_ATTEMPTS,
                        request=request,
                        planner_query=planner_query,
                        response="",
                        error=planner_error,
                        preference_reason="planner_llm_call_failed",
                    ),
                )
                trip_plan = self._generate_trip_plan_with_retries(
                    agent=self.fallback_planner_agent,
                    planner_query=planner_query,
                    request=request,
                    label="默认 Planner",
                    max_attempts=PLANNER_MAX_ATTEMPTS,
                    planner_context=planner_context,
                    max_call_failures=PLANNER_FALLBACK_CALL_FAILURE_LIMIT,
                )

            self.last_generation_status = "llm_success"
            self.last_generation_message = f"{used_planner_label} 生成成功"
            print(f"{'='*60}")
            print(f"✅ 旅行计划生成完成: {self.last_generation_message}")
            self._log_timer("旅行计划总耗时", total_started_at)
            print(f"{'='*60}\n")

            return trip_plan

        except Exception as e:
            print(f"❌ 生成旅行计划失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self.last_generation_status = "fallback_success"
            self.last_generation_message = f"Planner失败，已返回fallback计划: {str(e)}"
            print(f"⚠️  {self.last_generation_message}")
            self._log_timer("旅行计划总耗时", total_started_at)
            return create_fallback_plan(request)

    def _log_timer(self, label: str, started_at: float) -> None:
        """打印轻量耗时日志，便于定位慢点。"""
        elapsed = time.perf_counter() - started_at
        print(f"[TIMER] {label}: {elapsed:.2f}s", flush=True)

    def refine_trip(
        self,
        request: TripRequest,
        previous_plan: TripPlan,
        instruction: str,
    ) -> TripPlan:
        """在已有计划上做增量修改（多轮会话）。

        与 plan_trip 的三个关键差异（也是面试会被追问的点）：

        1. **不做 3 候选 rerank**：增量修改要的是"最小改动"，跑 3 个候选再择优，
           最优解很可能是"另一份完全不同的计划"，反而把用户已经满意的部分改掉。
           单次生成 + 低温度（0.1）反而更可控。
        2. **复用 PlannerContext 缓存**：同一趟旅行的第 2、3 轮修改不会再调高德
           （30 分钟 TTL 正好覆盖一次会话），省掉约 10 秒取数。
        3. **失败直接抛错，不回退模板计划**：这里已经有上一版计划兜底，
           返回一份模板计划会把用户满意的内容覆盖掉，比报错更糟。

        Args:
            request: 原始旅行请求（用于复用上下文和校验）
            previous_plan: 上一版计划
            instruction: 自然语言修改要求

        Returns:
            修改后的新计划

        Raises:
            PlannerGenerationError: 模型输出无法解析为合法 TripPlan
        """
        total_started_at = time.perf_counter()
        print(f"\n{'='*60}")
        print("✏️  开始增量修改行程...")
        print(f"修改要求: {instruction}")
        print(f"{'='*60}\n")

        step_started_at = time.perf_counter()
        planner_context = self.planner_context_builder.collect(request)
        self._log_timer("增量修改 上下文获取(可命中缓存)", step_started_at)

        query = build_refine_query(request, previous_plan, instruction, planner_context)

        # 工具仍然挂上：用户可能要求"加一个候选池外的地点"，
        # 这时模型需要能查证，而不是凭空编一个。
        # 预算按请求独立，避免吃掉并发请求的额度。
        self._tool_registry, self._tool_budget = build_amap_tool_registry(self.amap_api_key)
        has_tools = bool(self._tool_registry and self._tool_registry.list_tools())
        refine_agent = SimpleAgent(
            name="行程修改专家",
            llm=self.planner_llm,
            system_prompt=PLANNER_REFINE_PROMPT,
            tool_registry=self._tool_registry,
            enable_tool_calling=has_tools,
        )
        if has_tools:
            print(f"🔧 增量修改已挂载工具: {self._tool_registry.list_tools()}", flush=True)

        call_started_at = time.perf_counter()
        response = self._call_llm_with_backoff(
            "增量修改",
            lambda: refine_agent.run(
                query,
                max_tokens=planner_max_output_tokens(request),
                temperature=PLANNER_REFINE_TEMPERATURE,
                timeout=PLANNER_REQUEST_TIMEOUT,
                max_tool_iterations=PLANNER_TOOL_MAX_ITERATIONS,
            ),
        )
        self._log_timer("增量修改 LLM调用", call_started_at)

        trip_plan = self._parse_response(
            response,
            request,
            use_fallback=False,
            planner_context=planner_context,
        )

        self.last_generation_status = "llm_success"
        self.last_generation_message = "增量修改成功"
        print("✅ 增量修改完成")
        self._log_timer("增量修改总耗时", total_started_at)
        print(f"{'='*60}\n", flush=True)
        return trip_plan

    @staticmethod
    def _is_retryable_llm_error(error: Exception) -> bool:
        """判断是不是"等一会儿就能好"的错误（限流 / 5xx / 连接被重置）。"""
        if is_non_retryable_llm_error(error):
            return False
        text = str(error).lower()
        return any(hint in text for hint in _RETRYABLE_LLM_HINTS)

    def _call_llm_with_backoff(
        self,
        label: str,
        call: Callable[[], str],
    ) -> str:
        """执行一次 LLM 调用；遇到可重试错误做指数退避，并记录调用成功率指标。

        call 是一个无参可调用对象（闭包），这样并行候选和串行兜底两条路径都能复用。
        """
        for attempt in range(PLANNER_LLM_MAX_RETRIES + 1):
            try:
                result = call()
                metrics.record_llm(True)
                return result
            except Exception as error:
                retryable = self._is_retryable_llm_error(error)
                if attempt >= PLANNER_LLM_MAX_RETRIES or not retryable:
                    metrics.record_llm(False)
                    raise
                # 指数退避 + 抖动：避免 3 个线程同一时刻重试又撞上限流
                delay = PLANNER_LLM_RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                print(
                    f"⚠️  {label} LLM调用失败(可重试): {error} "
                    f"-> {delay:.1f}s 后重试 {attempt + 1}/{PLANNER_LLM_MAX_RETRIES}",
                    flush=True,
                )
                time.sleep(delay)
        raise RuntimeError(f"{label} LLM调用重试耗尽")

    def _run_agent_stateless(self, agent: SimpleAgent, input_text: str, **kwargs) -> str:
        """Run a SimpleAgent without carrying history across trip requests."""
        if hasattr(agent, "_history"):
            agent._history = []

        try:
            return agent.run(input_text, **kwargs)
        finally:
            if hasattr(agent, "_history"):
                agent._history = []

    def _parallel_candidates(
        self,
        agent: SimpleAgent,
        planner_query: str,
        request: TripRequest,
        label: str,
        planner_context: Optional[Dict[str, Any]],
        target_candidate_count: int,
    ) -> tuple[List[tuple[int, TripPlan]], List[Dict[str, Any]]]:
        """并行生成多个候选计划（改造点：候选生成并行化）。

        原实现用 for 循环串行调用 LLM：3 个候选各约 40s，串起来约 122s。
        这里改成线程池并行，每个候选用**独立的 SimpleAgent 实例**（各自的
        _history，互不干扰），共享线程安全的 LLM 客户端（OpenAI SDK 客户端线程安全）。
        返回 (有效候选[(attempt, TripPlan)], 失败记录列表)。
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        llm = agent.llm
        max_output_tokens = planner_max_output_tokens(request)
        tool_registry = getattr(self, "_tool_registry", None)
        candidate_prompt = (
            PLANNER_AGENT_PROMPT_WITH_TOOLS if tool_registry and tool_registry.list_tools()
            else PLANNER_AGENT_PROMPT
        )
        temperatures = [
            min(
                0.95,
                max(0.0, PLANNER_TEMPERATURE + index * PLANNER_RERANK_TEMPERATURE_STEP),
            )
            for index in range(target_candidate_count)
        ]

        def _one(index: int, temperature: float) -> tuple[int, TripPlan]:
            candidate_agent = SimpleAgent(
                name=f"{label}-候选{index + 1}",
                llm=llm,
                system_prompt=candidate_prompt,
                tool_registry=tool_registry,
                # 有工具才真正开启工具循环；没工具时框架会自动忽略该开关
                enable_tool_calling=bool(tool_registry and tool_registry.list_tools()),
            )
            started_at = time.perf_counter()
            response = self._call_llm_with_backoff(
                f"{label} 候选{index + 1}",
                lambda: candidate_agent.run(
                    planner_query,
                    max_tokens=max_output_tokens,
                    temperature=temperature,
                    timeout=PLANNER_REQUEST_TIMEOUT,
                    max_tool_iterations=PLANNER_TOOL_MAX_ITERATIONS,
                ),
            )
            self._log_timer(f"{label} 候选{index + 1} LLM调用", started_at)
            trip_plan = self._parse_response(
                response,
                request,
                use_fallback=False,
                planner_context=planner_context,
            )
            return index + 1, trip_plan

        valid: List[tuple[int, TripPlan]] = []
        failures: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=target_candidate_count) as executor:
            future_map = {
                executor.submit(_one, index, temperatures[index]): index + 1
                for index in range(target_candidate_count)
            }
            for future in as_completed(future_map):
                attempt_index = future_map[future]
                try:
                    valid.append(future.result())
                except Exception as error:
                    print(
                        f"⚠️  {label} 候选{attempt_index} 生成/解析失败: {error}",
                        flush=True,
                    )
                    failure = build_failure_row(
                        label=label,
                        attempt=attempt_index,
                        max_attempts=target_candidate_count,
                        request=request,
                        planner_query=planner_query,
                        response="",
                        error=error,
                        preference_reason="parallel_candidate_failed",
                    )
                    failures.append(failure)
                    try:
                        append_jsonl(PLANNER_FAILURE_LOG, failure)
                    except Exception:
                        pass

        valid.sort(key=lambda item: item[0])
        # 改造点：候选池不足要变成指标。原项目只打一行日志，接口照样返回成功，
        # rerank 却在用更少的候选择优——质量下降被静默掩盖了。
        metrics.record_candidates(target_candidate_count, len(valid))
        if len(valid) < target_candidate_count:
            print(
                f"⚠️  候选池不足: 目标 {target_candidate_count}，有效 {len(valid)}，"
                f"rerank 择优空间变小",
                flush=True,
            )
        return valid, failures

    def _generate_trip_plan_with_retries(
        self,
        agent: SimpleAgent,
        planner_query: str,
        request: TripRequest,
        label: str,
        max_attempts: int,
        planner_context: Optional[Dict[str, Any]] = None,
        max_call_failures: Optional[int] = None,
    ) -> TripPlan:
        """Generate and parse a TripPlan with clean retries.

        这里故意不把上一次失败输出拼回下一次请求。PlannerContext 已经很长，
        修复式重试会快速放大上下文，并且容易把错误格式再次强化给模型。
        """
        last_error = None
        failures: List[Dict[str, Any]] = []
        call_failures = 0
        rerank_enabled = (
            PLANNER_ENABLE_RERANK
            and planner_context is not None
            and PLANNER_RERANK_CANDIDATE_COUNT > 1
        )
        target_candidate_count = (
            min(max_attempts, PLANNER_RERANK_CANDIDATE_COUNT)
            if rerank_enabled
            else 1
        )
        valid_candidates: List[tuple[int, TripPlan]] = []

        if rerank_enabled:
            print(
                f"{label} 启用rerank: 目标候选={target_candidate_count}, "
                f"temperature_step={PLANNER_RERANK_TEMPERATURE_STEP}",
                flush=True,
            )
            # === 改造点：候选并行生成（122s -> ~40s）===
            # 并行一次性把候选跑完；若并行整体出错，则自动回退到下面串行重试（兜底不丢）。
            try:
                parallel_valid, parallel_failures = self._parallel_candidates(
                    agent=agent,
                    planner_query=planner_query,
                    request=request,
                    label=label,
                    planner_context=planner_context,
                    target_candidate_count=target_candidate_count,
                )
                valid_candidates.extend(parallel_valid)
                failures.extend(parallel_failures)
                print(
                    f"⚡ {label} 并行候选完成: 有效 {len(valid_candidates)}/{target_candidate_count}",
                    flush=True,
                )
            except Exception as parallel_error:
                print(
                    f"⚠️  {label} 并行候选出错，回退串行重试: {parallel_error}",
                    flush=True,
                )

        # 并行已拿到候选时 range 为空 -> 跳过串行循环；否则走原串行重试兜底。
        for attempt in (range(1, max_attempts + 1) if not valid_candidates else []):
            print(f"{label} 第 {attempt}/{max_attempts} 次生成...", flush=True)
            call_started_at = time.perf_counter()
            max_output_tokens = planner_max_output_tokens(request)
            if rerank_enabled:
                temperature = min(
                    0.95,
                    max(0.0, PLANNER_TEMPERATURE + (attempt - 1) * PLANNER_RERANK_TEMPERATURE_STEP),
                )
            else:
                temperature = PLANNER_TEMPERATURE
            print(f"{label} 第 {attempt} 次max_tokens: {max_output_tokens}", flush=True)
            print(f"{label} 第 {attempt} 次temperature: {temperature:.2f}", flush=True)
            try:
                planner_response = self._call_llm_with_backoff(
                    f"{label} 第{attempt}次",
                    lambda: self._run_agent_stateless(
                        agent,
                        planner_query,
                        max_tokens=max_output_tokens,
                        temperature=temperature,
                        timeout=PLANNER_REQUEST_TIMEOUT,
                        # 串行兜底路径同样允许工具调用，否则两条路径行为不一致
                        max_tool_iterations=PLANNER_TOOL_MAX_ITERATIONS,
                    ),
                )
            except Exception as error:
                call_failures += 1
                self._log_timer(f"{label} 第 {attempt} 次LLM调用失败耗时", call_started_at)
                last_error = error
                failure = build_failure_row(
                    label=label,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    request=request,
                    planner_query=planner_query,
                    response="",
                    error=error,
                    preference_reason="planner_llm_call_failed",
                )
                failures.append(failure)
                append_jsonl(PLANNER_FAILURE_LOG, failure)
                print(f"⚠️  {label} 第 {attempt} 次调用失败: {error}", flush=True)

                # 400 内容风控类错误通常是服务商直接拒绝同一份输入，
                # 干净重试也不会改变结果，立刻切到下一个兜底模型更合理。
                if is_non_retryable_llm_error(error):
                    print(f"⚠️  {label} 遇到非重试型接口拒绝，停止当前模型重试", flush=True)
                    break

                # 本地个性化 Planner 如果服务不可用，连续重试只会拖慢线上响应。
                # 解析失败仍然走 max_attempts；纯接口调用失败按更小阈值快速切兜底。
                if max_call_failures is not None and call_failures >= max_call_failures:
                    print(
                        f"⚠️  {label} LLM调用失败已达 {call_failures} 次，停止当前模型重试",
                        flush=True,
                    )
                    break
                continue

            self._log_timer(f"{label} 第 {attempt} 次LLM调用", call_started_at)
            print(f"{label} 输出: {planner_response[:300]}...\n", flush=True)

            parse_started_at = time.perf_counter()
            try:
                trip_plan = self._parse_response(
                    planner_response,
                    request,
                    use_fallback=False,
                    planner_context=planner_context,
                )
                self._log_timer(f"{label} 第 {attempt} 次解析校验", parse_started_at)
                if not rerank_enabled:
                    print(f"✅ {label} 第 {attempt} 次输出解析成功")
                    if failures:
                        log_preference_candidates(planner_query, request, trip_plan, failures, label)
                    return trip_plan

                valid_candidates.append((attempt, trip_plan))
                print(
                    f"✅ {label} 第 {attempt} 次输出解析成功，"
                    f"候选收集 {len(valid_candidates)}/{target_candidate_count}",
                    flush=True,
                )
                if len(valid_candidates) >= target_candidate_count:
                    break
            except Exception as error:
                self._log_timer(f"{label} 第 {attempt} 次解析校验失败耗时", parse_started_at)
                last_error = error
                failure = build_failure_row(
                    label=label,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    request=request,
                    planner_query=planner_query,
                    response=planner_response,
                    error=error,
                )
                failures.append(failure)
                append_jsonl(PLANNER_FAILURE_LOG, failure)
                print(f"⚠️  {label} 第 {attempt} 次输出解析失败: {error}")

        if valid_candidates:
            ranked_candidates = rerank_trip_plan_candidates(valid_candidates, request, planner_context)
            selected_candidate = ranked_candidates[0]
            self._print_rerank_result(label, ranked_candidates)
            if failures:
                log_preference_candidates(
                    planner_query,
                    request,
                    selected_candidate.trip_plan,
                    failures,
                    label,
                )
            return selected_candidate.trip_plan

        raise PlannerGenerationError(
            f"{label} 未成功生成合法 TripPlan JSON: {last_error}",
            failures=failures,
        )

    def _print_rerank_result(self, label: str, ranked_candidates: List[RerankCandidate]) -> None:
        """打印 rerank 结果摘要，便于线上观察。"""
        if not ranked_candidates:
            return

        best = ranked_candidates[0]
        print(
            f"🏁 {label} rerank选中 attempt={best.attempt}, score={best.score:.2f}, "
            f"recomputed_total={best.metrics.get('recomputed_budget_total')}",
            flush=True,
        )
        for row in ranked_candidates:
            metrics = row.metrics
            print(
                "   "
                f"attempt={row.attempt} score={row.score:.2f} "
                f"fit={metrics.get('recomputed_budget_fit_ok')} "
                f"hard_budget={metrics.get('budget_hard_constraint_ok')} "
                f"meal_div={metrics.get('meal_diversity_ok')} "
                f"meal_ground={metrics.get('meal_grounding_rate')} "
                f"attr_ground={metrics.get('attraction_grounding_rate')}",
                flush=True,
            )

    def _parse_response(
        self,
        response: str,
        request: TripRequest,
        use_fallback: bool = True,
        planner_context: Optional[Dict[str, Any]] = None,
    ) -> TripPlan:
        """
        解析Agent响应

        Args:
            response: Agent响应文本
            request: 原始请求

        Returns:
            旅行计划
        """
        try:
            data = extract_json_object(response)

            # 转换为TripPlan对象
            trip_plan = TripPlan(**data)
            if planner_context:
                enrich_trip_plan_poi_details(trip_plan, planner_context)
                # 改造点：把 grounding 从"日志里的警告"变成"响应里的字段"。
                # 编造的景点/餐厅会被标注 verified=False，编造的坐标会被真实坐标覆盖。
                # 放在 validate 之前：后面的业务校验可以直接复用这份结果，不必重算。
                verify_trip_plan(trip_plan, planner_context)
                metrics.record_grounding(trip_plan.grounding)
            validate_trip_plan_shape(trip_plan, request, planner_context)

            return trip_plan

        except Exception as e:
            if not use_fallback:
                raise
            print(f"⚠️  解析响应失败: {str(e)}")
            print("   将使用备用方案生成计划")
            return create_fallback_plan(request)


# 全局多智能体系统实例
_multi_agent_planner = None


def get_trip_planner_agent() -> MultiAgentTripPlanner:
    """获取多智能体旅行规划系统实例(单例模式)"""
    global _multi_agent_planner

    if _multi_agent_planner is None:
        _multi_agent_planner = MultiAgentTripPlanner()

    return _multi_agent_planner
