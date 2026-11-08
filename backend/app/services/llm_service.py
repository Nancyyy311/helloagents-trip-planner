"""LLM服务模块"""

import os
from contextlib import contextmanager
from typing import Iterator, Optional

from ..agents.llm_tracing import TracedLLM
from ..config import get_settings

# 全局LLM实例
_llm_instance = None
_planner_llm_instance = None


def _thinking_flag() -> Optional[bool]:
    """读取思考模式开关。

    `PLANNER_THINKING` 未设置时返回 None（不干预，沿用服务端默认），
    设为 0/1 时显式关闭/开启。

    为什么要默认关闭：DeepSeek V4 系列默认开思考，而思考 token 计入
    completion_tokens 并占用 max_tokens，实测导致 3 天行程的候选在
    480~3700 字符处被切断（历史 13 条有原文的失败样本里 12 条如此）。
    规划是"按候选池拼装 JSON"的任务，不需要长链推理；关掉之后
    输出预算全部留给正文，且 temperature 恢复生效（思考模式下它被忽略），
    候选多样性策略才真正起作用。
    """
    raw = os.getenv("PLANNER_THINKING", "0").strip().lower()
    if raw in ("", "auto"):
        return None
    return raw in ("1", "true", "yes", "on")


@contextmanager
def _temporary_env(overrides: dict[str, str]) -> Iterator[None]:
    """临时覆盖环境变量，用于创建独立 LLM 实例。"""
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def get_llm() -> TracedLLM:
    """
    获取LLM实例(单例模式)

    Returns:
        HelloAgentsLLM实例
    """
    global _llm_instance

    if _llm_instance is None:
        # HelloAgentsLLM会自动从环境变量读取配置
        # 包括OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL等
        _llm_instance = TracedLLM(thinking=_thinking_flag())

        print("✅ LLM服务初始化成功")
        print(f"   提供商: {_llm_instance.provider}")
        print(f"   模型: {_llm_instance.model}")
        print(f"   思考模式: {_llm_instance.thinking_effective}")

    return _llm_instance


def get_planner_llm() -> TracedLLM:
    """
    获取最终行程规划 LLM。

    默认返回通用 LLM；当 USE_PERSONALIZED_PLANNER=true 且配置完整时，
    仅 planner agent 使用个性化微调模型。
    """
    global _planner_llm_instance

    settings = get_settings()
    if not settings.use_personalized_planner:
        return get_llm()

    if _planner_llm_instance is None:
        model = os.getenv("PERSONALIZED_LLM_MODEL_ID") or settings.personalized_llm_model
        base_url = settings.personalized_llm_base_url
        api_key = settings.personalized_llm_api_key or "EMPTY"
        provider = settings.personalized_llm_provider or "openai"

        if not model or not base_url:
            print("⚠️  个性化 Planner 配置不完整，将使用默认 LLM")
            return get_llm()

        overrides = {
            "LLM_API_KEY": api_key,
            "OPENAI_API_KEY": api_key,
            "LLM_BASE_URL": base_url,
            "OPENAI_BASE_URL": base_url,
            "LLM_MODEL_ID": model,
            "OPENAI_MODEL": model,
            "LLM_PROVIDER": provider,
        }
        with _temporary_env(overrides):
            _planner_llm_instance = TracedLLM(thinking=_thinking_flag())

        print("✅ 个性化 Planner LLM 初始化成功")
        print(f"   提供商: {_planner_llm_instance.provider}")
        print(f"   模型: {_planner_llm_instance.model}")
        print(f"   思考模式: {_planner_llm_instance.thinking_effective}")
        print(f"   Base URL: {base_url}")

    return _planner_llm_instance


def reset_llm():
    """重置LLM实例(用于测试或重新配置)"""
    global _llm_instance, _planner_llm_instance
    _llm_instance = None
    _planner_llm_instance = None
