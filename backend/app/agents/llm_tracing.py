"""带调用元信息的 LLM 封装。

为什么需要这一层
----------------
hello_agents 的 `HelloAgentsLLM.invoke()` 只 `return response.choices[0].message.content`，
把 `finish_reason` 和 `usage` 全丢了。这直接导致我们**无法区分两种完全不同的失败**：

1. `finish_reason == "length"`：输出被 max_tokens 截断，JSON 天然不合法。
   补救方向是调大输出预算、压缩输出体积（少写描述），改 prompt 毫无用处。
2. `finish_reason == "stop"`：模型正常收尾，但 JSON 不符合 schema。
   这才是 prompt / 格式约束的问题。

项目原来的报错一律是"响应中未找到完整的顶层TripPlan JSON对象"，两种病因混在
一个错误里，于是失败样本只能靠猜——这正是候选成功率卡在 2/3 却迟迟无从下手的原因。

这里不改第三方库（venv 里的代码升级即丢），而是子类化，重写同一段调用逻辑，
把元信息额外记下来，供上层归因。

并发安全
--------
候选计划是 3 路线程池并行、共享同一个 LLM 客户端的，所以元信息必须**按线程隔离**
（`threading.local`），否则 A 候选的 finish_reason 会被 B 候选覆盖。
"""

from dataclasses import dataclass
from threading import local
from typing import Any, List, Optional

from hello_agents import HelloAgentsLLM
from hello_agents.core.exceptions import HelloAgentsException


@dataclass
class LLMCallMeta:
    """一次 LLM 调用的可观测元信息。"""

    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    content_chars: int
    model: str
    # 推理模型（DeepSeek V4 系列默认开启 thinking）会把思考过程算进
    # completion_tokens，而 completion_tokens 又受 max_tokens 限制。
    # 不单独记这一项，就永远看不出"输出为什么写到一半断了"。
    reasoning_tokens: int = 0

    @property
    def truncated(self) -> bool:
        """是否因为输出长度上限被动截断。"""
        return self.finish_reason == "length"

    @property
    def reasoning_share(self) -> float:
        """推理 token 占 completion 的比例，用来判断预算是否被思考吃掉。"""
        if self.completion_tokens <= 0:
            return 0.0
        return self.reasoning_tokens / self.completion_tokens

    def as_dict(self) -> dict:
        return {
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "content_chars": self.content_chars,
            "model": self.model,
            "reasoning_tokens": self.reasoning_tokens,
        }


class TracedLLM(HelloAgentsLLM):
    """在父类基础上补记 finish_reason / token 用量 / 推理 token。

    只重写 `invoke`（非流式），流式走父类实现——规划链路不用流式。
    异常消息保持与父类一致（`LLM调用失败: ...`），因为
    `is_non_retryable_llm_error()` 依赖这段前缀做 401/额度判断。

    关于 thinking 开关
    ------------------
    DeepSeek V4 系列**默认开启思考模式**，且有两个副作用直接坑到本项目：

    1. 思考 token 计入 `completion_tokens`，也就占用 `max_tokens` 额度。
       规划任务给 3 天行程预留 8400 token，推理一多正文就被切断，
       历史失败样本里 92% 的"JSON 不完整"就是这么来的；
    2. 思考模式下 `temperature` 不生效。本项目靠
       `temperature = 0.2 + 0.08 * i` 制造候选差异，开着 thinking
       等于三个候选共享同一个（被忽略的）温度，多样性策略形同虚设。

    所以这里把开关做成显式参数：`thinking=False` 时不带思考直接产 JSON，
    `thinking=True` 时保留推理但要由调用方给足 max_tokens，
    `thinking=None`（默认）不干预，沿用 API 默认行为。
    """

    def __init__(self, *args: Any, thinking: Optional[bool] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.thinking = thinking
        # 线程局部：3 路并行候选各自记录，互不覆盖
        self._call_meta = local()

    @property
    def last_call(self) -> Optional[LLMCallMeta]:
        """当前线程最近一次调用的元信息；未调用过则为 None。"""
        return getattr(self._call_meta, "last", None)

    @property
    def thinking_effective(self) -> str:
        """当前思考模式的可读描述，用于启动日志。"""
        if self.thinking is None:
            return "默认(未指定，服务端开启)"
        return "开启" if self.thinking else "关闭"

    def invoke(self, messages: List[dict], **kwargs: Any) -> str:
        extra_body = dict(kwargs.pop("extra_body", None) or {})
        if self.thinking is not None:
            # OpenAI SDK 不认 thinking 顶层参数，必须走 extra_body
            extra_body.setdefault(
                "thinking", {"type": "enabled" if self.thinking else "disabled"}
            )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                extra_body=extra_body or None,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ("temperature", "max_tokens")
                },
            )
        except Exception as error:  # noqa: BLE001 - 保持与父类一致的异常语义
            self._call_meta.last = None
            raise HelloAgentsException(f"LLM调用失败: {error}") from error

        choice = response.choices[0]
        content = choice.message.content or ""
        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        self._call_meta.last = LLMCallMeta(
            # 少数兼容实现不给 finish_reason，落成 unknown 而不是空串，
            # 统计时能和"确实没有值"区分开
            finish_reason=getattr(choice, "finish_reason", None) or "unknown",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            content_chars=len(content),
            model=getattr(response, "model", None) or self.model or "",
            reasoning_tokens=getattr(details, "reasoning_tokens", 0) or 0,
        )
        return content
