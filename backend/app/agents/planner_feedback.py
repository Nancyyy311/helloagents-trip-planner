"""Planner 在线失败样本和偏好候选落盘逻辑。"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from ..models.schemas import TripPlan, TripRequest
from ..services.mcp_env import PROJECT_ROOT


ONLINE_FEEDBACK_DIR = PROJECT_ROOT / "training" / "data" / "online_feedback"
PLANNER_FAILURE_LOG = ONLINE_FEEDBACK_DIR / "planner_failures.jsonl"
PLANNER_DPO_CANDIDATE_LOG = ONLINE_FEEDBACK_DIR / "planner_dpo_candidates.jsonl"


class PlannerGenerationError(ValueError):
    """Raised when a planner cannot produce valid TripPlan JSON."""

    def __init__(self, message: str, failures: List[Dict[str, Any]]):
        super().__init__(message)
        self.failures = failures


def is_non_retryable_llm_error(error: Exception) -> bool:
    """判断同一 prompt 继续重试也很难恢复的 LLM 接口错误。"""
    text = str(error).lower()
    return (
        "content exists risk" in text
        or ("invalid_request_error" in text and "400" in text)
    )


def build_failure_row(
    label: str,
    attempt: int,
    max_attempts: int,
    request: TripRequest,
    planner_query: str,
    response: str,
    error: Exception,
    preference_reason: str = "planner_schema_invalid",
) -> Dict[str, Any]:
    """Build an online feedback row for invalid planner output."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "request": request.model_dump(),
        "prompt": planner_query,
        "rejected": response,
        "error_type": type(error).__name__,
        "error": str(error),
        "preference_reason": preference_reason,
    }


def log_preference_candidates(
    planner_query: str,
    request: TripRequest,
    chosen_plan: TripPlan,
    failures: List[Dict[str, Any]],
    chosen_label: str = "默认 Planner",
) -> None:
    """Log DPO candidate rows using invalid planner outputs as rejected."""
    chosen_value = json.dumps(chosen_plan.model_dump(), ensure_ascii=False)
    for failure in failures:
        row = {
            "conversations": [{"from": "human", "value": planner_query}],
            "chosen": {"from": "gpt", "value": chosen_value},
            "rejected": {"from": "gpt", "value": failure["rejected"]},
            "preference_reason": failure.get("preference_reason", "planner_schema_invalid"),
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request": request.model_dump(),
                "rejected_label": failure.get("label"),
                "rejected_attempt": failure.get("attempt"),
                "chosen_label": chosen_label,
                "error": failure.get("error"),
            },
        }
        append_jsonl(PLANNER_DPO_CANDIDATE_LOG, row)


def append_jsonl(path, row: Dict[str, Any]) -> None:
    """Append one JSONL row, creating the directory on demand.

    改造点：加体积上限 + 轮转。
    失败样本每行带着完整 prompt（含候选 POI 池）和模型原始输出，实测单行
    48~56KB。原实现只 append，永不清理：一天几百次失败就是几十 MB，
    长时间跑会把磁盘写满，而且写满之后再失败就没人知道了。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _rotate_if_oversized(path)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


# 单个数据文件的上限，超过就轮转成 .1（只保留一代）
JSONL_MAX_BYTES = int(os.getenv("ONLINE_FEEDBACK_MAX_BYTES", str(32 * 1024 * 1024)))


def _rotate_if_oversized(path: Path) -> None:
    """超过上限就把当前文件改名成 .1，旧的 .1 直接覆盖。

    只保留一代是有意的：这些是在线收集的训练样本，不是审计日志，
    丢失更旧的一批可以接受，磁盘被写满不可接受。
    任何 IO 异常都不能向上冒泡——落盘失败不该拖垮一次正常的规划请求。
    """
    try:
        if path.stat().st_size < JSONL_MAX_BYTES:
            return
        backup = path.with_name(path.name + ".1")
        if backup.exists():
            backup.unlink()
        path.rename(backup)
    except OSError:
        return
