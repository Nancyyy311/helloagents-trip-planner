"""离线评测集与指标。

`training/eval/cases.jsonl` 是**固定**的用例集：同一批请求反复跑，
才能比较"改前 vs 改后"。改用例等于换考卷，成绩不可比。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models.schemas import TripRequest

EVAL_DIR = Path(__file__).resolve().parents[2] / "training" / "eval"
CASES_PATH = EVAL_DIR / "cases.jsonl"


def load_cases(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """读取评测用例；每行一个 JSON，坏行直接报错（静默跳过会让用例数悄悄变少）。"""
    target = Path(path) if path else CASES_PATH
    cases: List[Dict[str, Any]] = []
    with target.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{target}:{line_number} 不是合法 JSON: {error}") from error
    return cases


def build_request(case: Dict[str, Any]) -> TripRequest:
    return TripRequest(**case["request"])
