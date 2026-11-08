"""失败样本归因报告：把 planner_failures.jsonl 按病因拆开统计。

为什么单独写一个脚本
--------------------
`/metrics` 只统计**进程启动之后**的失败，历史样本在 jsonl 里躺着。
做优化决策前得先看清存量：到底是"模型不会输出 JSON"，还是"输出被截断"。
两者的修法完全不同，混在一起就是在错误的方向上加 prompt。

用法：
    cd backend
    ./venv/Scripts/python.exe scripts/analyze_failures.py            # 默认读线上日志
    ./venv/Scripts/python.exe scripts/analyze_failures.py path.jsonl # 指定文件

只读本地文件，不调 LLM，不消耗任何额度。
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# 允许从 backend/ 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.planner_feedback import PLANNER_FAILURE_LOG  # noqa: E402
from app.planner.failure_kind import (  # noqa: E402
    classify_failure,
    describe,
    is_recoverable,
    is_truncation,
)
from app.models.schemas import TripPlan, TripRequest  # noqa: E402
from app.planner.output import (  # noqa: E402
    extract_json_object,
    validate_trip_plan_shape,
)


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else PLANNER_FAILURE_LOG
    if not path.exists():
        print(f"找不到日志文件: {path}")
        return 1

    rows = load_rows(path)
    if not rows:
        print("日志为空，没有可归因的样本。")
        return 0

    # 老样本没有 failure_kind 字段，用当前规则回溯判定，
    # 这样历史数据和改造后的新数据能用同一套口径对比。
    for row in rows:
        if not row.get("failure_kind"):
            row["failure_kind"] = classify_failure(
                row.get("rejected") or "",
                row.get("finish_reason"),
            )

    kinds = Counter(row["failure_kind"] for row in rows)
    total = len(rows)
    truncated = sum(count for kind, count in kinds.items() if is_truncation(kind))

    print(f"文件     : {path}")
    print(f"样本总数 : {total}")
    print()
    print(f"{'病因':24s} {'数量':>5s} {'占比':>7s}  说明")
    print("-" * 78)
    for kind, count in kinds.most_common():
        share = count / total * 100
        print(f"{kind:24s} {count:5d} {share:6.1f}%  {describe(kind)}")
    print("-" * 78)
    print(f"其中「输出没写完」: {truncated}/{total} ({truncated / total * 100:.1f}%)")
    recoverable = sum(count for kind, count in kinds.items() if is_recoverable(kind))
    print(f"其中「原文其实够用」: {recoverable}/{total}")
    print()

    # 按原文是否为空再切一刀：老样本大量 empty 是上一轮修的回归（并行路径丢原文），
    # 修完之后 empty 应该迅速归零，这个数字本身就是回归是否修好的验收口径。
    empty_raw = sum(1 for row in rows if not (row.get("rejected") or "").strip())
    print(f"原文为空的样本: {empty_raw}/{total}"
          f"（改造前并行路径写死空串导致，修复后应趋近 0）")
    print()

    # 回放：用**当前**解析器重跑历史失败原文，看现在还能不能救回。
    # 这是"改造有没有用"最直接的证据——同一个模型输出，老解析器判死，
    # 新解析器能解出合法 TripPlan，说明问题在解析而不在模型。
    has_raw = [row for row in rows if (row.get("rejected") or "").strip()]
    rescued = 0
    rescued_kinds = Counter()
    for row in has_raw:
        raw = row["rejected"]
        # 严格口径：不仅要求解出对象，还要过 TripPlan schema + 天数/城市一致性校验，
        # 否则"救回"就是自欺欺人。
        try:
            payload = extract_json_object(raw)
            if not payload:
                continue
            plan = TripPlan.model_validate(payload)
            request = TripRequest.model_validate(row.get("request") or {})
            validate_trip_plan_shape(plan, request)
        except Exception:
            continue
        rescued += 1
        rescued_kinds[row["failure_kind"]] += 1
    if has_raw:
        print(f"回放（用当前解析器重解历史原文）: 救回 {rescued}/{len(has_raw)} 条"
              f"（{rescued / len(has_raw) * 100:.1f}%）")
        if rescued_kinds:
            print(f"  救回的样本按病因: {dict(rescued_kinds)}")
        print()

    samples = defaultdict(list)
    for row in rows:
        raw = (row.get("rejected") or "").strip()
        kind = row["failure_kind"]
        if raw and len(samples[kind]) < 3:
            samples[kind].append(raw[-40:].replace("\n", "\\n"))

    if samples:
        print("每类结尾 40 字符样例（看是否停在半句/半个数字上）：")
        for kind, tails in samples.items():
            print(f"  [{kind}]")
            for tail in tails:
                print(f"    …{tail!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
