"""评测运行器：跑固定用例集 → 落盘结果 → 离线打分出报告。

为什么要有它
------------
改造前所有结论都是"接口快了多少秒"，没有一条是"行程变好了多少"。
这个脚本把质量变成可比的数字：同一批 case、同一套口径，跑两次就能出
before/after。**打分是离线重算的**（`--score-only`），规则调整后不用再
花一次 LLM 额度。

用法
----
    cd backend
    # 跑基线（会真实调用 LLM，20 条约 3-5 分钟）
    ./venv/Scripts/python.exe scripts/run_eval.py --tag baseline

    # 只跑前 6 条（省钱）
    ./venv/Scripts/python.exe scripts/run_eval.py --tag smoke --limit 6

    # 规则改完，用已有结果重新打分（不调 LLM）
    ./venv/Scripts/python.exe scripts/run_eval.py --tag baseline --score-only

产物
----
    training/eval/runs/<tag>/plans.jsonl     每次生成的原始结果
    training/eval/runs/<tag>/agent.log       Agent 的完整 stdout（排障用）
    training/eval/runs/<tag>/summary.json    逐条指标 + 汇总
    training/eval/runs/<tag>/report.md       人看的报告
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.eval import build_request, load_cases  # noqa: E402
from app.eval.metrics import aggregate, evaluate_plan, render_markdown  # noqa: E402
from app.models.schemas import TripPlan, TripRequest  # noqa: E402

RUNS_DIR = BACKEND / "training" / "eval" / "runs"


def log(message: str) -> None:
    """进度打到 stderr：stdout 整段被重定向给 Agent 的日志了。"""
    print(message, file=sys.stderr, flush=True)


def _select(cases: List[Dict[str, Any]], only: Optional[str], limit: Optional[int]) -> List[Dict[str, Any]]:
    if only:
        wanted = {item.strip() for item in only.split(",") if item.strip()}
        cases = [case for case in cases if case.get("id") in wanted]
    if limit:
        cases = cases[:limit]
    return cases


def run_cases(cases: List[Dict[str, Any]], workers: int) -> List[Dict[str, Any]]:
    """逐条生成行程，返回可直接落盘的记录。"""
    from app.agents.trip_planner_agent import MultiAgentTripPlanner
    from app.services.llm_service import get_llm

    get_llm()  # 主线程先建好单例，避免多线程同时初始化

    def _one(case: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        record: Dict[str, Any] = {
            "case_id": case.get("id", ""),
            "request": case["request"],
            "case": case,
            "plan": None,
            "elapsed": None,
            "error": "",
        }
        try:
            planner = MultiAgentTripPlanner()
            plan = planner.plan_trip(build_request(case))
            record["plan"] = plan.model_dump()
        except Exception as error:  # noqa: BLE001 - 单条失败不能拖垮整批
            record["error"] = f"{type(error).__name__}: {error}"
            log(f"  ✗ {record['case_id']}: {record['error']}")
            log(traceback.format_exc(limit=3))
        record["elapsed"] = round(time.perf_counter() - started, 2)
        return record

    records: List[Dict[str, Any]] = []
    if workers <= 1:
        for index, case in enumerate(cases, 1):
            log(f"[{index}/{len(cases)}] {case.get('id')} ...")
            records.append(_one(case))
        return records

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_one, case): case for case in cases}
        for index, future in enumerate(as_completed(futures), 1):
            case = futures[future]
            log(f"[{index}/{len(cases)}] {case.get('id')} 完成")
            records.append(future.result())
    # 并发完成顺序随机，按用例顺序落盘，diff 才稳定
    order = {case.get("id"): position for position, case in enumerate(cases)}
    records.sort(key=lambda item: order.get(item["case_id"], 0))
    return records


def score_run(run_dir: Path, tag: str) -> Dict[str, Any]:
    """读取 plans.jsonl，离线计算指标并写出报告。"""
    plans_path = run_dir / "plans.jsonl"
    rows: List[Dict[str, Any]] = []
    with plans_path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for row in rows:
        if not row.get("plan"):
            errors.append({"case_id": row.get("case_id", ""), "error": row.get("error", "")})
            continue
        request = TripRequest(**row["request"])
        plan = TripPlan(**row["plan"])
        result = evaluate_plan(request, plan, row.get("case"))
        result["elapsed"] = row.get("elapsed")
        results.append(result)

    summary = aggregate(results)
    summary["errors"] = len(errors)
    summary["error_cases"] = errors
    summary["tag"] = tag
    summary["generated_at"] = datetime.now().isoformat(timespec="seconds")
    summary["avg_latency"] = (
        round(sum(item["elapsed"] for item in results) / len(results), 2) if results else None
    )

    (run_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(
        render_markdown(summary, results, title=f"评测报告 · {tag}"),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="行程 Agent 评测")
    parser.add_argument("--tag", required=True, help="本次运行的标签，同时也是目录名，例如 baseline")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--only", default=None, help="只跑指定 case id，逗号分隔")
    parser.add_argument("--workers", type=int, default=2, help="并发生成数，默认 2")
    parser.add_argument("--score-only", action="store_true", help="只用已有 plans.jsonl 重新打分")
    parser.add_argument("--cases", default=None, help="用例文件路径")
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.tag
    if args.score_only:
        if not (run_dir / "plans.jsonl").is_file():
            log(f"✗ 没有找到 {run_dir / 'plans.jsonl'}")
            return 2
        summary = score_run(run_dir, args.tag)
        print((run_dir / "report.md").read_text(encoding="utf-8"))
        log(f"汇总: {json.dumps(summary, ensure_ascii=False)}")
        return 0

    cases = _select(load_cases(Path(args.cases) if args.cases else None), args.only, args.limit)
    if not cases:
        log("✗ 没有选中任何用例")
        return 2

    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"开始评测 tag={args.tag} 用例={len(cases)} 并发={args.workers}")
    log_dir = run_dir
    started = time.perf_counter()

    # Agent 会打印大量过程日志；统一收进 agent.log，终端只留进度。
    with (log_dir / "agent.log").open("w", encoding="utf-8") as agent_log, redirect_stdout(agent_log):
        records = run_cases(cases, args.workers)

    with (log_dir / "plans.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = round(time.perf_counter() - started, 1)
    summary = score_run(run_dir, args.tag)
    print((run_dir / "report.md").read_text(encoding="utf-8"))
    log(f"耗时 {total}s，报告: {run_dir / 'report.md'}")
    log(f"汇总: {json.dumps(summary, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
