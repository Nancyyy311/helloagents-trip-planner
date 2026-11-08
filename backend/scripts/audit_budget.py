"""预算审计：把"代码纠正了多少"变成可复现的数字。

为什么需要它
------------
`budget_report_consistent_rate` 在归一化上线后**恒为 100%**——因为
`budget.total` 已经被重算值覆盖，再拿它和重算值比较必然相等。
这是同义反复，只能证明"归一化确实挂上了"，不能证明"模型变准了"，
更不能当简历/台账里的效果证据。

要说明效果，必须拿到"模型自己说了多少"。两个来源：
  - 新数据：`budget_reported_raw`（归一化前留存）
  - 旧数据：`budget.total` 本身就是模型原始值（当时尚未接入归一化）

本脚本统一处理，输出三个可复现的量：
  1. 账实不符份数与累计金额差
  2. 少报 / 多报的方向分布
  3. **隐藏超支**：模型上报"没超预算"、按单价算实际超了
     —— 这是用户到店才发现钱不够的那类错，比"偏差金额"更该报

用法
----
    cd backend
    ./venv/Scripts/python.exe scripts/audit_budget.py --tag baseline
    ./venv/Scripts/python.exe scripts/audit_budget.py --tag baseline --verbose
    ./venv/Scripts/python.exe scripts/audit_budget.py --tag baseline --markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.eval.metrics import BUDGET_REPORT_TOLERANCE  # noqa: E402
from app.models.schemas import TripPlan, TripRequest  # noqa: E402
from app.planner.rerank import recompute_budget_from_selected_items  # noqa: E402

RUNS_DIR = BACKEND / "training" / "eval" / "runs"


def audit_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """对单份计划算账。返回 None 由调用方过滤（本函数不抛异常）。"""
    plan_data = row.get("plan") or {}
    if not plan_data:
        return {"skipped": True}

    trip_plan = TripPlan(**plan_data)
    request = TripRequest(**row["request"])
    recomputed = recompute_budget_from_selected_items(trip_plan, request)

    real_total = int(recomputed.get("total") or 0)
    if real_total <= 0:
        # 全 0 价格（兜底计划）没有信息量，计入了只会污染分母。
        return {"skipped": True}

    shown_total = int((trip_plan.budget.total if trip_plan.budget else 0) or 0)
    raw = trip_plan.budget_reported_raw
    raw_available = raw is not None
    # 旧数据没有原始值，但那时 budget.total 就是模型原始值，可以直接用。
    raw_total = int(raw) if raw_available else shown_total

    tolerance = max(1, int(real_total * BUDGET_REPORT_TOLERANCE))
    diff = raw_total - real_total
    inconsistent = abs(diff) > tolerance
    target = int(request.budget_constraint.amount or 0)
    hidden_overrun = bool(target > 0 and raw_total <= target < real_total)

    return {
        "skipped": False,
        "case_id": row.get("case_id", ""),
        "city": request.city,
        "raw_total": raw_total,
        "real_total": real_total,
        "shown_total": shown_total,
        "diff": diff,
        "inconsistent": inconsistent,
        "raw_available": raw_available,
        "target": target,
        "hidden_overrun": hidden_overrun,
        "hidden_amount": (real_total - target) if hidden_overrun else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="预算审计：模型上报值 vs 按单价重算值")
    parser.add_argument("--tag", required=True, help="runs 目录下的运行标签，例如 baseline")
    parser.add_argument("--runs-dir", default=str(RUNS_DIR), help="runs 根目录")
    parser.add_argument("--verbose", action="store_true", help="逐条打印明细")
    parser.add_argument("--markdown", action="store_true", help="输出 markdown 表格")
    args = parser.parse_args()

    plans_path = Path(args.runs_dir) / args.tag / "plans.jsonl"
    if not plans_path.exists():
        print(f"找不到 {plans_path}", file=sys.stderr)
        return 2

    rows = [json.loads(line) for line in plans_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    audited: List[Dict[str, Any]] = []
    errors = 0
    for row in rows:
        if row.get("error"):
            errors += 1
            continue
        try:
            item = audit_row(row)
        except Exception as exc:  # 单条坏数据不该让整轮审计失败
            print(f"跳过 {row.get('case_id')}: {exc}", file=sys.stderr)
            errors += 1
            continue
        if item.get("skipped"):
            continue
        audited.append(item)

    if not audited:
        print("没有可审计的样本", file=sys.stderr)
        return 1

    total = len(audited)
    bad = [i for i in audited if i["inconsistent"]]
    under = [i for i in bad if i["diff"] < 0]
    over = [i for i in bad if i["diff"] > 0]
    amount = sum(abs(i["diff"]) for i in bad)
    hidden = [i for i in audited if i["hidden_overrun"]]
    raw_available = sum(1 for i in audited if i["raw_available"])

    print(f"运行：{args.tag}    可审计样本：{total}（跳过 {errors} 条）")
    if raw_available == 0:
        print(
            "⚠️  本批数据没有 budget_reported_raw，用 budget.total 代替原始值。\n"
            "    这只在「该运行尚未接入归一化」时成立（如改造前的 baseline）。\n"
            "    若该运行已接入归一化，total 本身就是重算值，下面的偏差会假性归零。",
            file=sys.stderr,
        )
    print(f"原始上报值可追溯：{raw_available}/{total}")
    print("")
    print(f"账实不符：{len(bad)}/{total} = {len(bad) / total:.1%}（容差 {BUDGET_REPORT_TOLERANCE:.0%}）")
    print(f"  少报（上报 < 真实）：{len(under)} 份")
    print(f"  多报（上报 > 真实）：{len(over)} 份")
    print(f"  累计金额差：{amount} 元")
    if under:
        worst = min(under, key=lambda i: i["diff"])
        print(f"  最大少报：{abs(worst['diff'])} 元（{worst['case_id']}  上报 {worst['raw_total']} / 真实 {worst['real_total']}）")
    if over:
        worst = max(over, key=lambda i: i["diff"])
        print(f"  最大多报：{worst['diff']} 元（{worst['case_id']}  上报 {worst['raw_total']} / 真实 {worst['real_total']}）")
    print("")
    print(f"隐藏超支（上报显示未超、实际超）：{len(hidden)} 份")
    for item in hidden:
        print(
            f"  {item['case_id']:<18} 上报 {item['raw_total']:>6}  真实 {item['real_total']:>6}  "
            f"预算 {item['target']:>6}  隐性超支 +{item['hidden_amount']}"
        )

    if args.verbose:
        print("")
        print("逐条（差值为正=多报，为负=少报；* 表示超出容差）")
        for item in sorted(audited, key=lambda i: i["diff"]):
            flag = "*" if item["inconsistent"] else " "
            print(
                f"  {flag} {item['case_id']:<18} 上报 {item['raw_total']:>6}  "
                f"真实 {item['real_total']:>6}  差 {item['diff']:>+7}"
            )

    if args.markdown:
        print("")
        print("| 指标 | 数值 |")
        print("|---|---|")
        print(f"| 可审计样本 | {total} |")
        print(f"| 账实不符 | {len(bad)}/{total}（{len(bad) / total:.1%}） |")
        print(f"| 少报 / 多报 | {len(under)} / {len(over)} |")
        print(f"| 累计金额差 | {amount} 元 |")
        print(f"| 隐藏超支 | {len(hidden)} 份，最大 {max([i['hidden_amount'] for i in hidden] or [0])} 元 |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
