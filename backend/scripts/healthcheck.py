#!/usr/bin/env python
"""一条命令核对"项目是否还站得住"。

为什么要有这个脚本：口头说"测过了、全绿"是不可核验的断言。前面几批改造里，
我至少犯过一次"以为验过、其实没验"（说 216 个测试全离线，删掉 .env 就红 2 条）。
所以把验收动作固化成代码，任何人 clone 下来都能自己跑一遍，不用信我的结论。

用法：
    cd backend
    ./venv/Scripts/python.exe scripts/healthcheck.py           # 快速（跳过前端构建）
    ./venv/Scripts/python.exe scripts/healthcheck.py --full    # 含 vue-tsc + vite build
    ./venv/Scripts/python.exe scripts/healthcheck.py --quiet   # 只打印结论表

退出码：0 全过；1 有 FAIL。
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
FRONTEND = ROOT / "frontend"

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"

# npm/vitest 带 ANSI 颜色，不剥掉会匹配不到 "Tests  28 passed"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

results: list[tuple[str, str, str]] = []  # (状态, 名称, 说明)


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))


def run(cmd: list[str], cwd: Path, timeout: int = 900) -> tuple[int, str]:
    """执行子进程，返回 (退出码, 合并输出)。"""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            env=env,
        )
    except FileNotFoundError:
        return 127, f"命令不存在: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, f"超时（>{timeout}s）: {' '.join(cmd)}"
    out = proc.stdout.decode("utf-8", errors="replace")
    return proc.returncode, ANSI_RE.sub("", out)


def tail(text: str, n: int = 3) -> str:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return " | ".join(lines[-n:])[:220]


def _pytest_summary(text: str) -> str:
    """从 pytest 输出里抠出 'N passed in X' 那一行，避开业务日志干扰。"""
    m = re.search(r"^=*\s*(\d+ passed[^=\n]*?)\s*=*$", text, re.M)
    if m:
        return m.group(1).strip()
    m = re.search(r"(\d+ (?:passed|failed)[^\n]*)", text)
    return m.group(1).strip()[:120] if m else tail(text, 2)


# ---------------------------------------------------------------- 检查项


def check_git_clean() -> None:
    code, out = run(["git", "status", "--porcelain"], ROOT)
    if code != 0:
        record(FAIL, "git 工作区", "git 命令失败")
        return
    dirty = [ln for ln in out.splitlines() if ln.strip()]
    if dirty:
        record(WARN, "git 工作区", f"有 {len(dirty)} 个未提交改动")
    else:
        record(PASS, "git 工作区", "干净")

    code, out = run(["git", "rev-list", "--count", "HEAD"], ROOT)
    if code == 0:
        record(PASS, "commit 数", out.strip())


def check_backend_tests(with_env: bool) -> None:
    env_path = BACKEND / ".env"
    backup = BACKEND / ".env.healthcheck-bak"
    label = "后端测试（有 .env）" if with_env else "后端测试（无 .env）"
    moved = False
    try:
        if not with_env:
            if not env_path.exists():
                record(SKIP, label, "本机本来就没有 .env，无需反向验证")
                return
            env_path.rename(backup)
            moved = True
        code, out = run([sys.executable, "-m", "pytest", "-q"], BACKEND)
        if code == 0:
            record(PASS, label, _pytest_summary(out))
        else:
            failed = re.findall(r"^FAILED (\S+)", out, re.M)
            detail = (
                f"{len(failed)} 条失败: {', '.join(failed[:3])}"
                if failed
                else _pytest_summary(out)
            )
            record(FAIL, label, detail)
    finally:
        if moved and backup.exists():
            backup.rename(env_path)


def check_frontend_tests() -> None:
    npm = shutil.which("npm")
    if not npm:
        record(SKIP, "前端测试", "未找到 npm")
        return
    code, out = run([npm, "test", "--silent"], FRONTEND, timeout=600)
    match = re.search(r"Tests\s+(\d+ passed[^\n]*)", out)
    detail = match.group(1) if match else tail(out, 2)
    record(PASS if code == 0 else FAIL, "前端测试", detail)


def check_frontend_build() -> None:
    npm = shutil.which("npm")
    if not npm:
        record(SKIP, "前端构建（vue-tsc + vite）", "未找到 npm")
        return
    code, out = run([npm, "run", "build"], FRONTEND, timeout=900)
    built = re.search(r"built in ([\d.]+s)", out)
    detail = f"构建成功（{built.group(1)}）" if built else tail(out, 2)
    record(PASS if code == 0 else FAIL, "前端构建（vue-tsc + vite）", detail)


def check_compile() -> None:
    code, out = run(
        [sys.executable, "-m", "compileall", "-q", "app", "scripts", "tests"], BACKEND
    )
    record(PASS if code == 0 else FAIL, "Python 编译", "无语法错误" if code == 0 else tail(out))


def check_secrets_history() -> None:
    """git 历史里有没有提交过 .env 或真 key。推之前必查。"""
    code, out = run(
        ["git", "log", "--all", "--diff-filter=A", "--name-only", "--pretty=format:"], ROOT
    )
    if code != 0:
        record(SKIP, "git 历史密钥", "git 命令失败")
        return
    leaked_env = {
        ln.strip()
        for ln in out.splitlines()
        if ln.strip() and Path(ln.strip()).name == ".env"
    }
    if leaked_env:
        record(FAIL, "git 历史密钥", f".env 曾被提交: {', '.join(sorted(leaked_env))}")
        return

    code, out = run(["git", "log", "--all", "-p"], ROOT, timeout=600)
    pattern = re.compile(
        r"(sk-[A-Za-z0-9]{20,}|AMAP_API_KEY\s*=\s*[\"']?[a-f0-9]{20,}"
        r"|LLM_API_KEY\s*=\s*[\"']?[A-Za-z0-9]{20,})"
    )
    hits = pattern.findall(out) if code == 0 else []
    if hits:
        record(FAIL, "git 历史密钥", f"疑似真 key {len(hits)} 处")
    else:
        record(PASS, "git 历史密钥", "从未提交 .env，无 key 形态")


def check_empty_tests() -> None:
    """没有任何断言的测试是"不抛即通过"，退化了也发现不了。"""
    offenders: list[str] = []
    total = 0
    for path in sorted((BACKEND / "tests").glob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                total += 1
                if _has_assertion(node):
                    continue
                offenders.append(f"{path.name}::{node.name}")
    if offenders:
        record(FAIL, "空断言测试", f"{len(offenders)}/{total}: {', '.join(offenders[:2])}")
    else:
        record(PASS, "空断言测试", f"{total} 个测试函数均有断言")


def _has_assertion(node: ast.FunctionDef) -> bool:
    if any(isinstance(n, ast.Assert) for n in ast.walk(node)):
        return True
    soft = {"raises", "fail", "assert_", "assertEqual", "assertTrue", "assertFalse", "check"}
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            fn = n.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name in soft:
                return True
    return False


def check_unused_imports() -> None:
    offenders: list[str] = []
    for path in sorted((BACKEND / "app").rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        src = path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        imported: dict[str, int] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported[(alias.asname or alias.name).split(".")[0]] = node.lineno
            elif isinstance(node, ast.ImportFrom):
                if node.module == "__future__":
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    imported[alias.asname or alias.name] = node.lineno
        # 去掉 import 行后按词边界找引用，比只看 ast.Name 更稳（能覆盖注解/字符串注解）
        body = "\n".join(
            ln
            for i, ln in enumerate(src.splitlines(), 1)
            if not re.match(r"\s*(from\s+\S+\s+)?import\s", ln) and i not in imported.values()
        )
        for name, line in imported.items():
            if not re.search(rf"\b{re.escape(name)}\b", body):
                offenders.append(f"{path.relative_to(BACKEND)}:{line} {name}")
    if offenders:
        record(WARN, "未使用导入", f"{len(offenders)} 处: {', '.join(offenders[:3])}")
    else:
        record(PASS, "未使用导入", "app/ 下无残留")


def check_todo_marks() -> None:
    pattern = re.compile(r"\b(TODO|FIXME|XXX)\b")
    hits: list[str] = []
    for path in sorted((BACKEND / "app").rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if pattern.search(line) and line.strip().startswith("#"):
                hits.append(f"{path.name}:{i}")
    record(
        PASS if not hits else WARN,
        "TODO/FIXME 残留",
        "无" if not hits else f"{len(hits)} 处: {', '.join(hits[:3])}",
    )


def check_budget_audit() -> None:
    """预算结论能否一条命令复现——这是简历上最主要的数字。"""
    runs = BACKEND / "training" / "eval" / "runs"
    if not (runs / "baseline" / "plans.jsonl").exists():
        record(SKIP, "预算审计可复现", "缺少 baseline/plans.jsonl")
        return
    code, out = run([sys.executable, "scripts/audit_budget.py", "--tag", "baseline"], BACKEND)
    if code != 0:
        record(FAIL, "预算审计可复现", tail(out, 2))
        return
    m = re.search(r"(\d+/\d+).*?账实不符|账实不符.*?(\d+/\d+)", out)
    detail = m.group(0) if m else tail(out, 2)
    record(PASS, "预算审计可复现", detail)


# ---------------------------------------------------------------- 主流程


def main() -> int:
    parser = argparse.ArgumentParser(description="项目健康检查")
    parser.add_argument("--full", action="store_true", help="含前端构建（较慢）")
    parser.add_argument("--quiet", action="store_true", help="只打印结论表")
    args = parser.parse_args()

    started = time.time()
    checks = [
        ("git", check_git_clean),
        ("backend-env", lambda: check_backend_tests(with_env=True)),
        ("backend-noenv", lambda: check_backend_tests(with_env=False)),
        ("frontend", check_frontend_tests),
        ("compile", check_compile),
        ("secrets", check_secrets_history),
        ("empty-tests", check_empty_tests),
        ("unused-imports", check_unused_imports),
        ("todo", check_todo_marks),
        ("budget", check_budget_audit),
    ]
    if args.full:
        checks.append(("build", check_frontend_build))

    for key, fn in checks:
        if not args.quiet:
            print(f"[..] {key}", flush=True)
        try:
            fn()
        except Exception as exc:  # 单项炸掉不能带走整份报告
            record(FAIL, key, f"检查本身异常: {type(exc).__name__}: {exc}")

    width = max((len(n) for _, n, _ in results), default=10)
    print()
    print(f"{'状态':<5} {'检查项':<{width}}  说明")
    print("-" * (width + 60))
    for status, name, detail in results:
        mark = {PASS: "OK  ", FAIL: "FAIL", WARN: "WARN", SKIP: "SKIP"}[status]
        print(f"{mark:<5} {name:<{width}}  {detail}")

    fails = sum(1 for s, _, _ in results if s == FAIL)
    warns = sum(1 for s, _, _ in results if s == WARN)
    elapsed = time.time() - started
    print("-" * (width + 60))
    print(f"合计 {len(results)} 项：{fails} FAIL / {warns} WARN / 耗时 {elapsed:.1f}s")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
