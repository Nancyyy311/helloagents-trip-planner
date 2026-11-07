#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D3 缺陷复现脚本（仅用 Python 标准库，无需 pip install）。

用法:
  # 1) 只跑"免费"检查：健康检查 503、/metrics 404、响应无 traceId 探查
  #    —— 不消耗任何 DeepSeek / 高德额度，建议先跑这个
  venv/Scripts/python.exe bench.py

  # 2) 额外跑"需要真实生成"的检查：无缓存、并发阻塞、traceId 实证
  #    —— 会真实调用 DeepSeek + 高德，消耗额度（约 5 次生成）
  venv/Scripts/python.exe bench.py --gen

  # 端口不是 7000 时:
  venv/Scripts/python.exe bench.py --base http://localhost:8080

说明:
  - 服务必须已在 http://localhost:7000 跑起来（见 D1 手册 python run.py）。
  - 本脚本只读不写，不改任何源码和 .env。
  - 缺陷2(假成功)需要手动"改坏模型名"复现，指南里有步骤，这里不自动做。
"""

import json
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

BASE = "http://localhost:7000"

# 一个最小可生成的请求体（北京 3 天，预算 3000）。
# 字段严格对齐 models/schemas.py 的 TripRequest：
#   transportation / accommodation 是必填；
#   party.total 必须等于 adults+children+elders；
#   budget 字段名是 budget_level / strictness，不是 level。
SAMPLE_REQUEST = {
    "city": "北京",
    "start_date": "2026-11-10",
    "end_date": "2026-11-12",
    "travel_days": 3,
    "transportation": "公共交通",
    "accommodation": "经济型酒店",
    "preferences": ["文化历史", "美食"],
    "free_text_input": "",
    "party": {"adults": 1, "children": 0, "elders": 0, "total": 1, "companion_type": "solo"},
    "budget_constraint": {
        "amount": 3000,
        "scope": "total",
        "currency": "CNY",
        "budget_level": "standard",
        "strictness": "none",
    },
}


def _request(method, path, payload=None, timeout=300):
    url = BASE + path
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
            return r.status, time.perf_counter() - t0, body
    except urllib.error.HTTPError as e:
        return e.code, time.perf_counter() - t0, e.read().decode("utf-8", "replace")
    except Exception as e:  # 连接拒绝等
        return -1, time.perf_counter() - t0, str(e)


def report(name, reproduced, detail):
    tag = "✅ 复现成功" if reproduced else "⚠️  未复现 / 正常"
    print(f"[{tag}] {name}")
    print(f"          {detail}\n")


def check_health_503():
    status, _, body = _request("GET", "/api/trip/health")
    report(
        "缺陷1  健康检查恒 503",
        status == 503,
        f"GET /api/trip/health 返回 {status}（预期 503）。"
        f"根因 trip.py:82 写了 agent.agent.name，但该对象没有 .agent 属性 → AttributeError → 被捕获返回 503。",
    )


def check_app_health_ok():
    status, _, _ = _request("GET", "/health")
    report(
        "对照  应用级 /health 正常",
        status == 200,
        f"GET /health 返回 {status}（预期 200）。说明只有 trip 子服务有 bug，应用本身没挂。",
    )


def check_metrics_404():
    status, _, _ = _request("GET", "/metrics")
    report(
        "缺陷5a  无 Prometheus 指标端点",
        status == 404,
        f"GET /metrics 返回 {status}（预期 404）。说明项目零可观测，没有 metrics 暴露。",
    )


def check_no_traceid():
    status, elapsed, body = _request("POST", "/api/trip/plan", SAMPLE_REQUEST)
    low = body.lower()
    has_trace = ("traceid" in low) or ("trace_id" in low) or ("requestid" in low)
    detail = f"POST /api/trip/plan 状态={status}, 耗时={elapsed:.1f}s, 响应含 traceId={has_trace}。"
    if status == 200:
        try:
            obj = json.loads(body)
            detail += f" success={obj.get('success')}, message={obj.get('message')}"
        except Exception:
            pass
    report(
        "缺陷5b  响应无 traceId / requestId",
        (not has_trace) and status in (200, -1),
        detail + "（预期无 traceId，说明一次请求无法被链路追踪串联）",
    )


def check_duplicate_no_cache():
    print(">>> 缺陷4 复现：连发 2 次相同请求，观察是否每次都实时调高德（无缓存）")
    t1 = _request("POST", "/api/trip/plan", SAMPLE_REQUEST)
    t2 = _request("POST", "/api/trip/plan", SAMPLE_REQUEST)
    report(
        "缺陷4  无缓存（计时证据）",
        True,
        f"第1次 {t1[1]:.1f}s, 第2次 {t2[1]:.1f}s。"
        f"两次耗时接近且都成功，说明第2次没命中缓存（有缓存应明显更快）。"
        f"请到【服务终端】确认出现了 2 次『步骤1: 并行获取...』的 print。",
    )


def check_concurrency_blocking():
    print(">>> 缺陷3 复现：并发发 2 个请求，观察事件循环是否被阻塞（无法真正并行）")
    n = 2
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as ex:
        futures = [ex.submit(_request, "POST", "/api/trip/plan", SAMPLE_REQUEST) for _ in range(n)]
        results = [f.result() for f in futures]
    wall = time.perf_counter() - t0
    singles = [r[1] for r in results if r[1] > 0]
    single = max(singles) if singles else 0
    ratio = (wall / single) if single > 0 else 0
    statuses = [r[0] for r in results]
    report(
        "缺陷3  并发串台 / 事件循环阻塞（计时证据）",
        ratio >= 1.5,
        f"{n} 个并发请求总墙钟 {wall:.1f}s, 单次最长 {single:.1f}s, 比值={ratio:.2f}, 各状态={statuses}。"
        f"理想并行应≈1；实际≈{n} 说明 async 里直接调同步阻塞的 agent.plan_trip()（trip.py:43）"
        f"把事件循环堵死，并发无法加速。也可在请求期间另开终端 GET /health 看是否变慢。",
    )


def main():
    global BASE
    args = sys.argv[1:]
    if "--base" in args:
        i = args.index("--base")
        BASE = args[i + 1]
    run_gen = "--gen" in args

    print(f"=== D3 缺陷复现（目标服务 {BASE}）===\n")
    check_health_503()
    check_app_health_ok()
    check_metrics_404()
    if run_gen:
        print("(以下检查会真实调用 DeepSeek + 高德，消耗额度)\n")
        check_no_traceid()
        check_duplicate_no_cache()
        check_concurrency_blocking()
    else:
        print("已跳过『需真实生成』的检查（缺陷2/3/4 的实证）。")
        print("要看完整复现请运行: venv/Scripts/python.exe bench.py --gen\n")
    print("=== 完成。缺陷2(假成功) 请按指南手动用『改坏模型名』复现 ===")


if __name__ == "__main__":
    main()
