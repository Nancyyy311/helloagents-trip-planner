"""天气预报缓存新鲜度审计。

用途：不调 LLM、不打开页面，一条命令回答"为什么结果页后面几天没有天气"。

做法：直接读 training/data/cache/planner_context/amap_api/weather_weatherInfo/ 下的
真实缓存文件（高德原始响应），对每个城市算两件事：

1. 这份缓存是什么时候拉的、casts 覆盖哪几天、是否从"今天"开始；
2. 对一次"今天+2 出发、5 天"的行程，旧逻辑（只按 TTL 命中）能覆盖几天，
   新逻辑（跨天即重拉，窗口 = 今天起 4 天）能覆盖几天。

旧逻辑的窗口直接读缓存；新逻辑的窗口按高德固定返回"今天起 4 天"推算，
这个前提可以用缓存文件自证：每个文件的 casts[0] 都等于它 reporttime 的日期。

用法：
    python scripts/audit_weather_cache.py                # 只看本地缓存
    python scripts/audit_weather_cache.py --live 上海     # 额外打一次真实接口对账
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.planner.amap import (  # noqa: E402
    AMAP_WEATHER_CACHE_TTL_SECONDS,
    PLANNER_CONTEXT_CACHE_DIR,
    AmapPlannerClient,
    weather_forecast_dates,
    weather_window_starts_today,
)

WEATHER_CACHE_SUBDIR = Path("amap_api") / "weather_weatherInfo"
TRIP_START_OFFSET_DAYS = 2
TRIP_DAYS = 5


def _load_rows(cache_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((cache_dir / WEATHER_CACHE_SUBDIR).glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # 坏文件不该让审计整体挂掉
            print(f"⚠️  跳过无法解析的缓存 {path.name}: {exc}")
            continue
        data = payload.get("data") or {}
        dates = weather_forecast_dates(data)
        created_at = float(payload.get("created_at") or 0)
        rows.append(
            {
                "city": path.name.split("_")[0],
                "file": path.name,
                "created_at": created_at,
                "age_hours": (datetime.now().timestamp() - created_at) / 3600 if created_at else None,
                "ttl_hours": float(payload.get("ttl_seconds") or 0) / 3600,
                "dates": dates,
                "starts_today": weather_window_starts_today(data),
                "reporttime": ((data.get("forecasts") or [{}])[0] or {}).get("reporttime", ""),
            }
        )
    return rows


def _coverage(dates: list[str], start: date, days: int) -> int:
    window = {(start + timedelta(days=i)).isoformat() for i in range(days)}
    return len(window & set(dates))


def _print_table(rows: list[dict], start: date) -> None:
    print(f"行程假设：{start.isoformat()} 出发，共 {TRIP_DAYS} 天")
    print()
    header = f"{'城市':<6}{'缓存写入':<14}{'年龄h':>7}{'TTLh':>7}  {'覆盖区间':<26}{'从今天起':<8}{'旧覆盖':>7}{'新覆盖':>7}"
    print(header)
    print("-" * len(header))
    stale = 0
    for row in rows:
        dates = row["dates"]
        span = f"{dates[0]}~{dates[-1]}" if dates else "（空）"
        old_cov = _coverage(dates, start, TRIP_DAYS)
        # 新逻辑：判据不通过就重拉，重拉后窗口 = 今天起 4 天
        fresh_dates = [(date.today() + timedelta(days=i)).isoformat() for i in range(4)]
        new_cov = _coverage(fresh_dates, start, TRIP_DAYS) if not row["starts_today"] else old_cov
        if not row["starts_today"]:
            stale += 1
        created = datetime.fromtimestamp(row["created_at"]).strftime("%m-%d %H:%M") if row["created_at"] else "-"
        age = f"{row['age_hours']:.1f}" if row["age_hours"] is not None else "-"
        print(
            f"{row['city']:<6}{created:<14}{age:>7}{row['ttl_hours']:>7.0f}  {span:<26}"
            f"{'是' if row['starts_today'] else '否':<8}{old_cov:>7}{new_cov:>7}"
        )
    print()
    print(f"共 {len(rows)} 个城市的天气缓存，其中 {stale} 个的预报窗口不从今天开始（旧逻辑会直接复用）。")
    if rows:
        old_total = sum(_coverage(r["dates"], start, TRIP_DAYS) for r in rows)
        new_total = sum(
            _coverage([(date.today() + timedelta(days=i)).isoformat() for i in range(4)], start, TRIP_DAYS)
            if not r["starts_today"]
            else _coverage(r["dates"], start, TRIP_DAYS)
            for r in rows
        )
        print(f"5 天行程的天气覆盖：旧逻辑合计 {old_total} 天 → 新逻辑 {new_total} 天（{len(rows)} 个城市合计）")


def _live_check(city: str) -> None:
    from app.config import settings

    key = (getattr(settings, "amap_api_key", "") or "").strip()
    if not key:
        print("\n❌ 未配置 AMAP_API_KEY，跳过 --live 对账")
        return

    client = AmapPlannerClient(key)
    raw = client.get("/weather/weatherInfo", {"city": city, "extensions": "all"})
    dates = weather_forecast_dates(raw)
    start = date.today() + timedelta(days=TRIP_START_OFFSET_DAYS)
    print()
    print(f"实时对账：{city}")
    print(f"  高德真实窗口：{dates[0] if dates else '（空）'}~{dates[-1] if dates else '（空）'}，共 {len(dates)} 天")
    print(f"  是否从今天开始：{'是' if weather_window_starts_today(raw) else '否'}")
    print(f"  {TRIP_DAYS} 天行程（{start.isoformat()} 起）可覆盖：{_coverage(dates, start, TRIP_DAYS)} 天")
    print(f"  天气缓存 TTL：{AMAP_WEATHER_CACHE_TTL_SECONDS / 3600:.0f} 小时（通用 TTL 是它的问题来源）")


def main() -> int:
    parser = argparse.ArgumentParser(description="天气预报缓存新鲜度审计")
    parser.add_argument("--live", metavar="CITY", help="额外请求一次真实高德接口做对账")
    parser.add_argument("--cache-dir", default=str(PLANNER_CONTEXT_CACHE_DIR), help="缓存目录")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    print(f"缓存目录：{cache_dir / WEATHER_CACHE_SUBDIR}")
    rows = _load_rows(cache_dir)
    if not rows:
        print("没有找到天气缓存文件；先跑一次规划，或加 --live 直接对账。")
    else:
        _print_table(rows, date.today() + timedelta(days=TRIP_START_OFFSET_DAYS))

    if args.live:
        _live_check(args.live)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
