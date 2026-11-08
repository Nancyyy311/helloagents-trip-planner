"""高德 API 调用、缓存和 POI 搜索封装。"""

import hashlib
import json
import os
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from .pois import dedupe_pois, filter_pois, normalize_pois, rank_pois


AMAP_BASE_URL = "https://restapi.amap.com/v3"
AMAP_HTTP_TIMEOUT = int(os.getenv("AMAP_HTTP_TIMEOUT", "30"))
AMAP_SEARCH_OFFSET = int(os.getenv("AMAP_SEARCH_OFFSET", "20"))
PLANNER_CONTEXT_PER_KEYWORD_LIMIT = int(os.getenv("PLANNER_CONTEXT_PER_KEYWORD_LIMIT", "5"))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLANNER_CONTEXT_CACHE_DIR = Path(
    os.getenv("PLANNER_CONTEXT_CACHE_DIR", str(PROJECT_ROOT / "training/data/cache/planner_context"))
)
PLANNER_CONTEXT_CACHE_TTL_SECONDS = int(
    os.getenv("PLANNER_CONTEXT_CACHE_TTL_SECONDS", str(60 * 24 * 60 * 60))
)
PLANNER_CONTEXT_AMAP_CACHE_TTL_SECONDS = int(
    os.getenv("PLANNER_CONTEXT_AMAP_CACHE_TTL_SECONDS", str(PLANNER_CONTEXT_CACHE_TTL_SECONDS))
)
PLANNER_CONTEXT_CLASSIC_CACHE_TTL_SECONDS = int(
    os.getenv("PLANNER_CONTEXT_CLASSIC_CACHE_TTL_SECONDS", str(PLANNER_CONTEXT_CACHE_TTL_SECONDS))
)
PLANNER_CONTEXT_CACHE_VERBOSE = os.getenv("PLANNER_CONTEXT_CACHE_VERBOSE", "0") == "1"
AMAP_QPS_LIMIT = float(os.getenv("AMAP_QPS_LIMIT", "3"))
AMAP_MIN_INTERVAL_SECONDS = 1.0 / AMAP_QPS_LIMIT if AMAP_QPS_LIMIT > 0 else 0.0

# 天气缓存单独治理：天气预报是"按天滚动"的，不是"按 TTL 过期"的。
# 复用 PLANNER_CONTEXT_CACHE_TTL_SECONDS（默认 24 小时，线上曾有 60 天的配置）
# 会出现：昨天 23:42 缓存了 11-08~11-11 的预报，今天 11-09 规划 11-11~11-15 时，
# 直接用昨天的预报，导致只有 11-11 一天有天气、后面 4 天全"远期天气暂无准确预报"。
# 规则改成：TTL 只兜"当天内的重复请求"，跨天必须重新拉——由 casts[0].date == 今天 判定。
AMAP_WEATHER_PATH = "/weather/weatherInfo"
AMAP_WEATHER_CACHE_TTL_SECONDS = int(os.getenv("AMAP_WEATHER_CACHE_TTL_SECONDS", str(60 * 60)))

_AMAP_RATE_LIMIT_LOCK = threading.Lock()
_AMAP_LAST_REQUEST_AT = 0.0


def weather_forecast_dates(payload: Optional[Dict[str, Any]]) -> List[str]:
    """取出高德天气预报覆盖的日期列表（按接口返回顺序）。"""
    forecasts = (payload or {}).get("forecasts") or []
    if not forecasts:
        return []
    casts = (forecasts[0] or {}).get("casts") or []
    return [str((cast or {}).get("date") or "") for cast in casts if (cast or {}).get("date")]


def weather_window_starts_today(payload: Optional[Dict[str, Any]], today: Optional[date] = None) -> bool:
    """判断一份天气预报是否"今天拉的"：高德的 casts[0] 永远是当天。

    这是天气缓存的新鲜度判据，比 TTL 更准——按 TTL 存 24 小时的话，
    昨天 23:42 的预报今天仍然"没过期"，但它只覆盖到 11-11，
    规划 11-11~11-15 时后面几天就只能写"远期天气暂无准确预报"。
    空结果（城市解析失败等）不走这个判据，交给短 TTL 兜住，避免每次请求都打接口。
    """
    dates = weather_forecast_dates(payload)
    if not dates:
        return True
    return dates[0] == (today or date.today()).isoformat()


def set_amap_qps_limit(qps_limit: float) -> None:
    """Override the process-local Amap HTTP QPS limit at runtime."""
    if qps_limit < 0:
        raise ValueError("AMAP QPS limit must be >= 0")

    global AMAP_QPS_LIMIT, AMAP_MIN_INTERVAL_SECONDS, _AMAP_LAST_REQUEST_AT
    AMAP_QPS_LIMIT = float(qps_limit)
    AMAP_MIN_INTERVAL_SECONDS = 1.0 / AMAP_QPS_LIMIT if AMAP_QPS_LIMIT > 0 else 0.0
    with _AMAP_RATE_LIMIT_LOCK:
        _AMAP_LAST_REQUEST_AT = 0.0


class AmapPlannerClient:
    """面向 PlannerContext 的高德查询客户端。"""

    def __init__(self, api_key: str, cache_dir: Path = PLANNER_CONTEXT_CACHE_DIR):
        self.api_key = api_key
        self.cache_dir = self._resolve_cache_dir(cache_dir)

    def get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """调用高德HTTP API，并在入口统一做缓存和限速。"""
        if not self.api_key:
            raise ValueError("AMAP_API_KEY未配置")

        query_params = {key: value for key, value in params.items() if value not in (None, "")}
        cache_path = self._amap_cache_path(path, query_params)
        ttl_seconds = self._ttl_seconds_for(path)
        cached = self._read_cache(
            cache_path,
            value_key="data",
            ttl_seconds=ttl_seconds,
        )
        if cached is not None and self._cache_still_current(path, cached):
            if PLANNER_CONTEXT_CACHE_VERBOSE:
                print(f"  - amap缓存命中: {path} {self._cache_label(query_params)}")
            return cached
        if cached is not None:
            # 没过 TTL 但预报窗口已过期（跨天），按未命中处理，往下重新请求
            cached = None

        request_params = {**query_params, "key": self.api_key}
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                self._wait_for_amap_slot()
                response = httpx.get(f"{AMAP_BASE_URL}{path}", params=request_params, timeout=AMAP_HTTP_TIMEOUT)
                response.raise_for_status()
                data = response.json()
                if data.get("status") == "0":
                    raise RuntimeError(f"高德API错误: {data.get('info')} ({data.get('infocode')})")
                self._write_cache(
                    cache_path,
                    data,
                    value_key="data",
                    ttl_seconds=ttl_seconds,
                    label="amap",
                )
                return data
            except Exception as exc:
                last_exc = exc
                time.sleep(0.5 + attempt)
        raise RuntimeError(f"高德API请求失败: {last_exc}") from last_exc

    @staticmethod
    def _is_weather_path(path: str) -> bool:
        return path.rstrip("/").endswith(AMAP_WEATHER_PATH)

    def _ttl_seconds_for(self, path: str) -> int:
        """天气走短 TTL，其余接口沿用通用 TTL。"""
        return AMAP_WEATHER_CACHE_TTL_SECONDS if self._is_weather_path(path) else PLANNER_CONTEXT_AMAP_CACHE_TTL_SECONDS

    def _cache_still_current(self, path: str, payload: Dict[str, Any]) -> bool:
        """TTL 之外再按业务语义判新鲜度：天气预报必须从"今天"开始。"""
        if not self._is_weather_path(path):
            return True
        return weather_window_starts_today(payload)

    def search_keywords(
        self,
        city: str,
        keywords: List[str],
        source_role: str,
        limit: int,
        require_location: bool = True,
        source_bucket: str = "",
    ) -> List[Dict[str, Any]]:
        """按多个关键词搜索POI并去重。"""
        rows = []
        for keyword in keywords:
            raw = self.get(
                "/place/text",
                {
                    "keywords": keyword,
                    "city": city,
                    "citylimit": "true",
                    "extensions": "all",
                    "offset": AMAP_SEARCH_OFFSET,
                    "page": 1,
                },
            )
            normalized = normalize_pois(raw, keyword, source_role, require_location, source_bucket)
            normalized = filter_pois(normalized, source_role)
            normalized = rank_pois(normalized, source_role)
            rows.extend(normalized[:PLANNER_CONTEXT_PER_KEYWORD_LIMIT])
        rows = dedupe_pois(rows)
        rows = rank_pois(rows, source_role)
        return rows[:limit]

    def search_classic_pois(self, city: str, keywords: List[str], limit: int) -> List[Dict[str, Any]]:
        """查询城市经典景点；这部分和用户无关，适合长 TTL 本地缓存。"""
        cache_path = self._classic_cache_path(city, keywords)
        cached = self._read_cache(
            cache_path,
            value_key="rows",
            ttl_seconds=PLANNER_CONTEXT_CLASSIC_CACHE_TTL_SECONDS,
        )
        if cached is not None:
            # print(f"  - classic_pois缓存命中: {city} ({len(cached)}条)")
            return cached

        rows = self.search_keywords(
            city,
            keywords,
            source_role="scenic",
            limit=limit,
            source_bucket="classic",
        )
        self._write_cache(
            cache_path,
            rows,
            value_key="rows",
            ttl_seconds=PLANNER_CONTEXT_CLASSIC_CACHE_TTL_SECONDS,
            label="classic_pois",
        )
        return rows

    def _resolve_cache_dir(self, path: Path) -> Path:
        """把相对缓存目录解析到项目根目录附近，避免依赖启动目录。"""
        if path.is_absolute():
            return path
        # backend/app/planner/amap.py -> project root
        project_root = Path(__file__).resolve().parents[3]
        return project_root / path

    def _classic_cache_path(self, city: str, keywords: List[str]) -> Path:
        """生成城市经典候选缓存路径。"""
        safe_city = "".join(ch for ch in city if ch.isalnum() or ch in ("-", "_")) or "unknown"
        keyword_sig = "_".join(keywords)
        # 关键词基本固定，hash 用内置 hash 不稳定；这里用长度较短的手写签名足够。
        suffix = str(abs(sum(ord(ch) for ch in keyword_sig)))
        return self.cache_dir / "classic_pois" / f"{safe_city}_{suffix}.json"

    def _amap_cache_path(self, path: str, params: Dict[str, Any]) -> Path:
        """生成高德原始HTTP响应缓存路径，不把API key写进缓存签名。"""
        safe_path = path.strip("/").replace("/", "_") or "root"
        payload = {
            "path": path,
            "params": {key: str(params[key]) for key in sorted(params)},
        }
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]
        label = self._cache_label(params) or "query"
        return self.cache_dir / "amap_api" / safe_path / f"{label}_{digest}.json"

    def _cache_label(self, params: Dict[str, Any]) -> str:
        """生成便于人工查看的缓存文件名前缀。"""
        parts = []
        for key in ("city", "keywords", "extensions"):
            value = str(params.get(key) or "").strip()
            if value:
                parts.append(self._safe_cache_text(value)[:24])
        return "_".join(parts)

    def _safe_cache_text(self, value: str) -> str:
        """把城市/关键词压成适合文件名的短文本。"""
        safe = "".join(ch for ch in value if ch.isalnum() or ch in ("-", "_"))
        return safe or "query"

    def _wait_for_amap_slot(self) -> None:
        """进程内限速，避免并发cache miss时超过高德QPS限制。"""
        if AMAP_MIN_INTERVAL_SECONDS <= 0:
            return

        global _AMAP_LAST_REQUEST_AT
        with _AMAP_RATE_LIMIT_LOCK:
            now = time.monotonic()
            wait_seconds = AMAP_MIN_INTERVAL_SECONDS - (now - _AMAP_LAST_REQUEST_AT)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            _AMAP_LAST_REQUEST_AT = time.monotonic()

    def _read_cache(
        self,
        path: Path,
        value_key: str = "rows",
        ttl_seconds: int = PLANNER_CONTEXT_CACHE_TTL_SECONDS,
    ) -> Optional[Any]:
        """读取未过期缓存。"""
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            created_at = float(data.get("created_at", 0))
            if time.time() - created_at > ttl_seconds:
                return None
            return data.get(value_key)
        except Exception:
            return None

    def _write_cache(
        self,
        path: Path,
        value: Any,
        value_key: str = "rows",
        ttl_seconds: int = PLANNER_CONTEXT_CACHE_TTL_SECONDS,
        label: str = "cache",
    ) -> None:
        """写入缓存；失败不影响主流程。"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "created_at": time.time(),
                "ttl_seconds": ttl_seconds,
                value_key: value,
            }
            tmp_path = path.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(path)
        except Exception as exc:
            print(f"⚠️  {label}缓存写入失败: {exc}")
