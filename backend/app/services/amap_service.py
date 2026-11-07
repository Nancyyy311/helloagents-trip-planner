"""高德地图服务封装（HTTP 版）。

改造记录（矛盾点清理）:
1. 原实现的 ``search_poi`` / ``get_weather`` 里写着 ``# TODO: 解析实际的POI数据``
   然后直接 ``return []``——两个方法**恒返回空列表**。而 ``/api/map/poi``、
   ``/api/map/weather``、``/api/poi/search`` 全部依赖它，
   等于这些接口永远"成功但没数据"，前端地图搜索必然是空白。
2. 原实现依赖 ``uvx amap-mcp-server`` 子进程；环境里没装 uvx 时
   构造 ``AmapService`` 直接抛异常，整个 /api/map/* 全部 500。
3. 原 ``/api/map/health`` 只读 ``mcp_tool._available_tools`` 的长度就报 healthy，
   即使所有查询都返回空——健康检查形同虚设。

现改成直接调用高德 HTTP API（复用 ``AmapPlannerClient``，自带磁盘缓存 + QPS 限速），
对外方法签名保持不变，调用方无需改动；行程规划与地图接口共用同一套客户端，
缓存也就能互相复用。
"""

from __future__ import annotations

import json
import time
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from ..config import get_settings
from ..models.schemas import Location, POIInfo, WeatherInfo
from ..planner.amap import AmapPlannerClient


# 健康探测结果缓存：60 秒内不重复打上游，避免探活把额度烧光
_PROBE_TTL_SECONDS = 60.0

_amap_http_client: Optional[AmapPlannerClient] = None
_client_lock = Lock()
_probe_cache: Dict[str, Any] = {"at": 0.0, "result": None}


def _get_client() -> AmapPlannerClient:
    """懒加载高德 HTTP 客户端（进程级单例）。"""
    global _amap_http_client
    if _amap_http_client is None:
        with _client_lock:
            if _amap_http_client is None:
                settings = get_settings()
                if not settings.amap_api_key:
                    raise ValueError(
                        "高德地图 API Key 未配置，请在 .env 中设置 AMAP_API_KEY"
                    )
                _amap_http_client = AmapPlannerClient(settings.amap_api_key)
    return _amap_http_client


def _split_location(text: Any) -> Tuple[Optional[float], Optional[float]]:
    """把高德的 "lng,lat" 字符串拆成两个 float。"""
    if not isinstance(text, str) or "," not in text:
        return None, None
    longitude, _, latitude = text.partition(",")
    try:
        return float(longitude), float(latitude)
    except (TypeError, ValueError):
        return None, None


class AmapService:
    """高德地图服务封装类（HTTP，无 MCP 子进程依赖）。"""

    # ---------------------------------------------------------------- POI
    def search_poi(
        self,
        keywords: str,
        city: str,
        citylimit: bool = True,
        limit: int = 10,
    ) -> List[POIInfo]:
        """按关键词搜索真实 POI；取不到坐标的记录会被丢弃（与规划链路一致）。"""
        raw = _get_client().get(
            "/place/text",
            {
                "keywords": keywords,
                "city": city,
                "citylimit": str(citylimit).lower(),
                "extensions": "all",
                "offset": limit,
                "page": 1,
            },
        )

        results: List[POIInfo] = []
        for poi in raw.get("pois") or []:
            longitude, latitude = _split_location(poi.get("location"))
            if longitude is None or latitude is None:
                continue
            address = poi.get("address")
            results.append(
                POIInfo(
                    id=str(poi.get("id") or ""),
                    name=str(poi.get("name") or ""),
                    type=str(poi.get("type") or ""),
                    address=address if isinstance(address, str) else "",
                    location=Location(longitude=longitude, latitude=latitude),
                    tel=poi.get("tel") if isinstance(poi.get("tel"), str) else None,
                )
            )
        return results

    # ------------------------------------------------------------- weather
    def get_weather(self, city: str) -> List[WeatherInfo]:
        """查询城市逐日天气预报（高德只提供未来几天，超期返回空列表）。"""
        raw = _get_client().get(
            "/weather/weatherInfo",
            {"city": city, "extensions": "all"},
        )
        forecasts = raw.get("forecasts") or []
        if not forecasts:
            return []

        casts = (forecasts[0] or {}).get("casts") or []
        return [
            WeatherInfo(
                date=str(cast.get("date") or ""),
                day_weather=str(cast.get("dayweather") or ""),
                night_weather=str(cast.get("nightweather") or ""),
                day_temp=cast.get("daytemp") or 0,
                night_temp=cast.get("nighttemp") or 0,
                wind_direction=str(cast.get("daywind") or ""),
                wind_power=str(cast.get("daypower") or ""),
            )
            for cast in casts
        ]

    # --------------------------------------------------------------- route
    def plan_route(
        self,
        origin_address: str,
        destination_address: str,
        origin_city: Optional[str] = None,
        destination_city: Optional[str] = None,
        route_type: str = "walking",
    ) -> Dict[str, Any]:
        """规划两点之间的路线，返回 {distance, duration, route_type, description}。"""
        route_type = route_type if route_type in ("walking", "driving", "transit") else "walking"
        params: Dict[str, Any] = {
            "origin": origin_address,
            "destination": destination_address,
        }
        if route_type == "walking":
            path = "/direction/walking"
        elif route_type == "driving":
            path = "/direction/driving"
            params["extensions"] = "base"
        else:
            path = "/direction/transit/integrated"
            params["city"] = origin_city or destination_city or ""
            params["cityd"] = destination_city or origin_city or ""
            params["strategy"] = 0

        raw = _get_client().get(path, params)
        route = raw.get("route") or {}

        if route_type == "transit":
            transits = route.get("transits") or []
            if not transits:
                raise ValueError(f"未找到从「{origin_address}」到「{destination_address}」的公交方案")
            best = transits[0]
            segments = len(best.get("segments") or [])
            return {
                "distance": float(best.get("distance") or 0),
                "duration": int(float(best.get("duration") or 0)),
                "route_type": route_type,
                "description": f"公共交通方案，共 {segments} 段换乘",
            }

        paths = route.get("paths") or []
        if not paths:
            raise ValueError(f"未找到从「{origin_address}」到「{destination_address}」的路线")
        best = paths[0]
        steps = len(best.get("steps") or [])
        return {
            "distance": float(best.get("distance") or 0),
            "duration": int(float(best.get("duration") or 0)),
            "route_type": route_type,
            "description": f"{best.get('strategy') or route_type} 方案，共 {steps} 段",
        }

    # ------------------------------------------------------------- geocode
    def geocode(self, address: str, city: Optional[str] = None) -> Optional[Location]:
        """把结构化地址转成经纬度。"""
        raw = _get_client().get(
            "/geocode/geo",
            {"address": address, "city": city or ""},
        )
        geocodes = raw.get("geocodes") or []
        if not geocodes:
            return None
        longitude, latitude = _split_location(geocodes[0].get("location"))
        if longitude is None or latitude is None:
            return None
        return Location(longitude=longitude, latitude=latitude)

    # --------------------------------------------------------- poi detail
    def get_poi_detail(self, poi_id: str) -> Dict[str, Any]:
        """获取 POI 详情（含图片），直接返回高德原始字段。"""
        raw = _get_client().get("/place/detail", {"id": poi_id, "extensions": "all"})
        pois = raw.get("pois") or []
        if not pois:
            return {}
        detail = dict(pois[0])
        photos = detail.get("photos") or []
        if isinstance(photos, list):
            detail["photos"] = [
                photo.get("url") for photo in photos if isinstance(photo, dict) and photo.get("url")
            ]
        return detail

    # -------------------------------------------------------------- health
    def probe(self) -> Dict[str, Any]:
        """真实探测上游可用性（结果缓存 60 秒，避免探活烧额度）。

        原实现只读 MCP 工具数量就报 healthy，即使所有查询恒返回空。
        """
        now = time.time()
        cached = _probe_cache.get("result")
        if cached and now - float(_probe_cache.get("at") or 0) < _PROBE_TTL_SECONDS:
            return dict(cached)

        try:
            location = self.geocode("天安门", city="北京")
            if location is None:
                result: Dict[str, Any] = {
                    "status": "degraded",
                    "reason": "高德地理编码返回空结果",
                }
            else:
                result = {
                    "status": "healthy",
                    "upstream": "amap-http",
                    "sample": {"longitude": location.longitude, "latitude": location.latitude},
                }
        except Exception as error:  # noqa: BLE001 - 探活失败要如实反映
            result = {"status": "unhealthy", "reason": str(error)}

        _probe_cache["at"] = now
        _probe_cache["result"] = result
        return dict(result)


# 进程级服务实例
_amap_service: Optional[AmapService] = None


def get_amap_service() -> AmapService:
    """获取高德地图服务实例（单例）。"""
    global _amap_service
    if _amap_service is None:
        _amap_service = AmapService()
    return _amap_service


def dumps_response(payload: Any) -> str:
    """调试用：把响应安全转成 JSON 字符串（不抛异常）。"""
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:  # pragma: no cover - 仅调试用
        return str(payload)
