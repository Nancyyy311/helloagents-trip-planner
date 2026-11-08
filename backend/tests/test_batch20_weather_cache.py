"""批次20：天气预报缓存"跨天不失效"的根因回归测试。

现场问题：2026-11-09 规划 11-11~11-15，结果页只有 11-11 有天气，
后面 4 天全是"远期天气暂无准确预报"。高德本身对未来 4 天有预报，
真正的原因是磁盘缓存里躺着 11-08 23:42 拉的那份响应（casts 只到 11-11），
而缓存 TTL 走的是通用配置（当时 1440 小时），于是复用了昨天的预报。

本文件锁三件事：
1. 缓存里的预报不从"今天"开始 → 必须重新拉，不能直接返回；
2. 当天拉的预报 → 继续命中缓存，不要为了修 bug 把缓存全废掉；
3. 空结果（城市解析失败）→ 不被新鲜度判据反复穿透，交给短 TTL 兜住。

不联网：把 httpx.get 换成可计数的桩。
"""

from datetime import date, datetime, timedelta

import pytest

from app.planner import amap as amap_mod
from app.planner.amap import AmapPlannerClient, weather_forecast_dates, weather_window_starts_today


def _payload(casts_dates):
    return {
        "status": "1",
        "info": "OK",
        "forecasts": [
            {
                "city": "测试市",
                "reporttime": f"{casts_dates[0]} 20:00:00",
                "casts": [
                    {
                        "date": day,
                        "dayweather": "晴",
                        "nightweather": "多云",
                        "daytemp": "20",
                        "nighttemp": "12",
                        "daywind": "东南",
                        "daypower": "1-3",
                    }
                    for day in casts_dates
                ],
            }
        ],
    }


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeHttp:
    """记录调用次数，返回固定 payload。"""

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return _FakeResponse(self.payload)

    def __call__(self, url, params=None, timeout=None):
        return self.get(url, params, timeout)


@pytest.fixture
def no_qps_wait(monkeypatch):
    """限速对测试没意义，去掉 0.33s/次的等待。"""
    monkeypatch.setattr(amap_mod, "AMAP_MIN_INTERVAL_SECONDS", 0.0)


def _write_cache(client, payload, created_ago_seconds, ttl_seconds):
    path = client._amap_cache_path("/weather/weatherInfo", {"city": "测试市", "extensions": "all"})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        __import__("json").dumps(
            {
                "created_at": datetime.now().timestamp() - created_ago_seconds,
                "ttl_seconds": ttl_seconds,
                "data": payload,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_stale_forecast_is_refetched_even_if_within_ttl(tmp_path, monkeypatch, no_qps_wait):
    """昨天的预报（哪怕只过了 2 小时、远没到 TTL）不能拿来用。"""
    today = date.today()
    yesterday = today - timedelta(days=1)
    # 复现现场：昨天 23:42 缓存，casts 从昨天开始，到 3 天后结束
    stale = _payload([(yesterday + timedelta(days=i)).isoformat() for i in range(4)])
    fresh = _payload([(today + timedelta(days=i)).isoformat() for i in range(4)])

    client = AmapPlannerClient("fake-key", cache_dir=tmp_path)
    # 用"过期前"的 TTL 写进缓存，模拟 1440 小时那种长 TTL
    _write_cache(client, stale, created_ago_seconds=2 * 3600, ttl_seconds=60 * 24 * 3600)

    fake = _FakeHttp(fresh)
    monkeypatch.setattr(amap_mod.httpx, "get", fake.get)

    data = client.get("/weather/weatherInfo", {"city": "测试市", "extensions": "all"})

    assert fake.calls == 1, "预报窗口不从今天开始时必须重新请求，不能直接返回缓存"
    assert weather_forecast_dates(data)[0] == today.isoformat()


def test_same_day_forecast_is_served_from_cache(tmp_path, monkeypatch, no_qps_wait):
    """当天拉的预报要能命中缓存，别为了修 bug 把缓存废掉。"""
    today = date.today()
    fresh = _payload([(today + timedelta(days=i)).isoformat() for i in range(4)])

    client = AmapPlannerClient("fake-key", cache_dir=tmp_path)
    _write_cache(client, fresh, created_ago_seconds=60, ttl_seconds=3600)

    fake = _FakeHttp(fresh)
    monkeypatch.setattr(amap_mod.httpx, "get", fake.get)

    data = client.get("/weather/weatherInfo", {"city": "测试市", "extensions": "all"})

    assert fake.calls == 0, "同一天内应命中缓存"
    assert weather_forecast_dates(data) == weather_forecast_dates(fresh)


def test_forecast_older_than_weather_ttl_is_refetched(tmp_path, monkeypatch, no_qps_wait):
    """即使 casts 恰好从今天开始，超过短 TTL 也要重拉（预报一天会更新好几次）。"""
    today = date.today()
    payload = _payload([(today + timedelta(days=i)).isoformat() for i in range(4)])

    client = AmapPlannerClient("fake-key", cache_dir=tmp_path)
    _write_cache(client, payload, created_ago_seconds=2 * 3600, ttl_seconds=3600)

    fake = _FakeHttp(payload)
    monkeypatch.setattr(amap_mod.httpx, "get", fake.get)

    client.get("/weather/weatherInfo", {"city": "测试市", "extensions": "all"})
    assert fake.calls == 1


def test_empty_forecast_is_not_hammered(tmp_path, monkeypatch, no_qps_wait):
    """城市解析不到时高德返回空 forecasts：短 TTL 内不要每请求都打接口。"""
    empty = {"status": "1", "info": "OK", "forecasts": []}

    client = AmapPlannerClient("fake-key", cache_dir=tmp_path)
    fake = _FakeHttp(empty)
    monkeypatch.setattr(amap_mod.httpx, "get", fake.get)

    client.get("/weather/weatherInfo", {"city": "不存在市", "extensions": "all"})
    client.get("/weather/weatherInfo", {"city": "不存在市", "extensions": "all"})

    assert fake.calls == 1, "空结果应被短 TTL 缓存住"


def test_non_weather_path_keeps_generic_ttl(tmp_path, monkeypatch, no_qps_wait):
    """只有天气走新鲜度判据，POI 搜索仍然吃通用 TTL。"""
    client = AmapPlannerClient("fake-key", cache_dir=tmp_path)
    assert client._ttl_seconds_for("/place/text") == amap_mod.PLANNER_CONTEXT_AMAP_CACHE_TTL_SECONDS
    assert client._ttl_seconds_for("/weather/weatherInfo") == amap_mod.AMAP_WEATHER_CACHE_TTL_SECONDS
    assert amap_mod.AMAP_WEATHER_CACHE_TTL_SECONDS < amap_mod.PLANNER_CONTEXT_AMAP_CACHE_TTL_SECONDS


def test_weather_window_helpers():
    today = date.today()
    today_text = today.isoformat()
    from_yesterday = _payload([(today - timedelta(days=1)).isoformat(), today_text])
    from_today = _payload([today_text, (today + timedelta(days=1)).isoformat()])

    assert weather_window_starts_today(from_today) is True
    assert weather_window_starts_today(from_yesterday) is False
    assert weather_window_starts_today({"forecasts": []}) is True  # 空结果不穿透
    assert weather_window_starts_today(None) is True
    assert weather_forecast_dates(from_today) == [today_text, (today + timedelta(days=1)).isoformat()]


def test_alignment_only_fills_dates_inside_forecast_window():
    """对齐函数本身没问题：窗口外的日子保持"远期"，不能张冠李戴。"""
    from app.models.schemas import TripRequest
    from app.planner.weather import align_trip_weather, normalize_weather

    today = date.today()
    request = TripRequest(
        city="测试市",
        start_date=(today + timedelta(days=2)).isoformat(),
        end_date=(today + timedelta(days=6)).isoformat(),
        travel_days=5,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=[],
        free_text_input="",
        party={"adults": 1, "children": 0, "elders": 0, "total": 1, "companion_type": "solo"},
        budget_constraint={
            "amount": 3000,
            "scope": "total",
            "currency": "CNY",
            "budget_level": "standard",
            "strictness": "none",
        },
    )
    payload = _payload([(today + timedelta(days=i)).isoformat() for i in range(4)])
    rows = align_trip_weather(request, normalize_weather(payload))

    assert len(rows) == 5
    covered = [row for row in rows if row["source"] == "amap_forecast"]
    missing = [row for row in rows if row["source"] == "unavailable_future_weather"]
    # 窗口是今天起 4 天（today,+1,+2,+3），行程从 today+2 开始 → 只覆盖 2 天，其余 3 天远期
    assert len(covered) == 2
    assert len(missing) == 3
    assert all(row["day_weather"] == "远期天气暂无准确预报" for row in missing)
