"""批次11：修"结果页很多都加载不出来"——图片链路重构 + 事件循环解阻塞。

问题现场（用户真机实测截图 + curl 取证）：
1. `/api/poi/photo?name=故宫` 耗时 **22 秒**、返回 photo_url: null ——
   Unsplash 搜索 API（api.unsplash.com）在国内网络不可达（curl 8s 超时），
   且 .env 里 UNSPLASH_ACCESS_KEY 是空串，每次请求白跑两次 10s 超时；
2. 该路由是 async def 却调用同步 requests.get，**阻塞整个事件循环**：
   结果页 15 个景点并发请求图片实际串行 15 × 22s ≈ 5 分半，
   期间餐饮坐标回填、任务轮询全部卡死——这就是"很多都加载不出来"；
3. 前端没有 .env，VITE_AMAP_WEB_JS_KEY 为 undefined，JSAPI 直接加载失败，
   全程地图 / 每日地图白屏（前端侧改动，本文件只测后端契约）。

改造（见 app/planner/local_photo_table.py 模块注释）：
- 图片不搜索、查本地映射表（零外网、微秒级）；
- 命中返回磁盘缓存代理地址 /api/poi/photo/asset/{id}（浏览器只连 localhost）；
- 代理端点用同步 def 路由（线程池执行），下载再慢也不碰事件循环。
"""

import threading

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.planner import local_photo_table as lpt

client = TestClient(app)


# === 本地映射表 ===

class TestLookupPhotoId:
    def test_keyword_hit_shanghai(self):
        # "外滩" 必须命中外滩夜景图（图片内容已人工核验）
        assert lpt.lookup_photo_id("外滩", "上海") == lpt.SHANGHAI_BUND

    def test_keyword_longest_match_wins(self):
        # 景点名同时包含"森林公园"和"海滩"相关词时，更具体的词优先。
        # 这里验证长度排序：长词表里"森林公园"必须先于短词被匹配。
        assert (
            lpt.lookup_photo_id("亚龙湾热带天堂森林公园", "三亚")
            == lpt.FOREST_PATH
        )

    def test_city_fallback_with_suffix(self):
        # "三亚市" 应去后缀命中城市表
        assert lpt.lookup_photo_id("某个无名公园", "三亚市") == lpt.BEACH_SUNSET

    def test_miss_returns_none(self):
        # 未命中必须显式返回 None（不是随机塞一张城市图），
        # 宁可给前端渐变占位卡，也不给内容对不上的图
        assert lpt.lookup_photo_id("不存在的景点xyz", "克拉玛依") is None

    def test_empty_inputs_safe(self):
        assert lpt.lookup_photo_id("", "") is None


# === /api/poi/photo 契约 ===

class TestPhotoEndpoint:
    def test_hit_returns_proxy_url_and_source(self):
        resp = client.get("/api/poi/photo", params={"name": "外滩", "city": "上海"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["source"] == "local_table"
        assert data["photo_url"].startswith("/api/poi/photo/asset/")

    def test_miss_returns_null_not_500(self):
        # 旧实现：查不到图也返回 success:true + photo_url:null（这个契约保留），
        # 但耗时从 22s 降到毫秒级（本测试能跑完本身就是证据）
        resp = client.get(
            "/api/poi/photo", params={"name": "不存在的景点xyz", "city": "克拉玛依"}
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["photo_url"] is None
        assert data["source"] == "placeholder"

    def test_endpoint_is_fast_no_external_io(self):
        # 回归哨兵：整个查表是纯内存操作。22 秒的旧实现会让这个测试直接超时
        import time

        t0 = time.monotonic()
        for _ in range(20):
            client.get("/api/poi/photo", params={"name": "故宫", "city": "北京"})
        elapsed = time.monotonic() - t0
        assert elapsed < 5.0, f"20 次请求耗时 {elapsed:.1f}s，疑似又走到外网搜索"


# === /api/poi/photo/asset/{id} 磁盘缓存代理 ===

class TestPhotoAssetEndpoint:
    @pytest.fixture(autouse=True)
    def _isolate_cache(self, tmp_path, monkeypatch):
        # 每个用例独立的缓存目录，互不污染、不动真实 .cache/
        monkeypatch.setattr(lpt, "CACHE_DIR", tmp_path)
        yield

    def _stub_download(self, monkeypatch, payload: bytes, calls: list):
        class _FakeResp:
            status_code = 200

            def raise_for_status(self):
                pass

            @property
            def content(self):
                return payload

        def _fake_get(url, timeout=None):
            calls.append(url)
            return _FakeResp()

        monkeypatch.setattr(lpt.requests, "get", _fake_get)

    def test_serves_image_and_caches_to_disk(self, monkeypatch, tmp_path):
        calls: list = []
        payload = b"\xff\xd8fakejpeg" * 200  # >1KB，过短内容过滤
        self._stub_download(monkeypatch, payload, calls)

        resp = client.get(f"/api/poi/photo/asset/{lpt.SHANGHAI_BUND}")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert resp.content == payload
        # 缓存文件确实落盘
        assert lpt.cached_photo_path(lpt.SHANGHAI_BUND).is_file()

    def test_second_request_reads_cache_no_network(self, monkeypatch):
        calls: list = []
        payload = b"\xff\xd8fakejpeg" * 200
        self._stub_download(monkeypatch, payload, calls)

        client.get(f"/api/poi/photo/asset/{lpt.SHANGHAI_BUND}")
        client.get(f"/api/poi/photo/asset/{lpt.SHANGHAI_BUND}")
        client.get(f"/api/poi/photo/asset/{lpt.SHANGHAI_BUND}")
        # 三次请求只有第一次触网——磁盘缓存生效
        assert len(calls) == 1

    def test_unknown_photo_id_is_404_open_proxy_guard(self, monkeypatch):
        # 代理只认表内 id：任意 id / 路径穿越一律 404，
        # 不能被当成"帮我下载任意图"的开放代理
        calls: list = []
        self._stub_download(monkeypatch, b"x" * 2000, calls)
        for bad in ["../../etc/passwd", "0000000000-not-in-table", "", "a"]:
            resp = client.get(f"/api/poi/photo/asset/{bad}")
            assert resp.status_code in (404, 307, 404), bad
        assert calls == []

    def test_download_failure_is_404_not_500_hang(self, monkeypatch):
        # CDN 抖动时的降级契约：404 让前端 <img> 走 @error 兜底，
        # 而不是 500 或者像旧实现一样把请求挂 22 秒
        def _fail(url, timeout=None):
            raise lpt.requests.ConnectionError("network down")

        monkeypatch.setattr(lpt.requests, "get", _fail)
        resp = client.get(f"/api/poi/photo/asset/{lpt.SHANGHAI_BUND}")
        assert resp.status_code == 404

    def test_concurrent_same_photo_downloads_once(self, monkeypatch, tmp_path):
        # 10 个景点共用一张城市图时，并发只应发起 1 次 CDN 下载
        calls: list = []
        gate = threading.Event()

        def _slow_get(url, timeout=None):
            calls.append(url)
            gate.wait(timeout=5)
            return type("R", (), {"status_code": 200, "raise_for_status": lambda s: None,
                                  "content": b"\xff\xd8" + b"j" * 2000})()

        monkeypatch.setattr(lpt.requests, "get", _slow_get)
        results: list = []

        def _fetch():
            results.append(lpt.load_photo_bytes(lpt.CITY_SKYLINE))

        threads = [threading.Thread(target=_fetch) for _ in range(8)]
        for t in threads:
            t.start()
        import time

        time.sleep(0.2)
        gate.set()
        for t in threads:
            t.join(timeout=5)
        assert all(r == b"\xff\xd8" + b"j" * 2000 for r in results)
        assert len(calls) == 1
