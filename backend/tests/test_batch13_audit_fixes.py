"""批次13：交付前漏洞核查发现的问题，固化为回归测试。

这一批不加功能，只做"别在交付前留明显错误"。三个发现对应三组断言：

1. **负向约束误伤故宫**：`TERM_EXPANSIONS["寺庙"]` 原本含单字 `"宫"`，
   用户说"不想去寺庙"时，`故宫博物院` 会被判违规 —— rerank 会把这个
   中国最重要的景点之一排到最后。已移除，但必须有测试钉住，
   否则下次有人"顺手补充同义词"又会加回去。

2. **路由阻塞事件循环**：`/map/*` 与 `/poi/search`、`/poi/detail` 六个路由
   曾经是 `async def`，内部却调同步 httpx（高德接口 timeout=30s、
   重试 3 次 → 最坏 92 秒）。这是批次11 图片接口同一个病，当时只修了一半。
   用 `inspect.iscoroutinefunction` 直接断言"这些路由必须不是协程"，
   因为"同步路由由 FastAPI 丢线程池"是框架契约，比压测更确定。

3. **`.env.example` 里的死配置**：前端从不读 `VITE_AMAP_WEB_KEY`，
   留着只会让人以为要填两个高德 key。这条没法用测试表达，
   写在台账里。
"""

from __future__ import annotations

import inspect

import pytest

from app.api.routes import map as map_routes
from app.api.routes import poi as poi_routes
from app.planner.constraints import TERM_EXPANSIONS, _hit, expand_terms


def _endpoints(module):
    """把一个 APIRouter 的 (path -> endpoint) 摊平，方便按路径断言。"""
    return {
        getattr(route, "path", ""): getattr(route, "endpoint", None)
        for route in module.router.routes
    }


class TestTempleConstraintNoFalsePositive:
    """"不想去寺庙"不能把故宫判成寺庙。"""

    def test_forbidden_city_is_not_a_temple(self):
        terms = expand_terms(["不要寺庙"])
        # 故宫是明清皇宫，不是宗教场所。改前因为扩展表里有单字"宫"被误判。
        assert _hit("故宫博物院", terms) == ""

    def test_palace_museum_name_variants(self):
        terms = expand_terms(["不要寺庙"])
        for name in ("故宫", "故宫博物院", "景山公园", "天坛"):
            assert _hit(name, terms) == "", f"{name} 不该被判为寺庙"

    def test_real_temples_still_caught(self):
        """去掉"宫"不能把真正的寺庙一起放过。

        只断言佛寺与教堂：道教场所没有通用字可用——"观"会命中
        观前街 / 观景台 / 观海园，误伤面比"宫"还大，所以本规则
        本来就不覆盖道观（白云观这类漏判是已知局限，不是回归）。
        """
        terms = expand_terms(["不要寺庙"])
        for name in ("大昭寺", "少林寺", "法源寺", "灵隐寺", "王府井教堂"):
            assert _hit(name, terms), f"{name} 应该被判为寺庙"

    def test_known_tradeoff_yonghegong(self):
        """已知取舍：雍和宫漏判。

        "宫" 既用于藏传佛教寺庙（雍和宫），也用于皇宫（故宫、沈阳故宫）。
        名称级匹配无法区分，只能二选一。按 constraints.py 的原则
        "宁可漏判也不要误伤"（误伤会把合理候选排到最后，漏判只是少抓一个），
        选择漏判雍和宫。这条断言把这个取舍写下来，避免以后被当成 bug 改回去。
        """
        terms = expand_terms(["不要寺庙"])
        assert _hit("雍和宫", terms) == ""

    def test_rule_table_shape(self):
        # 防止有人重新加回单字"宫"或"索道"这类高危同义词
        temple_terms = TERM_EXPANSIONS["寺庙"]
        assert "宫" not in temple_terms
        assert "索道" not in TERM_EXPANSIONS["爬山"]


class TestRoutesDoNotBlockEventLoop:
    """同步 IO 必须走同步路由，交给 FastAPI 的线程池。

    为什么用 inspect 而不是压测：`async def` 路由在事件循环里跑，
    同步路由由 Starlette 丢进线程池——这是框架契约，断言签名即可，
    不受网络抖动影响，测试也就不会 flaky。
    """

    # 这七个函数体里都有同步 httpx 调用（高德 API / CDN 下载）。
    # 注意：router 创建时已带 prefix，route.path 是 "/map/poi" 这种完整形式。
    BLOCKING_PRONE = {
        "/map/poi": map_routes,            # 高德 POI 搜索
        "/map/weather": map_routes,        # 高德天气
        "/map/route": map_routes,          # 高德路径规划
        "/map/health": map_routes,         # 高德健康探测（缓存过期时会真的打上游）
        "/poi/detail/{poi_id}": poi_routes,      # 高德 POI 详情
        "/poi/search": poi_routes,               # 高德关键词搜索
        "/poi/photo/asset/{photo_id}": poi_routes,  # 缓存未命中时会下载 CDN
    }

    @pytest.mark.parametrize("path,module", list(BLOCKING_PRONE.items()))
    def test_route_is_sync(self, path, module):
        endpoint = _endpoints(module).get(path)
        assert endpoint is not None, f"路由 {path} 不存在，测试需同步更新"
        assert not inspect.iscoroutinefunction(endpoint), (
            f"{path} 又被写成 async def 了：函数体里有同步 IO，"
            f"会把事件循环堵住（高德 timeout=30s × 3 次重试 = 最坏 92s）"
        )

    def test_pure_memory_route_stays_async(self):
        """反过来的约束：纯内存查表的路由应保持 async。

        图片查询只查本地 dict（微秒级），走 async 更轻——没有线程切换开销。
        这条断言防的是"为了统一风格把纯内存路由也改成 def"。
        """
        endpoint = _endpoints(poi_routes).get("/poi/photo")
        assert endpoint is not None
        assert inspect.iscoroutinefunction(endpoint)

    def test_trip_plan_routes_stay_async(self):
        """行程路由必须保持 async：它们内部用 asyncio.to_thread / 后台任务，
        改成 def 会让整条生成链路（几十秒）占死一个线程池 worker。
        """
        from app.api.routes import trip as trip_routes

        endpoints = _endpoints(trip_routes)
        for path in ("/trip/plan", "/trip/plan/async", "/trip/plan/{plan_id}/refine"):
            assert inspect.iscoroutinefunction(endpoints[path]), f"{path} 不该改成同步路由"
