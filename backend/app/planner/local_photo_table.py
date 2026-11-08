"""本地图片映射表 + 磁盘缓存代理（批次11）。

为什么要做这一层（面试可直接讲）：

原实现每个景点调两次 Unsplash 搜索 API（api.unsplash.com），实测三个问题：
1. **该 API 在国内网络环境不可达**（curl 直测 8 秒超时，状态码 000），
   且 .env 里 UNSPLASH_ACCESS_KEY 本来就是空串，即使可达也会 401；
2. 每次失败要空转 2 × 10s 超时，实测 `/api/poi/photo?name=故宫` 耗时 **22 秒**、
   最终 photo_url 还是 null —— 用户什么都没等到；
3. 最致命的：路由是 `async def` 却调用同步 `requests.get`，
   **整个 FastAPI 事件循环被阻塞**。结果页 15 个景点并发请求图片，
   实际串行排队 15 × 22s ≈ 5 分半，期间餐饮坐标回填、任务轮询全部卡死，
   用户看到的就是"很多都加载不出来"。

改造方案：
1. **图片不搜索、查本地表**：人工筛选并逐张核验内容的 Unsplash CDN 直链
   （images.unsplash.com 图片 CDN 与搜索 API 是不同域名，CDN 国内可达），
   按"景点关键词 → 城市 → 无"三级匹配，未命中返回 None，
   前端退回渐变占位图（景点名文字卡），视觉不空缺。
2. **磁盘缓存代理**：浏览器永远只连 localhost 的
   `GET /api/poi/photo/asset/{photo_id}`；后端首次命中时下载并缓存到
   `.cache/photos/`，之后全部读本地磁盘。CDN 偶发抖动（实测约 1/3 请求
   首次超时）由服务端重试吸收，对前端 100% 稳定。
3. **代理端点用 `def`（同步）路由**：FastAPI 会把它丢进线程池执行，
   下载再慢也不碰事件循环——这正是原 bug 的镜像修复。
"""

import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

import requests

# === 图片池：每张都经过人工内容核验（下载后逐张查看确认） ===
# key 为 Unsplash CDN 的 photo id；value 为内容描述，防止后续维护者
# 拿一张"停车场跑车"图配到西湖上（候选池里真的出现过这种图，已剔除）。
SHANGHAI_BUND = "1538428494232-9c0d8a3ab403"      # 上海外滩夜景
BEACH_SUNSET = "1507525428034-b723cf961d3e"       # 海滩日落
ALPINE_LAKE = "1501785888041-af3ef285b470"        # 高山湖泊+小船
VALLEY_RIVER = "1506744038136-46273834b3fb"       # 山谷河流森林
FOREST_PATH = "1441974231531-c6227db76b6e"        # 森林步道
MOUNTAIN_DAWN = "1470071459604-3b5ec3a7fe05"      # 云雾山峦日出
CITY_SKYLINE = "1477959858617-67f85cf4f1df"       # 湖畔都市天际线
CITY_MORNING = "1480714378408-67cf0d13bc1b"       # 清晨都市天际线
RIVER_CITY = "1513635269975-59663e0ac1ad"         # 河畔城市塔桥
NEON_STREET = "1540959733332-eab4deabeeaf"        # 霓虹商业步行街

# 首页 hero 背景图（Home.vue 原来直连 CDN，一并收进代理）
HOME_HERO = "1500530855697-b586d89ba3ee"

# 所有允许通过代理访问的图片 id。代理端点只认表内 id，
# 防止被当成任意图片的开放代理（open proxy）用。
KNOWN_PHOTO_IDS = frozenset(
    {
        SHANGHAI_BUND, BEACH_SUNSET, ALPINE_LAKE, VALLEY_RIVER,
        FOREST_PATH, MOUNTAIN_DAWN, CITY_SKYLINE, CITY_MORNING,
        RIVER_CITY, NEON_STREET, HOME_HERO,
    }
)

PHOTO_CDN_URL = "https://images.unsplash.com/photo-{photo_id}?w=800&q=75&auto=format&fit=crop"

# === 景点关键词 → 图片 ===
# 匹配规则：景点名包含关键词（"故宫博物院" 命中 "故宫"）。
# 只收录对图片内容有把握的词条，宁可不给图也不给错图。
PHOTO_ID_BY_KEYWORD: Dict[str, str] = {
    # 上海
    "外滩": SHANGHAI_BUND,
    "东方明珠": SHANGHAI_BUND,
    "陆家嘴": SHANGHAI_BUND,
    "黄浦江": SHANGHAI_BUND,
    "南京路": SHANGHAI_BUND,
    # 海滨
    "亚龙湾": BEACH_SUNSET,
    "鼓浪屿": BEACH_SUNSET,
    "海滨": BEACH_SUNSET,
    "沙滩": BEACH_SUNSET,
    "海滩": BEACH_SUNSET,
    "海边": BEACH_SUNSET,
    "海滩公园": BEACH_SUNSET,
    # 湖泊
    "西湖": ALPINE_LAKE,
    "泸沽湖": ALPINE_LAKE,
    "洱海": ALPINE_LAKE,
    "天池": ALPINE_LAKE,
    "千岛湖": ALPINE_LAKE,
    "青海湖": ALPINE_LAKE,
    # 山岳
    "黄山": MOUNTAIN_DAWN,
    "泰山": MOUNTAIN_DAWN,
    "华山": MOUNTAIN_DAWN,
    "峨眉山": MOUNTAIN_DAWN,
    "云海": MOUNTAIN_DAWN,
    "日出": MOUNTAIN_DAWN,
    # 森林/峡谷
    "张家界": VALLEY_RIVER,
    "九寨沟": VALLEY_RIVER,
    "武陵源": VALLEY_RIVER,
    "森林公园": FOREST_PATH,
    "原始森林": FOREST_PATH,
    "莫干山": FOREST_PATH,
    # 步行街/夜市
    "洪崖洞": NEON_STREET,
    "宽窄巷子": NEON_STREET,
    "回民街": NEON_STREET,
    "步行街": NEON_STREET,
    "夜市": NEON_STREET,
}

# === 城市 → 图片（关键词未命中时的城市级兜底） ===
PHOTO_ID_BY_CITY: Dict[str, str] = {
    "上海": SHANGHAI_BUND,
    "三亚": BEACH_SUNSET,
    "厦门": BEACH_SUNSET,
    "青岛": BEACH_SUNSET,
    "大连": BEACH_SUNSET,
    "珠海": BEACH_SUNSET,
    "北海": BEACH_SUNSET,
    "秦皇岛": BEACH_SUNSET,
    "重庆": NEON_STREET,
    "长沙": NEON_STREET,
    "成都": NEON_STREET,
    "西安": NEON_STREET,
    "张家界": VALLEY_RIVER,
    "黄山": MOUNTAIN_DAWN,
    "桂林": VALLEY_RIVER,
    "杭州": ALPINE_LAKE,
    "苏州": RIVER_CITY,
    "无锡": RIVER_CITY,
    "绍兴": RIVER_CITY,
    "南京": CITY_SKYLINE,
    "深圳": CITY_SKYLINE,
    "广州": CITY_SKYLINE,
    "北京": CITY_MORNING,
    "天津": CITY_SKYLINE,
    "武汉": CITY_SKYLINE,
    "郑州": CITY_MORNING,
    "洛阳": CITY_MORNING,
}

# 关键词匹配时按长度优先："亚龙湾热带天堂森林公园" 应命中更具体的词
_KEYWORDS_BY_LENGTH: List[str] = sorted(PHOTO_ID_BY_KEYWORD, key=len, reverse=True)

# 归一化城市名：去掉"市"后缀再查表
_CITY_ALIASES = ("市", "地区", "自治州")


def lookup_photo_id(name: str, city: str = "") -> Optional[str]:
    """按 景点关键词 → 城市 两级查表，命中返回图片 id，未命中返回 None。

    未命中不是失败：前端有带景点名的渐变占位图兜底，
    视觉上是设计过的卡片而不是裂图。
    """
    clean_name = (name or "").strip()
    clean_city = (city or "").strip()
    for suffix in _CITY_ALIASES:
        if clean_city.endswith(suffix):
            clean_city = clean_city[: -len(suffix)]
    for keyword in _KEYWORDS_BY_LENGTH:
        if keyword and keyword in clean_name:
            return PHOTO_ID_BY_KEYWORD[keyword]
    return PHOTO_ID_BY_CITY.get(clean_city)


def photo_cdn_url(photo_id: str) -> str:
    """拼 CDN 直链（仅代理端点内部使用）。"""
    return PHOTO_CDN_URL.format(photo_id=photo_id)


# === 磁盘缓存 ===
# backend/.cache/photos/。.gitignore 应包含 .cache/。
CACHE_DIR = Path(
    __import__("os").getenv(
        "PHOTO_CACHE_DIR",
        str(Path(__file__).resolve().parents[2] / ".cache" / "photos"),
    )
)
_DOWNLOAD_TIMEOUT = (10, 20)  # (连接超时, 读超时)。CDN 的 TLS 握手实测可到 7s+，
                              # 连接超时给 5s 会让重试全部死在握手阶段（实测复现）
_DOWNLOAD_RETRIES = 3
# 同一张图并发请求只下载一次：进程内锁 + "正在下载"标记
_download_locks: Dict[str, threading.Lock] = {}
_download_locks_guard = threading.Lock()
_photo_id_pattern = re.compile(r"^[0-9a-zA-Z\-]{10,64}$")


def is_valid_photo_id(photo_id: str) -> bool:
    """只放行表内、且格式合法的图片 id（防路径穿越 / 开放代理）。"""
    return bool(_photo_id_pattern.match(photo_id or "")) and photo_id in KNOWN_PHOTO_IDS


def cached_photo_path(photo_id: str) -> Path:
    return CACHE_DIR / f"{photo_id}.jpg"


def load_photo_bytes(photo_id: str) -> Optional[bytes]:
    """读缓存的图片字节；缓存未命中则下载一次并落盘。失败返回 None。

    并发安全：同一 photo_id 用独立的锁串行化下载，
    10 个景点共用一张城市图时也只会发起 1 次 CDN 请求。
    """
    if not is_valid_photo_id(photo_id):
        return None

    path = cached_photo_path(photo_id)
    if path.is_file() and path.stat().st_size > 0:
        try:
            return path.read_bytes()
        except OSError:
            pass  # 读失败就走重新下载

    with _download_locks_guard:
        lock = _download_locks.setdefault(photo_id, threading.Lock())
    with lock:
        # 双重检查：等锁期间可能已被别的线程下载完成
        if path.is_file() and path.stat().st_size > 0:
            try:
                return path.read_bytes()
            except OSError:
                pass
        for _ in range(_DOWNLOAD_RETRIES):
            try:
                resp = requests.get(photo_cdn_url(photo_id), timeout=_DOWNLOAD_TIMEOUT)
                resp.raise_for_status()
                data = resp.content
                if len(data) < 1000:  # CDN 报错也常返回 200 + 短 JSON
                    continue
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(data)
                tmp.replace(path)  # 原子落盘，读侧永远看不到半张图
                return data
            except (requests.RequestException, OSError):
                continue
    return None
