"""POI相关API路由"""

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel
from typing import Optional
from ...config import get_settings
from ...planner.amap import AmapPlannerClient
from ...planner.local_photo_table import load_photo_bytes, lookup_photo_id
from ...services.amap_service import get_amap_service

router = APIRouter(prefix="/poi", tags=["POI"])
SEARCH_SOURCE_ROLES = {"food", "scenic", "hotel"}


class POIDetailResponse(BaseModel):
    """POI详情响应"""
    success: bool
    message: str
    data: Optional[dict] = None


@router.get(
    "/detail/{poi_id}",
    response_model=POIDetailResponse,
    summary="获取POI详情",
    description="根据POI ID获取详细信息,包括图片"
)
async def get_poi_detail(poi_id: str):
    """
    获取POI详情

    Args:
        poi_id: POI ID

    Returns:
        POI详情响应
    """
    try:
        amap_service = get_amap_service()

        # 调用高德地图POI详情API
        result = amap_service.get_poi_detail(poi_id)

        return POIDetailResponse(
            success=True,
            message="获取POI详情成功",
            data=result
        )

    except Exception as e:
        print(f"❌ 获取POI详情失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"获取POI详情失败: {str(e)}"
        )


@router.get(
    "/search",
    summary="搜索POI",
    description="根据关键词搜索POI"
)
async def search_poi(
    keywords: str,
    city: str = "北京",
    source_role: str = Query(default="food", description="POI类型: food/scenic/hotel")
):
    """
    搜索POI

    Args:
        keywords: 搜索关键词
        city: 城市名称

    Returns:
        搜索结果
    """
    try:
        role = source_role if source_role in SEARCH_SOURCE_ROLES else "food"
        settings = get_settings()
        amap_key = settings.amap_api_key
        if not amap_key:
            raise ValueError("高德地图API Key未配置")

        amap_client = AmapPlannerClient(amap_key)
        result = amap_client.search_keywords(
            city=city,
            keywords=[keywords],
            source_role=role,
            limit=5,
            require_location=True,
            source_bucket="frontend_search",
        )

        return {
            "success": True,
            "message": "搜索成功",
            "data": result
        }

    except Exception as e:
        print(f"❌ 搜索POI失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"搜索POI失败: {str(e)}"
        )


@router.get(
    "/photo",
    summary="获取景点图片",
    description="按本地图片映射表返回缓存代理地址；未命中返回 null，前端展示渐变占位图"
)
async def get_attraction_photo(name: str, city: str = ""):
    """
    获取景点图片（批次11改造）。

    原实现：对每个景点同步调用两次 Unsplash 搜索 API。
    实测三个问题（详见 app/planner/local_photo_table.py 模块注释）：
    搜索 API 国内不可达（8s 超时）、.env 里 key 为空串、
    async 路由里跑同步 requests 阻塞事件循环 22s/景点，
    15 个景点串行 5 分半，把轮询和餐饮坐标回填全部卡死。

    现在：纯内存查本地映射表（零外网、微秒级），
    命中时返回指向本地磁盘缓存代理的地址。
    """
    photo_id = lookup_photo_id(name, city)

    if not photo_id:
        # 未命中不是失败：前端 getAttractionImage 会用带景点名的渐变卡兜底
        return {
            "success": True,
            "message": "无匹配图片，使用占位图",
            "data": {
                "name": name,
                "photo_url": None,
                "source": "placeholder",
            }
        }

    return {
        "success": True,
        "message": "获取图片成功",
        "data": {
            "name": name,
            # 相对路径：开发态经 vite 代理到后端，生产态同域反代，均无需拼 host
            "photo_url": f"/api/poi/photo/asset/{photo_id}",
            "source": "local_table",
        }
    }


@router.get(
    "/photo/asset/{photo_id}",
    summary="景点图片资源",
    description="磁盘缓存代理：首次从 CDN 拉取并落盘，之后直接读本地文件"
)
def get_photo_asset(photo_id: str):
    """图片资源端点（批次11改造）。

    为什么用 `def` 而不是 `async def`：函数内部有一次真正的网络 IO
    （缓存未命中时下载 CDN）。写成同步路由，FastAPI 会把它丢进
    线程池执行，下载再慢也不阻塞事件循环——原实现正是栽在
    "async 路由里跑同步 requests" 上，22s 一个地把整个后端卡死。
    """
    data = load_photo_bytes(photo_id)
    if data is None:
        # 404 让前端 <img> 触发 @error，退回灰色占位
        raise HTTPException(status_code=404, detail="图片不存在或源站不可达")
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400"})
