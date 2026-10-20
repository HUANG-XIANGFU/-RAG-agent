"""Agent 工具：查空气质量、查当前时间。

空气质量使用 Open-Meteo Air Quality API（免费、无需 API Key），
通过内置的城市经纬度字典把城市名转换为坐标后查询。
时间工具使用 Python 标准库 zoneinfo，按 IANA 时区返回当前时间。
"""

from datetime import datetime
from typing import Optional

import requests
from langchain_core.tools import tool

# 常见城市的经纬度（ lat, lon ），覆盖国内外主要城市
CITY_COORDS: dict[str, tuple[float, float]] = {
    "北京": (39.9042, 116.4074),
    "上海": (31.2304, 121.4737),
    "广州": (23.1291, 113.2644),
    "深圳": (22.5431, 114.0579),
    "杭州": (30.2741, 120.1551),
    "成都": (30.5728, 104.0668),
    "武汉": (30.5928, 114.3055),
    "西安": (34.3416, 108.9398),
    "南京": (32.0603, 118.7969),
    "重庆": (29.5630, 106.5516),
    "天津": (39.3434, 117.3616),
    "苏州": (31.2989, 120.5853),
    "长沙": (28.2282, 112.9388),
    "青岛": (36.0671, 120.3829),
    "郑州": (34.7466, 113.6253),
    "厦门": (24.4798, 118.0894),
    "哈尔滨": (45.8038, 126.5340),
    "沈阳": (41.8057, 123.4315),
    "大连": (38.9140, 121.6143),
    "昆明": (24.8801, 102.8329),
    "香港": (22.3193, 114.1694),
    "台北": (25.0330, 121.5654),
    "东京": (35.6762, 139.6503),
    "首尔": (37.5665, 126.9780),
    "新加坡": (1.3521, 103.8198),
    "纽约": (40.7128, -74.0060),
    "伦敦": (51.5072, -0.1276),
    "巴黎": (48.8566, 2.3522),
    "柏林": (52.5200, 13.4050),
}


@tool
def get_air_quality(city: str) -> str:
    """查询指定城市的实时空气质量（PM2.5、PM10、美国 AQI）。

    仅当用户明确询问某城市的空气质量/污染情况时调用；科研文献问答、
    概述总结类问题一律不要调用本工具。

    参数:
        city: 城市名称，支持中文或英文，例如 "北京"、"上海"、"东京"。
    返回:
        该城市当前的空气质量信息字符串。
    """
    coords = _resolve_city(city)
    if coords is None:
        return f"暂未收录城市「{city}」的坐标，无法查询空气质量。"
    lat, lon = coords
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "pm10,pm2_5,us_aqi",
        "timezone": "auto",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("current", {})
        pm25 = data.get("pm2_5")
        pm10 = data.get("pm10")
        us_aqi = data.get("us_aqi")
        if pm25 is None and pm10 is None and us_aqi is None:
            return f"已查询到「{city}」，但接口未返回空气质量数据。"
        return (
            f"城市「{city}」（{lat}, {lon}）实时空气质量："
            f"PM2.5={pm25} μg/m³，PM10={pm10} μg/m³，美国 AQI={us_aqi}。"
        )
    except Exception as e:
        return f"查询「{city}」空气质量失败：{e}"


@tool
def get_current_time(timezone: Optional[str] = None) -> str:
    """查询当前时间。

    仅当用户明确询问当前时间/日期（或需按当前日期推算）时调用；
    科研文献问答等其它问题一律不要调用本工具。

    参数:
        timezone: 可选的 IANA 时区名，例如 "Asia/Shanghai"、"UTC"、"America/New_York"。
                 不传时默认使用系统本地时区。
    返回:
        格式化的当前时间字符串，包含时区信息。
    """
    try:
        from zoneinfo import ZoneInfo

        if timezone:
            tz = ZoneInfo(timezone)
            now = datetime.now(tz)
            return f"当前时间（{timezone}）：{now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    except Exception as e:
        # 时区无效或 zoneinfo 不可用，回退到本地时间
        now = datetime.now()
        return f"时区「{timezone}」无效（{e}），本地时间：{now.strftime('%Y-%m-%d %H:%M:%S')}"
    now = datetime.now()
    return f"当前本地时间：{now.strftime('%Y-%m-%d %H:%M:%S')}"


def _resolve_city(city: str) -> Optional[tuple[float, float]]:
    """根据城市名解析经纬度，支持中英文模糊匹配。"""
    if city in CITY_COORDS:
        return CITY_COORDS[city]
    # 简单的去后缀匹配，例如 "北京市" -> "北京"
    for key, val in CITY_COORDS.items():
        if city.startswith(key) or key.startswith(city):
            return val
    return None


# 暴露给 agent 的工具列表
TOOLS = [get_air_quality, get_current_time]
