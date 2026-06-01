"""
同花顺热点题材 — 骨架（需反爬处理）。

当日强势股题材归因，作为短线信号层。
当前为骨架占位，数据源接入需处理同花顺反爬机制。

用法
--------
>>> from data.alternative.ths_hotspot import THSHotspot
>>> ths = THSHotspot()
>>> df = ths.fetch_hotspots()  # 需完成反爬实现
"""

import logging
from datetime import datetime

import pandas as pd

from config.log import get_logger

logger = get_logger("ths_hotspot")


class THSHotspot:
    """
    同花顺热点数据接口（骨架）。

    待实现：
    1. 反爬策略（动态 User-Agent/请求间隔/IP 代理池）
    2. 热点题材列表解析
    3. 关联个股提取
    4. 题材持续天数/强度追踪
    """

    def __init__(self):
        self._last_fetch = None

    def fetch_hotspots(self, date: str = "") -> pd.DataFrame:
        """
        获取当日热点题材。

        返回
        -------
        DataFrame: topic/strength/stock_count/leading_stock/description
        """
        if not date:
            date = datetime.now().strftime("%Y%m%d")
        logger.info(f"同花顺热点数据接口（骨架），当前不可用。日期={date}")
        return pd.DataFrame(columns=["topic", "strength", "stock_count", "leading_stock", "description"])

    def fetch_topic_stocks(self, topic: str) -> list[dict]:
        """获取题材关联个股"""
        logger.info(f"题材关联个股接口（骨架），当前不可用。topic={topic}")
        return []

    def is_available(self) -> bool:
        """检查数据源是否可用"""
        return False  # 骨架，始终返回不可用
