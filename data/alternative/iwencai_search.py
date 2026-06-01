"""
问财（iwencai）语义搜索 — 骨架（需登录API Key）。

跨主题选股，自然语言→股票筛选。
当前为骨架占位，需配置 iwencai API Key 后启用。

用法
--------
>>> from data.alternative.iwencai_search import IwenCai
>>> iwc = IwenCai()
>>> df = iwc.search("市盈率<20 且 ROE>15% 且市值<500亿")  # 需API Key
"""

import logging
import os
from typing import Optional

import pandas as pd

from config.log import get_logger

logger = get_logger("iwencai")


class IwenCai:
    """
    问财语义搜索接口（骨架）。

    待实现：
    1. 用户登录/Token 获取
    2. 自然语言→结构化查询
    3. 搜索结果解析
    4. 结果缓存（降低API调用频率）
    """

    def __init__(self):
        self._api_key = os.environ.get("IWENCAI_API_KEY", "")
        self._available = bool(self._api_key)

    def search(self, query: str, limit: int = 50) -> pd.DataFrame:
        """
        语义搜索选股。

        参数
        ----------
        query : str
            自然语言查询，如 "市盈率<20 且 ROE>15%"
        limit : int
            最大返回数量

        返回
        -------
        DataFrame
        """
        if not self._available:
            logger.info(f"iwencai 搜索不可用（未配置 IWENCAI_API_KEY）。查询={query}")
            return pd.DataFrame()

        logger.info(f"iwencai 搜索: {query}")
        # TODO: 实现 iwencai API 调用
        return pd.DataFrame()

    def is_available(self) -> bool:
        return self._available
