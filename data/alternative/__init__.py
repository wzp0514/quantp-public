"""
另类数据 — 新闻情绪、地缘政治、雪球、财报等多源数据融合。

用法
--------
>>> from data.alternative import AlternativeData
>>> ad = AlternativeData()
>>> result = ad.full_scan()
"""

from data.alternative.pipeline import (
    AlternativeData,
    should_trade_today,
    get_position_multiplier,
)

__all__ = ["AlternativeData", "should_trade_today", "get_position_multiplier"]
