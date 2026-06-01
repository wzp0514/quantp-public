"""数据获取测试"""

import pytest


class TestDataFetch:
    def test_fallback_import(self):
        """确认降级链模块可导入"""
        from data.fetchers.fallback import check_dependencies
        deps = check_dependencies()
        assert isinstance(deps, dict)
        assert "akshare" in deps or "tushare" in deps or "baostock" in deps

    def test_cache_factor_path(self):
        """确认缓存路径函数"""
        from backtest.analysis.factor_miner import factor_cache_path
        path = factor_cache_path("沪深300")
        assert str(path).endswith(".parquet")
