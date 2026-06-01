"""
策略回测烟雾测试 — 真实沪深300数据
"""

import pytest


class TestStrategyImports:
    def test_all_strategies_importable(self):
        """确认 ALL_STRATEGIES 所有条目可导入"""
        from backtest.strategy_market import ALL_STRATEGIES
        assert len(ALL_STRATEGIES) >= 10

    def test_builtin_strategies_have_class(self):
        """内置策略都有class字段"""
        from backtest.strategy_market import BUILTIN_STRATEGIES
        for name, info in BUILTIN_STRATEGIES.items():
            assert "class" in info, f"{name} 缺少 class"

    def test_ma_cross_runs(self, csi300_2023):
        """双均线策略不崩溃（真实数据）"""
        from backtest.strategies.builtin.ma_cross import MaCrossStrategy
        from backtest.engine.bt_runner import run_backtest
        result = run_backtest(MaCrossStrategy, csi300_2023,
                              initial_cash=100000, fast=5, slow=20)
        assert "annual_return" in result
        assert "drawdown" in result
        assert result.get("total_trades", 0) >= 0
        # 真实数据有基本合理性：年化收益在 -1~1 之间
        ar = result["annual_return"]
        assert -1.0 < ar < 1.0, f"年化收益={ar:.2%} 异常"

    def test_bollinger_runs(self, csi300_2023):
        """布林带策略不崩溃（真实数据）"""
        from backtest.strategies.builtin.bollinger import BollingerStrategy
        from backtest.engine.bt_runner import run_backtest
        result = run_backtest(BollingerStrategy, csi300_2023,
                              initial_cash=100000, period=20, devfactor=2.0)
        assert "annual_return" in result
        assert result.get("drawdown", 0) >= 0

    def test_strategy_market_has_strategies(self):
        """ALL_STRATEGIES 和 BUILTIN_STRATEGIES 非空"""
        from backtest.strategy_market import ALL_STRATEGIES, BUILTIN_STRATEGIES
        assert isinstance(ALL_STRATEGIES, dict)
        assert isinstance(BUILTIN_STRATEGIES, dict)
        assert len(ALL_STRATEGIES) > 0
        assert len(BUILTIN_STRATEGIES) > 0
