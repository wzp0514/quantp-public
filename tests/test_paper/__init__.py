"""
纸上交易引擎烟雾测试 — 测试 PaperTrader 初始化 + 结果结构
"""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    price = 3500 * np.cumprod(1 + np.random.randn(n) * 0.012 + 0.0003)
    return pd.DataFrame({
        "date": dates,
        "open": price * 0.998,
        "high": price * 1.012,
        "low": price * 0.988,
        "close": price,
        "volume": np.random.randint(100, 500, n).astype(float) * 1e6,
    })


class TestPaperTraderInit:
    def test_init_with_real_strategy(self, sample_data):
        from live.paper_trader import PaperTrader
        from backtest.strategies.builtin.ma_cross import MaCrossStrategy
        pt = PaperTrader(MaCrossStrategy, sample_data, initial_cash=100000,
                         fast=5, slow=20)
        assert pt.initial_cash == 100000
        assert len(pt.data) == len(sample_data)
        assert pt.position_mgr.cash == 100000

    def test_init_with_guard_enabled(self, sample_data):
        from live.paper_trader import PaperTrader
        from backtest.strategies.builtin.ma_cross import MaCrossStrategy
        pt = PaperTrader(MaCrossStrategy, sample_data, enable_guard=True)
        assert pt.guard is not None

    def test_init_with_guard_disabled(self, sample_data):
        from live.paper_trader import PaperTrader
        from backtest.strategies.builtin.ma_cross import MaCrossStrategy
        pt = PaperTrader(MaCrossStrategy, sample_data, enable_guard=False)
        assert pt.guard is None

    def test_init_empty_data(self):
        from live.paper_trader import PaperTrader
        from backtest.strategies.builtin.ma_cross import MaCrossStrategy
        pt = PaperTrader(MaCrossStrategy, pd.DataFrame())
        assert pt.data.empty


class TestPaperTraderRun:
    def test_run_returns_dict_structure(self, sample_data):
        from live.paper_trader import PaperTrader
        from backtest.strategies.builtin.ma_cross import MaCrossStrategy
        pt = PaperTrader(MaCrossStrategy, sample_data, initial_cash=100000,
                         fast=5, slow=20, enable_guard=False)
        result = pt.run()
        assert isinstance(result, dict)
        assert "strategy" in result
        assert "total_return" in result
        assert "signals" in result
        assert "daily_log" in result
        assert "orders" in result
        assert "warnings" in result
        assert "summary" in result

    def test_run_builds_daily_log(self, sample_data):
        from live.paper_trader import PaperTrader
        from backtest.strategies.builtin.ma_cross import MaCrossStrategy
        pt = PaperTrader(MaCrossStrategy, sample_data, initial_cash=100000,
                         fast=5, slow=20, enable_guard=False)
        result = pt.run()
        assert len(result["daily_log"]) > 0

    def test_run_with_bollinger(self, sample_data):
        from live.paper_trader import PaperTrader
        from backtest.strategies.builtin.bollinger import BollingerStrategy
        pt = PaperTrader(BollingerStrategy, sample_data, initial_cash=100000,
                         period=20, dev=2.0, enable_guard=False)
        result = pt.run()
        assert isinstance(result, dict)
        assert "total_return" in result

    def test_run_return_within_range(self, sample_data):
        from live.paper_trader import PaperTrader
        from backtest.strategies.builtin.ma_cross import MaCrossStrategy
        pt = PaperTrader(MaCrossStrategy, sample_data, initial_cash=100000,
                         fast=5, slow=20, enable_guard=False)
        result = pt.run()
        tr = result["total_return"]
        assert -1.0 <= tr <= 10.0, f"total_return {tr} out of reasonable range"


def test_run_paper_trading_convenience(sample_data):
    from live.paper_trader import run_paper_trading
    from backtest.strategies.builtin.ma_cross import MaCrossStrategy
    result = run_paper_trading(MaCrossStrategy, sample_data, initial_cash=100000,
                               fast=5, slow=20)
    assert isinstance(result, dict)
    assert "total_return" in result
