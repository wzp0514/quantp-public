"""
仓位管理单元测试 — 测试 PositionManager 的买卖/持仓/盈亏/敞口
"""
import pytest
from datetime import date
from live.execution.position_mgr import PositionManager


@pytest.fixture
def pm():
    return PositionManager(initial_cash=100000.0)


class TestInitialState:
    def test_initial_cash(self, pm):
        assert pm.cash == 100000.0

    def test_no_positions_initially(self, pm):
        assert pm.get_all_positions() == []

    def test_get_position_none(self, pm):
        assert pm.get_position("600519") is None


class TestBuy:
    def test_buy_creates_position(self, pm):
        pm.update_position("600519", "buy", 100, 50.0)
        pos = pm.get_position("600519")
        assert pos["size"] == 100
        assert pos["avg_cost"] == 50.0

    def test_buy_reduces_cash(self, pm):
        pm.update_position("600519", "buy", 100, 50.0)
        assert pm.cash == 100000.0 - 5000.0

    def test_buy_multiple_averages_cost(self, pm):
        pm.update_position("600519", "buy", 100, 50.0)
        pm.update_position("600519", "buy", 100, 60.0)
        pos = pm.get_position("600519")
        assert pos["size"] == 200
        assert pos["avg_cost"] == 55.0

    def test_buy_trade_history(self, pm):
        pm.update_position("600519", "buy", 100, 50.0)
        assert len(pm.trade_history) == 1
        assert pm.trade_history[0]["side"] == "buy"


class TestSell:
    def test_sell_reduces_position(self, pm):
        pm.update_position("600519", "buy", 100, 50.0)
        pm.update_position("600519", "sell", 50, 60.0)
        pos = pm.get_position("600519")
        assert pos["size"] == 50

    def test_sell_increases_cash(self, pm):
        pm.update_position("600519", "buy", 100, 50.0)
        pm.update_position("600519", "sell", 50, 60.0)
        assert pm.cash == 100000.0 - 5000.0 + 3000.0

    def test_sell_records_realized_pnl(self, pm):
        pm.update_position("600519", "buy", 100, 50.0)
        pm.update_position("600519", "sell", 50, 60.0)
        pos = pm.get_position("600519")
        expected_pnl = (60.0 - 50.0) * 50
        assert pos["realized_pnl"] == pytest.approx(expected_pnl)

    def test_sell_more_than_holding_clamped(self, pm):
        pm.update_position("600519", "buy", 100, 50.0)
        pm.update_position("600519", "sell", 200, 60.0)
        pos = pm.get_position("600519")
        assert pos["size"] == 0

    def test_sell_all_closes_position(self, pm):
        pm.update_position("600519", "buy", 100, 50.0)
        pm.update_position("600519", "sell", 100, 60.0)
        pos = pm.get_position("600519")
        assert pos["size"] == 0


class TestPnL:
    def test_unrealized_pnl_profit(self, pm):
        pm.update_position("600519", "buy", 100, 50.0)
        pnl = pm.get_unrealized_pnl("600519", 55.0)
        assert pnl == 500.0

    def test_unrealized_pnl_loss(self, pm):
        pm.update_position("600519", "buy", 100, 50.0)
        pnl = pm.get_unrealized_pnl("600519", 45.0)
        assert pnl == -500.0

    def test_unrealized_pnl_no_position(self, pm):
        assert pm.get_unrealized_pnl("600519", 50.0) == 0.0

    def test_total_unrealized_pnl(self, pm):
        pm.update_position("600519", "buy", 100, 50.0)
        pm.update_position("000001", "buy", 200, 30.0)
        total = pm.get_total_unrealized_pnl({"600519": 55.0, "000001": 33.0})
        assert total == 500.0 + 600.0

    def test_total_value(self, pm):
        pm.update_position("600519", "buy", 100, 50.0)
        value = pm.get_total_value({"600519": 55.0})
        assert value == pm.cash + 100 * 55.0

    def test_total_return(self, pm):
        pm.update_position("600519", "buy", 100, 50.0)
        ret = pm.get_total_return({"600519": 55.0})
        expected_total = pm.cash + 100 * 55.0
        assert ret == pytest.approx(expected_total / 100000.0 - 1)


class TestExposure:
    def test_exposure_no_positions(self, pm):
        assert pm.get_exposure({}) == 0.0

    def test_exposure_with_positions(self, pm):
        pm.update_position("600519", "buy", 100, 50.0)
        exposure = pm.get_exposure({"600519": 55.0})
        total = pm.get_total_value({"600519": 55.0})
        expected = (100 * 55.0) / total
        assert exposure == pytest.approx(expected)


class TestSummary:
    def test_summary_no_positions(self, pm):
        assert "无持仓" in pm.get_summary({})

    def test_summary_has_columns(self, pm):
        pm.update_position("600519", "buy", 100, 50.0)
        text = pm.get_summary({"600519": 55.0})
        assert "600519" in text
        assert "现金" in text


class TestGetAllPositions:
    def test_only_nonzero(self, pm):
        pm.update_position("600519", "buy", 100, 50.0)
        pm.update_position("600519", "sell", 100, 55.0)
        positions = pm.get_all_positions()
        assert len(positions) == 0
