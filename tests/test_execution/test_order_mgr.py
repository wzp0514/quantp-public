"""
订单管理单元测试 — 测试 OrderManager 的完整生命周期
"""
import pytest
from datetime import date
from live.execution.order_mgr import (
    OrderManager, BUY, SELL,
    STATUS_CREATED, STATUS_SUBMITTED, STATUS_FILLED,
    STATUS_PARTIAL, STATUS_CANCELLED, STATUS_REJECTED,
)


@pytest.fixture
def om():
    return OrderManager()


class TestCreateOrder:
    def test_create_buy_order(self, om):
        order = om.create_market_order("600519", BUY, 100)
        assert order["id"] == 1
        assert order["side"] == BUY
        assert order["size"] == 100
        assert order["status"] == STATUS_CREATED

    def test_create_sell_order(self, om):
        order = om.create_market_order("600519", SELL, 50)
        assert order["side"] == SELL

    def test_create_with_date(self, om):
        d = date(2025, 1, 15)
        order = om.create_market_order("600519", BUY, 100, order_date=d)
        assert order["created_at"] == d

    def test_invalid_side_raises(self, om):
        with pytest.raises(ValueError, match="side"):
            om.create_market_order("600519", "hold", 100)

    def test_zero_size_raises(self, om):
        with pytest.raises(ValueError, match="size"):
            om.create_market_order("600519", BUY, 0)

    def test_negative_size_raises(self, om):
        with pytest.raises(ValueError, match="size"):
            om.create_market_order("600519", BUY, -10)

    def test_id_auto_increments(self, om):
        o1 = om.create_market_order("600519", BUY, 100)
        o2 = om.create_market_order("000001", SELL, 50)
        assert o2["id"] == o1["id"] + 1


class TestSubmit:
    def test_submit_changes_status(self, om):
        order = om.create_market_order("600519", BUY, 100)
        om.submit(order["id"])
        assert order["status"] == STATUS_SUBMITTED


class TestFill:
    def test_fill_full(self, om):
        order = om.create_market_order("600519", BUY, 100)
        om.submit(order["id"])
        om.fill(order["id"], 1500.0, date(2025, 1, 15))
        assert order["status"] == STATUS_FILLED
        assert order["price"] == 1500.0
        assert order["filled_size"] == 100

    def test_fill_sets_filled_at(self, om):
        order = om.create_market_order("600519", BUY, 100)
        om.submit(order["id"])
        d = date(2025, 1, 15)
        om.fill(order["id"], 1500.0, d)
        assert order["filled_at"] == d


class TestPartialFill:
    def test_partial_fill_status(self, om):
        order = om.create_market_order("600519", BUY, 100)
        om.submit(order["id"])
        om.partial_fill(order["id"], 60, 1500.0)
        assert order["status"] == STATUS_PARTIAL
        assert order["filled_size"] == 60

    def test_partial_then_full(self, om):
        order = om.create_market_order("600519", BUY, 100)
        om.submit(order["id"])
        om.partial_fill(order["id"], 40, 1500.0)
        om.partial_fill(order["id"], 60, 1510.0)
        assert order["status"] == STATUS_FILLED
        assert order["filled_size"] == 100


class TestCancel:
    def test_cancel_pending(self, om):
        order = om.create_market_order("600519", BUY, 100)
        om.cancel(order["id"])
        assert order["status"] == STATUS_CANCELLED

    def test_cancel_already_filled_no_change(self, om):
        order = om.create_market_order("600519", BUY, 100)
        om.submit(order["id"])
        om.fill(order["id"], 1500.0)
        om.cancel(order["id"])
        assert order["status"] == STATUS_FILLED


class TestReject:
    def test_reject_with_reason(self, om):
        order = om.create_market_order("600519", BUY, 100)
        om.reject(order["id"], "资金不足")
        assert order["status"] == STATUS_REJECTED
        assert order["note"] == "资金不足"


class TestGetOrders:
    def test_get_all(self, om):
        om.create_market_order("600519", BUY, 100)
        om.create_market_order("000001", SELL, 50)
        assert len(om.get_orders()) == 2

    def test_get_by_status(self, om):
        om.create_market_order("600519", BUY, 100)
        filled_order = om.create_market_order("000001", SELL, 50)
        om.submit(filled_order["id"])
        om.fill(filled_order["id"], 10.0)
        assert len(om.get_orders(status=STATUS_FILLED)) == 1
        assert len(om.get_orders(status=STATUS_CREATED)) == 1

    def test_get_by_symbol(self, om):
        om.create_market_order("600519", BUY, 100)
        om.create_market_order("000001", SELL, 50)
        assert len(om.get_orders(symbol="600519")) == 1

    def test_get_order_by_id(self, om):
        om.create_market_order("600519", BUY, 100)
        order = om.get_order(1)
        assert order is not None
        assert order["symbol"] == "600519"

    def test_get_order_nonexistent(self, om):
        assert om.get_order(999) is None


class TestGetPending:
    def test_pending_includes_created_and_submitted(self, om):
        o1 = om.create_market_order("600519", BUY, 100)
        o2 = om.create_market_order("000001", SELL, 50)
        om.submit(o2["id"])
        om.fill(o2["id"], 10.0)
        pending = om.get_pending()
        assert len(pending) == 1
        assert pending[0]["id"] == o1["id"]
