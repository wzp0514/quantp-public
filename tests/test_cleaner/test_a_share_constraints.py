"""
A股约束单元测试 — 涨跌停/T+1/ST/多品种
"""
import pytest
from datetime import date
from data.cleaner.a_share_constraints import (
    get_price_limit_pct, is_st_stock, limit_up_price, limit_down_price,
    AShareConstraints,
)


class TestPriceLimitPct:
    def test_main_board(self):
        assert get_price_limit_pct("600519") == 0.10

    def test_shenzhen_main(self):
        assert get_price_limit_pct("000001") == 0.10

    def test_gem_300(self):
        assert get_price_limit_pct("300750") == 0.20

    def test_gem_301(self):
        assert get_price_limit_pct("301000") == 0.20

    def test_star_market(self):
        assert get_price_limit_pct("688001") == 0.20

    def test_beijing(self):
        assert get_price_limit_pct("800001") == 0.30


class TestIsST:
    def test_st_detected(self):
        assert is_st_stock("ST东航") is True

    def test_star_st_detected(self):
        assert is_st_stock("*ST康得") is True

    def test_normal_not_st(self):
        assert is_st_stock("600519") is False


class TestLimitPrices:
    def test_limit_up_10pct(self):
        assert limit_up_price(100.0, 0.10) == 110.0

    def test_limit_down_10pct(self):
        assert limit_down_price(100.0, 0.10) == 90.0

    def test_limit_up_20pct(self):
        assert limit_up_price(100.0, 0.20) == 120.0


class TestCanBuy:
    def test_normal_buy_ok(self):
        asc = AShareConstraints()
        ok, msg = asc.can_buy("600519", price=105.0, prev_close=100.0)
        assert ok is True

    def test_limit_up_blocked(self):
        asc = AShareConstraints()
        ok, msg = asc.can_buy("600519", price=110.0, prev_close=100.0)
        assert ok is False
        assert "涨停" in msg

    def test_near_limit_up_blocked(self):
        asc = AShareConstraints()
        ok, msg = asc.can_buy("600519", price=109.5, prev_close=100.0)
        assert ok is False

    def test_st_blocked_by_default(self):
        asc = AShareConstraints()
        ok, msg = asc.can_buy("ST东航", price=5.0, prev_close=5.0)
        assert ok is False
        assert "ST" in msg

    def test_st_allowed_when_configured(self):
        asc = AShareConstraints(allow_st=True)
        ok, msg = asc.can_buy("ST东航", price=5.0, prev_close=5.0)
        assert ok is True

    def test_etf_main_board_limit(self):
        """主板ETF(510300开头)是10%涨跌停，4.0在3.80允许范围内"""
        asc = AShareConstraints(asset_type="etf")
        ok, msg = asc.can_buy("510300", price=4.00, prev_close=3.80)
        assert ok is True

    def test_etf_gem_limit(self):
        """科创/创业板ETF(588/159开头)是20%涨跌停"""
        asc = AShareConstraints(asset_type="etf")
        ok, msg = asc.can_buy("588000", price=4.20, prev_close=3.80)
        assert ok is True

    def test_convertible_bond_no_limit(self):
        """可转债无涨跌停限制，大幅上涨也应该允许买入"""
        asc = AShareConstraints(asset_type="convertible_bond")
        ok, msg = asc.can_buy("123456", price=200.0, prev_close=130.0)
        assert ok is True


class TestCanSell:
    def test_normal_sell_ok(self):
        asc = AShareConstraints()
        ok, msg = asc.can_sell("600519", price=105.0, prev_close=100.0,
                               buy_date=date(2025, 1, 5), today=date(2025, 1, 8))
        assert ok is True

    def test_t1_same_day_blocked(self):
        asc = AShareConstraints()
        ok, msg = asc.can_sell("600519", price=105.0, prev_close=100.0,
                               buy_date=date(2025, 1, 6), today=date(2025, 1, 6))
        assert ok is False
        assert "T+1" in msg

    def test_t1_buy_after_today_blocked(self):
        asc = AShareConstraints()
        ok, msg = asc.can_sell("600519", price=105.0, prev_close=100.0,
                               buy_date=date(2025, 1, 10), today=date(2025, 1, 8))
        assert ok is False

    def test_t1_disabled(self):
        asc = AShareConstraints(enable_t1=False)
        ok, msg = asc.can_sell("600519", price=105.0, prev_close=100.0,
                               buy_date=date(2025, 1, 6), today=date(2025, 1, 6))
        assert ok is True

    def test_limit_down_blocked(self):
        asc = AShareConstraints()
        ok, msg = asc.can_sell("600519", price=90.0, prev_close=100.0,
                               buy_date=date(2025, 1, 5), today=date(2025, 1, 8))
        assert ok is False
        assert "跌停" in msg

    def test_price_limit_disabled(self):
        asc = AShareConstraints(enable_price_limit=False)
        ok, msg = asc.can_buy("600519", price=110.0, prev_close=100.0)
        assert ok is True


class TestCheckMethods:
    def test_check_buy_alias(self):
        asc = AShareConstraints()
        ok1, _ = asc.can_buy("600519", 105.0, 100.0)
        ok2, _ = asc.check_buy("600519", 105.0, 100.0)
        assert ok1 == ok2

    def test_check_sell_alias(self):
        asc = AShareConstraints()
        ok1, _ = asc.can_sell("600519", 105.0, 100.0,
                              date(2025, 1, 5), date(2025, 1, 8))
        ok2, _ = asc.check_sell("600519", 105.0, 100.0,
                                date(2025, 1, 5), date(2025, 1, 8))
        assert ok1 == ok2
