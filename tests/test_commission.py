"""
A股佣金+印花税模型单元测试 — AShareCommission

万2.5佣金(双向,最低5元) + 万5印花税(仅卖出)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from backtest.engine.bt_runner import AShareCommission


def _make_commission(**kwargs):
    return AShareCommission(**kwargs)


class TestBuyCommission:
    """买入只收佣金，不收印花税"""

    def test_small_buy_min_commission(self):
        """100股*10元=1000, 佣金0.25→最低5元 + 过户费0.01"""
        comm = _make_commission()
        cost = comm._getcommission(size=100, price=10.0, pseudoexec=True)
        assert cost == 5.01  # 5(min_comm) + 0.01(transfer_fee)

    def test_large_buy_pct_commission(self):
        """10000股*100元=1M, 佣金250 + 过户费10"""
        comm = _make_commission()
        cost = comm._getcommission(size=10000, price=100.0, pseudoexec=True)
        assert cost == 260.0  # 250(comm) + 10(transfer)

    def test_buy_no_stamp_duty(self):
        """买入不收印花税，但有过户费"""
        comm = _make_commission()
        cost = comm._getcommission(size=1000, price=50.0, pseudoexec=True)
        assert cost == 13.0  # 12.5(comm) + 0.5(transfer)


class TestSellCommission:
    """卖出收佣金+印花税"""

    def test_small_sell_both(self):
        """小金额卖出: 佣金最低5元+过户费0.01+印花税0.5=5.51"""
        comm = _make_commission()
        cost = comm._getcommission(size=-100, price=10.0, pseudoexec=True)
        assert cost == 5.51

    def test_large_sell_both(self):
        """大金额卖出: 佣金125+过户费5+印花税250=380"""
        comm = _make_commission()
        cost = comm._getcommission(size=-10000, price=50.0, pseudoexec=True)
        assert cost == 380.0

    def test_sell_expensive_stock(self):
        """高价股: 100股*1500=150k, 佣金37.5+过户费1.5+印花税75=114"""
        comm = _make_commission()
        cost = comm._getcommission(size=-100, price=1500.0, pseudoexec=True)
        assert cost == 114.0


class TestEdgeCases:
    def test_zero_size(self):
        comm = _make_commission()
        cost = comm._getcommission(size=0, price=100.0, pseudoexec=True)
        assert cost >= 0

    def test_cost_ratio_sanity(self):
        """双向买卖总成本约成交额0.102% (佣金万2.5×2 + 过户费万0.1×2 + 印花税万5)"""
        comm = _make_commission()
        price, size = 50.0, 1000
        value = size * price
        buy = comm._getcommission(size=size, price=price, pseudoexec=True)
        sell = comm._getcommission(size=-size, price=price, pseudoexec=True)
        total = buy + sell
        ratio = total / value
        # 12.5+0.5(buy) + 12.5+0.5+25(sell) = 51.0 / 50000 = 0.00102
        assert 0.0008 < ratio < 0.0013


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
