"""
风控引擎单元测试 — 覆盖8个check函数 + run_all_checks
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from live.risk.risk_engine import (
    check_position_limit,
    check_single_loss,
    check_daily_loss,
    check_drawdown,
    check_consecutive_losses,
    check_martingale,
    check_order_frequency,
    check_leverage,
    run_all_checks,
    get_params,
)


class TestPositionLimit:
    def test_ok_below_limit(self):
        ok, msg = check_position_limit(0.05)
        assert ok
        assert "通过" in msg

    def test_over_limit(self):
        ok, msg = check_position_limit(0.30)
        assert not ok
        assert "超限" in msg

    def test_at_limit(self):
        limit = get_params().get("max_position_pct", 0.20)
        ok, _ = check_position_limit(limit)
        assert ok


class TestSingleLoss:
    def test_loss_ok(self):
        ok, msg = check_single_loss(0.005)
        assert ok
        assert "未触发" in msg

    def test_loss_triggered(self):
        ok, msg = check_single_loss(0.05)
        assert not ok
        assert "触发" in msg


class TestDailyLoss:
    def test_daily_ok(self):
        ok, _ = check_daily_loss(0.01)
        assert ok

    def test_daily_meltdown(self):
        ok, msg = check_daily_loss(0.10)
        assert not ok
        assert "熔断" in msg


class TestDrawdown:
    def test_drawdown_ok(self):
        ok, _ = check_drawdown(0.05)
        assert ok

    def test_drawdown_permanent_stop(self):
        ok, msg = check_drawdown(0.25)
        assert not ok
        assert "永久停用" in msg


class TestConsecutiveLosses:
    def test_few_losses_ok(self):
        ok, _ = check_consecutive_losses(2)
        assert ok

    def test_many_losses_pause(self):
        ok, msg = check_consecutive_losses(7)
        assert not ok
        assert "暂停" in msg

    def test_exactly_limit(self):
        limit = get_params().get("stop_after_n_losses", 5)
        ok, msg = check_consecutive_losses(limit)
        # < limit means ok at count==limit → not ok
        assert not ok


class TestMartingale:
    def test_no_history(self):
        ok, msg = check_martingale([], 0.1, 0.1)
        assert ok

    def test_normal_scaling(self):
        history = [{"pnl": -100}, {"pnl": 200}]
        ok, _ = check_martingale(history, 0.12, 0.10)
        assert ok

    def test_martingale_detected(self):
        history = [{"pnl": -100}, {"pnl": -200}]
        ok, msg = check_martingale(history, 0.30, 0.10)
        assert not ok
        assert "马丁" in msg


class TestOrderFrequency:
    def test_low_frequency_ok(self):
        ok, _ = check_order_frequency(5, time_minutes=60)
        assert ok

    def test_high_frequency_trigger(self):
        ok, msg = check_order_frequency(30, time_minutes=60)
        assert not ok
        assert "频率" in msg


class TestLeverage:
    def test_no_leverage_ok(self):
        ok, _ = check_leverage(0)
        assert ok

    def test_leverage_rejected(self):
        ok, msg = check_leverage(2.0)
        assert not ok
        assert "拒绝" in msg or "禁止" in msg


class TestRunAllChecks:
    def test_all_green(self):
        results = run_all_checks(
            position_pct=0.05,
            single_loss_pct=0.005,
            daily_loss_pct=0.01,
            drawdown_pct=0.05,
            consecutive_losses=2,
        )
        assert all(ok for ok, _ in results)

    def test_all_red(self):
        results = run_all_checks(
            position_pct=0.50,
            single_loss_pct=0.10,
            daily_loss_pct=0.15,
            drawdown_pct=0.30,
            consecutive_losses=10,
            requested_leverage=3.0,
        )
        failures = [ok for ok, _ in results if not ok]
        assert len(failures) >= 3  # most should fail

    def test_returns_list_of_tuples(self):
        results = run_all_checks()
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], bool)
            assert isinstance(item[1], str)


class TestParams:
    def test_get_params_returns_dict(self):
        p = get_params()
        assert isinstance(p, dict)
        assert "max_position_pct" in p
        assert "no_leverage" in p


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
