"""
守护进程单元测试 — 测试 Guardian 的策略漂移/仓位漂移/自动熔断/状态
"""
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from live.monitor.guardian import Guardian


@pytest.fixture
def guardian():
    """创建 Guardian，修改状态文件路径到临时目录"""
    import live.monitor.guardian as gm
    orig = gm.STATE_FILE
    with tempfile.TemporaryDirectory() as tmp:
        state_file = Path(tmp) / ".guardian_state.json"
        gm.STATE_FILE = state_file
        g = Guardian()
        yield g
        gm.STATE_FILE = orig


class TestStrategyDrift:
    def test_no_drift(self, guardian):
        result = guardian.check_strategy_drift(
            "双均线", backtest_signals_per_week=5.0,
            live_signals_per_week=5.5, backtest_win_rate=0.45,
            live_win_rate=0.42,
        )
        assert result["healthy"] is True

    def test_signal_drift_detected(self, guardian):
        result = guardian.check_strategy_drift(
            "双均线", backtest_signals_per_week=5.0,
            live_signals_per_week=10.0, backtest_win_rate=0.45,
            live_win_rate=0.44,
        )
        assert result["healthy"] is False
        assert result["signal_deviation"] > 0.5
        assert any("信号" in w for w in result["warnings"])

    def test_winrate_drift_detected(self, guardian):
        result = guardian.check_strategy_drift(
            "双均线", backtest_signals_per_week=5.0,
            live_signals_per_week=5.5, backtest_win_rate=0.45,
            live_win_rate=0.15,
        )
        assert result["healthy"] is False
        assert any("胜率" in w for w in result["warnings"])

    def test_both_drifts(self, guardian):
        result = guardian.check_strategy_drift(
            "双均线", backtest_signals_per_week=5.0,
            live_signals_per_week=12.0, backtest_win_rate=0.45,
            live_win_rate=0.10,
        )
        assert result["healthy"] is False
        assert len(result["warnings"]) == 2


class TestLiveVsBacktestDD:
    def test_ratio_below_threshold(self, guardian):
        result = guardian.check_live_vs_backtest_dd(
            "双均线", backtest_max_dd=0.15, live_max_dd=0.20,
        )
        assert result["warning"] is False

    def test_ratio_above_threshold(self, guardian):
        result = guardian.check_live_vs_backtest_dd(
            "双均线", backtest_max_dd=0.10, live_max_dd=0.20,
        )
        assert result["warning"] is True
        assert result["ratio"] > 1.5

    def test_zero_backtest_dd(self, guardian):
        result = guardian.check_live_vs_backtest_dd(
            "双均线", backtest_max_dd=0.0, live_max_dd=0.10,
        )
        assert result["warning"] is False
        assert "无效" in result["detail"]


class TestPositionDrift:
    def test_no_drift_both_zero(self, guardian):
        result = guardian.check_position_drift(
            {"size": 0}, {"size": 0},
        )
        assert result["drift"] is False
        assert "空仓" in result["detail"]

    def test_no_drift_equal(self, guardian):
        result = guardian.check_position_drift(
            {"size": 100}, {"size": 100},
        )
        assert result["drift"] is False

    def test_drift_detected(self, guardian):
        result = guardian.check_position_drift(
            {"size": 100}, {"size": 120},
        )
        assert result["drift"] is True

    def test_small_diff_not_drift(self, guardian):
        result = guardian.check_position_drift(
            {"size": 100}, {"size": 104},
        )
        assert result["drift"] is False

    def test_zero_expected_nonzero_actual(self, guardian):
        result = guardian.check_position_drift(
            {"size": 0}, {"size": 50},
        )
        assert result["drift"] is True


class TestEmergencyStop:
    def test_emergency_stop_returns_record(self, guardian):
        record = guardian.emergency_stop("测试熔断")
        assert record["action"] == "emergency_stop"
        assert record["reason"] == "测试熔断"

    def test_emergency_stop_disables_live_mode(self, guardian):
        guardian._live_mode = True
        guardian.emergency_stop("测试")
        assert guardian.is_live() is False


class TestAutoEmergencyCheck:
    def test_no_trigger_normal(self, guardian):
        with patch("live.risk.risk_engine.get_params", return_value={
            "max_daily_loss_pct": 0.05, "max_drawdown_pct": 0.15,
            "stop_after_n_losses": 5,
        }):
            triggered = guardian.auto_emergency_check(0.01, 0.05, 1)
            assert triggered is False

    def test_daily_loss_triggers(self, guardian):
        with patch("live.risk.risk_engine.get_params", return_value={
            "max_daily_loss_pct": 0.05, "max_drawdown_pct": 0.15,
            "stop_after_n_losses": 5,
        }):
            triggered = guardian.auto_emergency_check(0.10, 0.05, 1)
            assert triggered is True

    def test_max_drawdown_triggers(self, guardian):
        with patch("live.risk.risk_engine.get_params", return_value={
            "max_daily_loss_pct": 0.05, "max_drawdown_pct": 0.15,
            "stop_after_n_losses": 5,
        }):
            triggered = guardian.auto_emergency_check(0.01, 0.20, 1)
            assert triggered is True

    def test_consecutive_losses_triggers(self, guardian):
        with patch("live.risk.risk_engine.get_params", return_value={
            "max_daily_loss_pct": 0.05, "max_drawdown_pct": 0.15,
            "stop_after_n_losses": 5,
        }):
            triggered = guardian.auto_emergency_check(0.01, 0.05, 7)
            assert triggered is True


class TestStatus:
    def test_status_returns_dict(self, guardian):
        s = guardian.status()
        assert "live_mode" in s
        assert "data_health" in s
        assert isinstance(s["live_mode"], bool)
