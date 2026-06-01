"""
环境过滤器烟雾测试 — 不调LLM，L4+Agent规则引擎。
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def env_df():
    n = 252
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    trend = np.linspace(0, 0.3, n)
    noise = np.random.randn(n) * 0.01
    price = 100 * np.cumprod(1 + trend / n + noise)
    return pd.DataFrame({
        "date": dates, "open": price * 0.998, "high": price * 1.012,
        "low": price * 0.988, "close": price, "volume": np.ones(n) * 1e6,
    })


class TestEnvironmentFilter:
    def test_import_and_create(self, env_df):
        from backtest.analysis.env_filter import EnvironmentFilter
        ef = EnvironmentFilter(env_df)
        assert ef.df is not None

    def test_assess_returns_complete_structure(self, env_df):
        from backtest.analysis.env_filter import EnvironmentFilter
        ef = EnvironmentFilter(env_df)
        r = ef.assess()
        for key in ("should_trade", "direction", "position_multiplier",
                     "position_action", "position_note", "risk_level",
                     "l4_signal", "agent_signal", "reasoning"):
            assert key in r, f"缺少字段: {key}"
        assert r["direction"] in ("long", "short", "hold")
        assert r["position_action"] in ("keep", "reduce", "close")
        assert 0 <= r["position_multiplier"] <= 1

    def test_assess_no_llm_no_crash(self, env_df):
        from backtest.analysis.env_filter import EnvironmentFilter
        ef = EnvironmentFilter(env_df, use_llm=False)
        r = ef.assess()
        assert r["agent_signal"]["mode"] in ("rule_based", "error")

    def test_direction_change_detection(self):
        from backtest.analysis.env_filter import EnvironmentFilter
        # 方向反转
        assert EnvironmentFilter.check_direction_change("long", "short") is True
        assert EnvironmentFilter.check_direction_change("short", "long") is True
        # 非反转
        assert EnvironmentFilter.check_direction_change("long", "hold") is False
        assert EnvironmentFilter.check_direction_change("hold", "short") is False
        assert EnvironmentFilter.check_direction_change("long", "long") is False

    def test_strategy_filter_direction(self, env_df):
        from backtest.analysis.env_filter import EnvironmentFilter
        ef = EnvironmentFilter(env_df)
        # hold env → 不限制方向
        allowed, mult = ef.as_strategy_filter("long")
        assert isinstance(allowed, bool)
        assert 0 < mult <= 1

    def test_pipeline_with_env_filter(self, env_df):
        from core.pipeline import Pipeline, Stage
        from backtest.analysis.env_filter import EnvironmentFilter
        ef = EnvironmentFilter(env_df)
        pl = Pipeline(env_df, cash=100000, env_filter=ef)
        assert pl.env_filter is ef
        assert pl.env_result is None  # 跑之前为None


class TestPositionAction:
    """覆盖position_action的4种状态转换 + direction_change方向反转"""

    @staticmethod
    def _make_ef(l4_action="hold", l4_conf=0.0, agent_action="hold",
                  agent_conf=0.0, risk_flags=None):
        """构造EnvironmentFilter并注入受控的L4/Agent结果"""
        import numpy as np
        import pandas as pd
        from backtest.analysis.env_filter import EnvironmentFilter

        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=100, freq="B"),
            "open": np.ones(100), "high": np.ones(100),
            "low": np.ones(100), "close": np.linspace(100, 110, 100),
            "volume": np.ones(100) * 1e6,
        })
        ef = EnvironmentFilter(df)
        ef._l4_result = {
            "action": l4_action, "confidence": l4_conf, "signal": 0.5,
            "components": {"alt": 0.5, "rl": 0.5, "ml": 0.5},
        }
        ef._agent_result = {
            "action": agent_action, "confidence": agent_conf,
            "mode": "test", "risk_flags": risk_flags or [],
            "reasoning": "test",
        }
        # 替换方法使其返回注入的结果
        ef._run_l4 = lambda use_rl=False: ef._l4_result
        ef._run_agent = lambda: ef._agent_result
        return ef

    # ---- 4种状态转换 ----

    def test_high_hold_to_reduce(self):
        """risk=high + direction=hold → position_action=reduce"""
        ef = self._make_ef(
            l4_action="hold", l4_conf=0.0,
            agent_action="hold", agent_conf=0.0,
        )
        # l4=hold, agent=hold → direction=hold, should_trade=False → risk_level forced high
        r = ef.assess()
        assert r["risk_level"] == "high"
        assert r["direction"] == "hold"
        assert r["position_action"] == "reduce"
        assert "减至半仓" in r["position_note"]

    def test_high_direction_to_close(self):
        """risk=high + direction!=hold → position_action=close"""
        ef = self._make_ef(
            l4_action="buy", l4_conf=0.8,
            agent_action="buy", agent_conf=0.8,
            risk_flags=["flag1", "flag2", "flag3"],  # ≥3 → high
        )
        r = ef.assess()
        assert r["risk_level"] == "high"
        assert r["direction"] == "long"
        assert r["position_action"] == "close"
        assert "平仓" in r["position_note"]

    def test_medium_to_reduce(self):
        """risk=medium → position_action=reduce"""
        ef = self._make_ef(
            l4_action="buy", l4_conf=0.8,
            agent_action="buy", agent_conf=0.8,
            risk_flags=["flag1"],  # 1个 → medium
        )
        r = ef.assess()
        assert r["risk_level"] == "medium"
        assert r["position_action"] == "reduce"
        assert "仓位上限" in r["position_note"]

    def test_low_to_keep(self):
        """risk=low → position_action=keep"""
        ef = self._make_ef(
            l4_action="buy", l4_conf=0.8,
            agent_action="buy", agent_conf=0.8,
            risk_flags=[],  # 0个 → low
        )
        r = ef.assess()
        assert r["risk_level"] == "low"
        assert r["position_action"] == "keep"
        assert "维持" in r["position_note"]

    # ---- 边界+组合 ----

    def test_conflicting_signals_high_risk(self):
        """L4和Agent方向不一致 → direction=hold, should_trade=False → risk=high, position_action=reduce"""
        ef = self._make_ef(
            l4_action="buy", l4_conf=0.8,
            agent_action="sell", agent_conf=0.7,
        )
        r = ef.assess()
        assert r["direction"] == "hold"
        assert r["should_trade"] is False
        assert r["risk_level"] == "high"
        assert r["position_action"] == "reduce"

    def test_short_direction_with_high_risk(self):
        """做空方向+高风险 → close"""
        ef = self._make_ef(
            l4_action="sell", l4_conf=0.8,
            agent_action="sell", agent_conf=0.8,
            risk_flags=["a", "b", "c"],
        )
        r = ef.assess()
        assert r["direction"] == "short"
        assert r["risk_level"] == "high"
        assert r["position_action"] == "close"

    def test_low_confidence_no_trade_but_same_direction(self):
        """低置信度双方同向 → should_trade=False但direction保持同向 → risk=high → close"""
        ef = self._make_ef(
            l4_action="buy", l4_conf=0.3,
            agent_action="buy", agent_conf=0.3,
        )
        # consensus_conf=0.3, should_trade=False, 但两方同向buy→direction=long
        r = ef.assess()
        assert r["should_trade"] is False
        assert r["direction"] == "long"  # 方向一致，不受should_trade影响
        assert r["risk_level"] == "high"
        assert r["position_action"] == "close"  # high + 有方向 → close

    # ---- check_direction_change ----

    def test_direction_reversal_long_to_short(self):
        from backtest.analysis.env_filter import EnvironmentFilter
        assert EnvironmentFilter.check_direction_change("long", "short") is True

    def test_direction_reversal_short_to_long(self):
        from backtest.analysis.env_filter import EnvironmentFilter
        assert EnvironmentFilter.check_direction_change("short", "long") is True

    def test_direction_no_change(self):
        from backtest.analysis.env_filter import EnvironmentFilter
        assert EnvironmentFilter.check_direction_change("long", "long") is False
        assert EnvironmentFilter.check_direction_change("short", "short") is False
        assert EnvironmentFilter.check_direction_change("hold", "hold") is False

    def test_direction_to_from_hold_not_reversal(self):
        from backtest.analysis.env_filter import EnvironmentFilter
        # hold↔其他不算方向反转
        assert EnvironmentFilter.check_direction_change("long", "hold") is False
        assert EnvironmentFilter.check_direction_change("hold", "long") is False
        assert EnvironmentFilter.check_direction_change("short", "hold") is False
        assert EnvironmentFilter.check_direction_change("hold", "short") is False
