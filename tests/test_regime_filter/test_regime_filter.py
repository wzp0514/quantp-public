"""
体制识别模块单元测试 — 真实CSI300数据，覆盖get_regime_signal/should_trade/label_trades_with_regime/regime_summary。
"""

import numpy as np
import pandas as pd
import pytest


class TestGetRegimeSignal:
    def test_output_structure(self, csi300_2023):
        """get_regime_signal 返回完整结构"""
        from backtest.analysis.regime_filter import get_regime_signal

        close = csi300_2023.set_index("date")["close"]
        sig = get_regime_signal(close)

        for key in ("current_regime", "signal", "bull_prob", "bear_prob",
                     "long_ok", "short_ok", "persistence", "stationary"):
            assert key in sig, f"缺少字段: {key}"

    def test_regime_is_valid(self, csi300_2023):
        """current_regime 在三种状态内"""
        from backtest.analysis.regime_filter import get_regime_signal

        close = csi300_2023.set_index("date")["close"]
        sig = get_regime_signal(close)
        assert sig["current_regime"] in ("Bull", "Bear", "Sideways")

    def test_signal_range(self, csi300_2023):
        """signal 在 [-1, 1] 范围内"""
        from backtest.analysis.regime_filter import get_regime_signal

        close = csi300_2023.set_index("date")["close"]
        sig = get_regime_signal(close)
        assert -1.0 <= sig["signal"] <= 1.0

    def test_probabilities_sum_to_one(self, csi300_2023):
        """bull_prob + bear_prob 在合理范围（加Sideways 概率≈1）"""
        from backtest.analysis.regime_filter import get_regime_signal

        close = csi300_2023.set_index("date")["close"]
        sig = get_regime_signal(close)
        # bull + bear + sideways ≈ 1.0 (从transition matrix下一行来)
        total = sig["bull_prob"] + sig["bear_prob"]
        assert 0 <= total <= 1.0

    def test_long_ok_short_ok_are_bool(self, csi300_2023):
        from backtest.analysis.regime_filter import get_regime_signal

        close = csi300_2023.set_index("date")["close"]
        sig = get_regime_signal(close)
        assert isinstance(sig["long_ok"], bool) or isinstance(sig["long_ok"], (np.bool_,))
        assert isinstance(sig["short_ok"], bool) or isinstance(sig["short_ok"], (np.bool_,))

    def test_persistence_structure(self, csi300_2023):
        """persistence 含三个区制粘性"""
        from backtest.analysis.regime_filter import get_regime_signal

        close = csi300_2023.set_index("date")["close"]
        sig = get_regime_signal(close)
        for k in ("bear", "sideways", "bull"):
            assert k in sig["persistence"], f"persistence 缺少: {k}"
            assert 0 <= sig["persistence"][k] <= 1

    def test_stationary_structure(self, csi300_2023):
        """stationary 含三个区制长期分布"""
        from backtest.analysis.regime_filter import get_regime_signal

        close = csi300_2023.set_index("date")["close"]
        sig = get_regime_signal(close)
        for k in ("bear", "sideways", "bull"):
            assert k in sig["stationary"], f"stationary 缺少: {k}"
        total = sum(sig["stationary"].values())
        assert abs(total - 1.0) < 0.01, f"stationary distribution 应≈1, got {total}"

    def test_custom_window(self, csi300_2023):
        """自定义窗口不崩溃"""
        from backtest.analysis.regime_filter import get_regime_signal

        close = csi300_2023.set_index("date")["close"]
        sig = get_regime_signal(close, window=10, threshold=0.03)
        assert sig["current_regime"] in ("Bull", "Bear", "Sideways")


class TestShouldTrade:
    def _make_regime_info(self, regime="Bull", long_ok=True, short_ok=False, signal=0.3):
        return {
            "current_regime": regime,
            "signal": signal,
            "long_ok": long_ok,
            "short_ok": short_ok,
        }

    def test_bull_long_allowed(self):
        """Bull区制做多允许"""
        from backtest.analysis.regime_filter import should_trade
        info = self._make_regime_info("Bull", long_ok=True)
        assert should_trade(info, direction="long") is True

    def test_bear_blocked_by_default(self):
        """Bear区制默认禁止"""
        from backtest.analysis.regime_filter import should_trade
        info = self._make_regime_info("Bear", long_ok=False, short_ok=True)
        assert should_trade(info, direction="long") is False

    def test_bear_allowed_when_explicit(self):
        """Bear区制显式加入allowed_regimes后允许"""
        from backtest.analysis.regime_filter import should_trade
        info = self._make_regime_info("Bear", long_ok=False, short_ok=True)
        assert should_trade(info, direction="short", allowed_regimes=("Bear",)) is True

    def test_short_in_bull_blocked(self):
        """Bull区制做空被long_ok=False阻止"""
        from backtest.analysis.regime_filter import should_trade
        info = self._make_regime_info("Bull", long_ok=True, short_ok=False)
        assert should_trade(info, direction="short") is False

    def test_min_confidence_filter(self):
        """min_confidence门控：信号太弱拒绝"""
        from backtest.analysis.regime_filter import should_trade
        info = self._make_regime_info("Bull", long_ok=True, signal=0.05)
        assert should_trade(info, direction="long", min_confidence=0.1) is False
        assert should_trade(info, direction="long", min_confidence=0.0) is True

    def test_sideways_long_ok_with_positive_signal(self, csi300_2023):
        """真实数据：Sideways + 正信号 → long_ok=True"""
        from backtest.analysis.regime_filter import get_regime_signal
        close = csi300_2023.set_index("date")["close"]
        sig = get_regime_signal(close)
        if sig["current_regime"] == "Sideways":
            assert sig["long_ok"] == (sig["signal"] > 0)
            assert sig["short_ok"] == (sig["signal"] < 0)

    def test_direction_not_recognized_still_returns_bool(self):
        """未知方向也返回bool不崩溃"""
        from backtest.analysis.regime_filter import should_trade
        info = self._make_regime_info("Bull", long_ok=True)
        result = should_trade(info, direction="unknown")
        assert isinstance(result, bool)


class TestLabelTradesWithRegime:
    def test_returns_same_df_when_empty(self, csi300_2023):
        """空trades_df原样返回"""
        from backtest.analysis.regime_filter import label_trades_with_regime
        close = csi300_2023.set_index("date")["close"]
        empty = pd.DataFrame()
        result = label_trades_with_regime(empty, close)
        assert result.empty

    def test_adds_regime_columns(self, csi300_2023):
        """给交易表添加regime和regime_signal列"""
        from backtest.analysis.regime_filter import label_trades_with_regime
        from backtest.engine.bt_runner import run_backtest
        from backtest.strategies.builtin.ma_cross import MaCrossStrategy

        close = csi300_2023.set_index("date")["close"]
        df_raw = csi300_2023.copy()

        result = run_backtest(MaCrossStrategy, df_raw, fast=5, slow=20)
        trades = result["trades_df"]
        if trades.empty:
            pytest.skip("回测无交易，跳过")

        labeled = label_trades_with_regime(trades, close)
        assert "regime" in labeled.columns
        assert "regime_signal" in labeled.columns
        assert all(labeled["regime"].isin(["Bull", "Sideways", "Bear"]))

    def test_regime_values_valid(self, csi300_2023):
        """regime列值在有效集合内"""
        from backtest.analysis.regime_filter import label_trades_with_regime
        from backtest.engine.bt_runner import run_backtest
        from backtest.strategies.builtin.ma_cross import MaCrossStrategy

        close = csi300_2023.set_index("date")["close"]
        df_raw = csi300_2023.copy()

        result = run_backtest(MaCrossStrategy, df_raw, fast=5, slow=20)
        trades = result["trades_df"]
        if trades.empty:
            pytest.skip("回测无交易，跳过")

        labeled = label_trades_with_regime(trades, close)
        valid = {"Bull", "Sideways", "Bear"}
        assert set(labeled["regime"].unique()).issubset(valid)


class TestRegimeSummary:
    def test_empty_trades_returns_placeholder(self):
        """空trades或缺少regime列返回占位推荐"""
        from backtest.analysis.regime_filter import regime_summary
        empty = pd.DataFrame()
        result = regime_summary(empty)
        assert "no regime data" in result["recommendation"]

    def test_has_by_regime_keys(self, csi300_2023):
        """by_regime 含三个区制"""
        from backtest.analysis.regime_filter import label_trades_with_regime, regime_summary
        from backtest.engine.bt_runner import run_backtest
        from backtest.strategies.builtin.ma_cross import MaCrossStrategy

        close = csi300_2023.set_index("date")["close"]
        df_raw = csi300_2023.copy()

        result = run_backtest(MaCrossStrategy, df_raw, fast=5, slow=20)
        trades = result["trades_df"]
        if trades.empty:
            pytest.skip("回测无交易，跳过")

        labeled = label_trades_with_regime(trades, close)
        summary = regime_summary(labeled)
        for regime in ("Bull", "Sideways", "Bear"):
            assert regime in summary["by_regime"], f"缺少区制: {regime}"

    def test_recommendation_is_string(self, csi300_2023):
        """推荐是字符串"""
        from backtest.analysis.regime_filter import label_trades_with_regime, regime_summary
        from backtest.engine.bt_runner import run_backtest
        from backtest.strategies.builtin.ma_cross import MaCrossStrategy

        close = csi300_2023.set_index("date")["close"]
        df_raw = csi300_2023.copy()

        result = run_backtest(MaCrossStrategy, df_raw, fast=5, slow=20)
        trades = result["trades_df"]
        if trades.empty:
            pytest.skip("回测无交易，跳过")

        labeled = label_trades_with_regime(trades, close)
        summary = regime_summary(labeled)
        assert isinstance(summary["recommendation"], str)
        assert len(summary["recommendation"]) > 0
