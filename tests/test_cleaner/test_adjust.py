"""
复权处理单元测试 — 测试 price_gap 检测和手动复权
"""
import pytest
import pandas as pd
import numpy as np
from data.cleaner.adjust import check_price_gap, forward_adjust_manual


class TestCheckPriceGap:
    def test_no_gap_normal_data(self):
        prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        gaps = check_price_gap(prices, threshold_pct=5.0)
        assert gaps.empty

    def test_detect_large_gap(self):
        prices = pd.Series([100.0, 100.5, 80.0, 81.0])
        gaps = check_price_gap(prices, threshold_pct=5.0)
        assert not gaps.empty
        assert abs(gaps.iloc[0]["change_pct"]) > 5

    def test_short_data_returns_empty(self):
        prices = pd.Series([100.0])
        gaps = check_price_gap(prices)
        assert gaps.empty

    def test_output_columns(self):
        prices = pd.Series([100.0, 100.5, 80.0, 81.0])
        gaps = check_price_gap(prices, threshold_pct=5.0)
        expected_cols = {"date", "prev_close", "close", "change_pct"}
        assert expected_cols.issubset(set(gaps.columns))

    def test_custom_threshold(self):
        prices = pd.Series([100.0, 103.0, 106.0])
        gaps_default = check_price_gap(prices, threshold_pct=5.0)
        gaps_strict = check_price_gap(prices, threshold_pct=1.0)
        assert len(gaps_strict) >= len(gaps_default)


class TestForwardAdjustManual:
    def test_adjust_adds_columns(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="B"),
            "open": [10.0] * 5,
            "high": [11.0] * 5,
            "low": [9.0] * 5,
            "close": [10.5] * 5,
            "volume": [1000] * 5,
            "adjust_factor": [0.95, 1.0, 1.0, 0.9, 1.0],
        })
        result = forward_adjust_manual(df)
        assert "adj_close" in result.columns
        assert "adj_open" in result.columns
        assert "adj_high" in result.columns
        assert "adj_low" in result.columns

    def test_no_adjust_factor_raises(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="B"),
            "close": [10.0] * 5,
        })
        with pytest.raises(ValueError, match="adjust_factor"):
            forward_adjust_manual(df)

    def test_adjust_preserves_original_columns(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="B"),
            "open": [10.0] * 5,
            "high": [11.0] * 5,
            "low": [9.0] * 5,
            "close": [10.5] * 5,
            "volume": [1000] * 5,
            "adjust_factor": [1.0] * 5,
        })
        result = forward_adjust_manual(df)
        assert "close" in result.columns
        assert "adj_close" in result.columns
