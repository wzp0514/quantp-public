"""
管线入口信号引擎单元测试 — 覆盖ResearchSignal/置信度门控/冷静期去重/scan_all。
"""

from datetime import datetime, timedelta

import pytest

from core.research_source import (
    ResearchSignal,
    ResearchSource,
    _confidence_gate,
    _cooldown_check,
    _fwer_filter,
)


class TestResearchSignal:
    def test_create_defaults(self):
        s = ResearchSignal(source="test", title="测试信号")
        assert s.source == "test"
        assert s.title == "测试信号"
        assert s.action == "alert"
        assert s.priority == 1
        assert s.confidence == 0.5
        assert s.created_at != ""

    def test_high_priority_triggers_review(self):
        s = ResearchSignal(source="test", title="高危", priority=5)
        assert s.needs_review is True

    def test_low_priority_no_review(self):
        s = ResearchSignal(source="test", title="普通", priority=3)
        assert s.needs_review is False

    def test_to_dict_contains_all_fields(self):
        s = ResearchSignal(source="test", title="标题", action="factor",
                           priority=4, confidence=0.8, suggested_track="track1")
        d = s.to_dict()
        for key in ("source", "title", "action", "priority", "data",
                     "confidence", "suggested_track", "needs_review", "created_at"):
            assert key in d, f"缺少字段: {key}"

    def test_custom_created_at_preserved(self):
        ts = "2025-01-01T00:00:00"
        s = ResearchSignal(source="test", title="x", created_at=ts)
        assert s.created_at == ts


class TestConfidenceGate:
    def test_filters_below_threshold(self):
        signals = [
            ResearchSignal(source="s1", title="a", confidence=0.2),
            ResearchSignal(source="s1", title="b", confidence=0.5),
            ResearchSignal(source="s1", title="c", confidence=0.3),
        ]
        result = _confidence_gate(signals, min_conf=0.3)
        assert len(result) == 2
        titles = {s.title for s in result}
        assert titles == {"b", "c"}

    def test_empty_list(self):
        assert _confidence_gate([], min_conf=0.5) == []

    def test_all_pass(self):
        signals = [
            ResearchSignal(source="s1", title="a", confidence=0.8),
            ResearchSignal(source="s1", title="b", confidence=0.9),
        ]
        result = _confidence_gate(signals, min_conf=0.3)
        assert len(result) == 2


class TestCooldownCheck:
    def test_no_history_returns_all(self):
        signals = [
            ResearchSignal(source="s1", title="a", action="alert"),
            ResearchSignal(source="s1", title="b", action="factor"),
        ]
        result = _cooldown_check(signals, [], cooldown_hours=24)
        assert len(result) == 2

    def test_recent_duplicate_blocked(self):
        recent_ts = datetime.now().isoformat()
        history = [{"source": "s1", "action": "alert", "created_at": recent_ts}]
        signals = [
            ResearchSignal(source="s1", title="a", action="alert"),
            ResearchSignal(source="s1", title="b", action="factor"),
        ]
        result = _cooldown_check(signals, history, cooldown_hours=24)
        assert len(result) == 1
        assert result[0].action == "factor"

    def test_old_history_not_blocked(self):
        old_ts = (datetime.now() - timedelta(hours=48)).isoformat()
        history = [{"source": "s1", "action": "alert", "created_at": old_ts}]
        signals = [
            ResearchSignal(source="s1", title="a", action="alert"),
        ]
        result = _cooldown_check(signals, history, cooldown_hours=24)
        assert len(result) == 1

    def test_different_source_not_blocked(self):
        recent_ts = datetime.now().isoformat()
        history = [{"source": "s1", "action": "alert", "created_at": recent_ts}]
        signals = [
            ResearchSignal(source="s2", title="a", action="alert"),
        ]
        result = _cooldown_check(signals, history, cooldown_hours=24)
        assert len(result) == 1

    def test_history_bad_timestamp_not_blocked(self):
        history = [{"source": "s1", "action": "alert", "created_at": "not-a-date"}]
        signals = [
            ResearchSignal(source="s1", title="a", action="alert"),
        ]
        result = _cooldown_check(signals, history, cooldown_hours=24)
        assert len(result) == 1  # 坏时间戳被跳过，不阻塞


class TestFWERFilter:
    def test_single_signal_passes(self):
        signals = [ResearchSignal(source="s1", title="a", confidence=0.1)]
        result = _fwer_filter(signals)
        assert len(result) == 1

    def test_high_confidence_survives(self):
        signals = [
            ResearchSignal(source="s1", title="high", confidence=0.99),
            ResearchSignal(source="s1", title="mid", confidence=0.7),
            ResearchSignal(source="s1", title="low", confidence=0.3),
        ]
        result = _fwer_filter(signals, alpha=0.05)
        # BH: i=1阈值=1-1/3*0.05=0.9833, 0.99>=0.9833通过(k=1), 0.7<1-2/3*0.05=0.9667不通过
        assert len(result) == 1
        assert result[0].title == "high"

    def test_high_alpha_passes_more(self):
        """大alpha放宽阈值，更多信号通过"""
        signals = [
            ResearchSignal(source="s1", title="a", confidence=0.95),
            ResearchSignal(source="s1", title="b", confidence=0.7),
            ResearchSignal(source="s1", title="c", confidence=0.3),
        ]
        result = _fwer_filter(signals, alpha=0.5)
        # alpha=0.5, i=1阈值=0.833, i=2阈值=0.667, 前两个通过
        assert len(result) >= 2

    def test_multi_source_independent(self):
        """不同来源独立做BH"""
        signals = [
            ResearchSignal(source="s1", title="a", confidence=0.9),
            ResearchSignal(source="s2", title="b", confidence=0.1),
        ]
        result = _fwer_filter(signals, alpha=0.05)
        assert len(result) == 2  # 各自单独一组，都保留

    def test_empty(self):
        assert _fwer_filter([]) == []


class TestResearchSourceBasic:
    def test_init_defaults(self, sample_ohlcv_df):
        rs = ResearchSource(sample_ohlcv_df, cash=50000)
        assert rs.df is not None
        assert rs.cash == 50000

    def test_init_empty(self):
        rs = ResearchSource()
        assert rs.df is None
        assert rs.cash == 100000.0

    def test_calendar_returns_list(self):
        """日历事件不依赖外部数据，返回列表不崩溃"""
        rs = ResearchSource()
        signals = rs.check_calendar_events()
        assert isinstance(signals, list)
        for s in signals:
            assert s.source == "calendar"

    def test_detect_regime_change_empty_df(self):
        """无数据时返回空"""
        rs = ResearchSource()
        signals = rs.detect_regime_change()
        assert signals == []

    def test_detect_regime_change_short_df(self):
        """数据不足60行返回空"""
        import pandas as pd
        import numpy as np
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=30, freq="B"),
            "close": np.linspace(100, 110, 30),
        })
        rs = ResearchSource(df)
        signals = rs.detect_regime_change()
        assert signals == []

    def test_scan_all_without_data(self):
        """无行情数据时scan_all不崩溃"""
        rs = ResearchSource()
        signals = rs.scan_all(use_cache=False)
        assert isinstance(signals, list)

    def test_print_report_empty(self):
        rs = ResearchSource()
        report = rs.print_report([])
        assert "无研究信号" in report

    def test_print_report_with_signals(self):
        rs = ResearchSource()
        signals = [
            ResearchSignal(source="test", title="信号1", priority=3, confidence=0.8),
            ResearchSignal(source="test", title="信号2", priority=5, confidence=0.9),
        ]
        report = rs.print_report(signals)
        assert "信号1" in report
        assert "信号2" in report
        assert "总计" in report

    def test_scan_all_with_data(self, csi300_2023):
        """有行情数据时scan_all不崩溃"""
        rs = ResearchSource(csi300_2023)
        signals = rs.scan_all(use_cache=False)
        assert isinstance(signals, list)
        # 至少有日历事件
        assert len(signals) > 0
        # 验证排序: priority降序
        for i in range(len(signals) - 1):
            assert signals[i].priority >= signals[i + 1].priority

    def test_to_tracks_basic(self):
        rs = ResearchSource()
        signals = [
            ResearchSignal(source="test", title="因子信号", action="factor",
                           priority=4, confidence=0.8, suggested_track="test_track"),
        ]
        tracks = rs.to_tracks(signals)
        assert isinstance(tracks, list)
