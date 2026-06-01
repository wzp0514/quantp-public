"""
通知模块单元测试 — 测试消息模板构建 + 渠道选择逻辑
不实际发送 HTTP 请求，只测消息格式和路由逻辑
"""
import pytest
from unittest.mock import patch, MagicMock
from live.gateway.notifier import (
    alert_trade, alert_risk, alert_error, alert_daily_summary,
    send, _load_notify_config,
)


class TestAlertTemplates:
    def test_alert_trade_buy(self):
        msg = alert_trade("600519", "buy", 100, 1500.0)
        assert "[BUY]" in msg
        assert "600519" in msg
        assert "买入" in msg
        assert "1500" in msg or "1500.0000" in msg

    def test_alert_trade_sell_with_pnl(self):
        msg = alert_trade("600519", "sell", 100, 1600.0, pnl=10000.0)
        assert "[SELL]" in msg
        assert "卖出" in msg
        assert "+10000" in msg or "10000" in msg

    def test_alert_risk_structure(self):
        msg = alert_risk("单笔亏损超限", "亏损 2.5% > 上限 2.0%")
        assert "[ALERT]" in msg
        assert "风控告警" in msg
        assert "单笔亏损超限" in msg
        assert "已自动停止交易" in msg

    def test_alert_error_structure(self):
        msg = alert_error("API中断", "连接超时", positions="600519: 100股")
        assert "[ERROR]" in msg
        assert "API中断" in msg
        assert "持仓" in msg

    def test_alert_error_no_positions(self):
        msg = alert_error("网络异常", "DNS解析失败")
        assert "[ERROR]" in msg

    def test_alert_daily_summary_structure(self):
        msg = alert_daily_summary(0.015, -50.0, 3, 5, 0)
        assert "[DAILY]" in msg
        assert "1.5" in msg or "1.50" in msg

    def test_alert_daily_summary_with_warnings(self):
        msg = alert_daily_summary(0.015, -50.0, 3, 5, 2)
        assert "风控预警" in msg


class TestChannelRouting:
    def test_send_dingtalk_no_config_returns_false(self):
        with patch("live.gateway.notifier._load_notify_config",
                   return_value={}):
            ok = send("test", channel="dingtalk")
            assert ok is False

    def test_send_auto_falls_through_all(self):
        """auto 模式尝试所有渠道，全未配置则返回最终结果"""
        with patch("live.gateway.notifier._load_notify_config",
                   return_value={}):
            ok = send("test", channel="auto")
            assert ok is False

    def test_send_unknown_channel_falls_to_auto(self):
        with patch("live.gateway.notifier._load_notify_config",
                   return_value={}):
            ok = send("test", channel="unknown")
            assert ok is False

    def test_send_feishu_no_config(self):
        with patch("live.gateway.notifier._load_notify_config",
                   return_value={}):
            ok = send("test", channel="feishu")
            assert ok is False

    def test_send_pushplus_no_config(self):
        with patch("live.gateway.notifier._load_notify_config",
                   return_value={}):
            ok = send("test", channel="pushplus")
            assert ok is False

    def test_send_serverchan_no_config(self):
        with patch("live.gateway.notifier._load_notify_config",
                   return_value={}):
            ok = send("test", channel="serverchan")
            assert ok is False

    def test_send_telegram_no_config(self):
        with patch("live.gateway.notifier._load_notify_config",
                   return_value={}):
            ok = send("test", channel="telegram")
            assert ok is False

    def test_send_wecom_no_config(self):
        with patch("live.gateway.notifier._load_notify_config",
                   return_value={}):
            ok = send("test", channel="wecom")
            assert ok is False


class TestNotifyConfig:
    def test_load_config_returns_dict(self):
        cfg = _load_notify_config()
        assert isinstance(cfg, dict)
