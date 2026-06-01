"""社区策略库 — 来自 vnpy、Freqtrade、聚宽、Backtrader 社区的经典策略。"""
from backtest.strategies.community.vnpy_double_ma import VnpyDoubleMaStrategy
from backtest.strategies.community.turtle_trading import TurtleTradingStrategy
from backtest.strategies.community.freqtrade_rsi import FreqtradeRSIStrategy
from backtest.strategies.community.joinquant_macd import JoinQuantMACDStrategy
from backtest.strategies.community.donchian_channel import DonchianChannelStrategy
from backtest.strategies.community.atr_trailing_stop import ATRTrailingStopStrategy

MARKET_STRATEGIES = {
    "vnpy双均线": {
        "class": VnpyDoubleMaStrategy,
        "source": "imported",
        "source_url": "vnpy 社区 / github.com/vnpy/vnpy",
        "desc": "经典双均线交叉——量化入门第一课",
        "params": {"fast": 5, "slow": 20},
        "type": "趋势跟踪",
    },
    "海龟交易": {
        "class": TurtleTradingStrategy,
        "source": "imported",
        "source_url": "Richard Dennis, 1983",
        "desc": "突破20日高点买入，跌破10日低点卖出，盈利后加仓",
        "params": {"entry_period": 20, "exit_period": 10, "atr_period": 20},
        "type": "趋势跟踪",
    },
    "Freqtrade RSI": {
        "class": FreqtradeRSIStrategy,
        "source": "imported",
        "source_url": "Freqtrade 社区 / github.com/freqtrade/freqtrade",
        "desc": "RSI<30超卖买入，RSI>70超买卖出",
        "params": {"rsi_period": 14, "oversold": 30, "overbought": 70},
        "type": "均值回归",
    },
    "聚宽MACD": {
        "class": JoinQuantMACDStrategy,
        "source": "imported",
        "source_url": "聚宽社区 / joinquant.com",
        "desc": "MACD金叉买入，死叉卖出——聚宽热门策略",
        "params": {"fast": 12, "slow": 26, "signal": 9},
        "type": "趋势跟踪",
    },
    "Donchian通道": {
        "class": DonchianChannelStrategy,
        "source": "imported",
        "source_url": "Backtrader 社区",
        "desc": "突破N日最高价买入，跌破N日最低价卖出",
        "params": {"period": 20},
        "type": "趋势跟踪",
    },
    "ATR移动止损": {
        "class": ATRTrailingStopStrategy,
        "source": "imported",
        "source_url": "量化经典模式",
        "desc": "均线入场 + ATR动态止损（波动大放宽，波动小收紧）",
        "params": {"ma_fast": 10, "ma_slow": 30, "atr_period": 14, "atr_mult": 2.0},
        "type": "趋势跟踪",
    },
}
