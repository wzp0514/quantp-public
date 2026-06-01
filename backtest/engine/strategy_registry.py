"""
策略注册中心 — 统一管理所有可运行策略的注册、发现、加载。

职责：
  1. 维护 ALL_STRATEGIES 字典（所有策略的单一数据源）
  2. 策略分类常量（STRATEGY_TYPES / SOURCE_LABELS）
  3. FEP 格式策略的动态加载与适配
"""

import logging

from config.log import get_logger
from config.loader import is_crypto_enabled

logger = get_logger("strategy_registry")

from backtest.strategies.community import (
    VnpyDoubleMaStrategy,
    TurtleTradingStrategy,
    FreqtradeRSIStrategy,
    JoinQuantMACDStrategy,
    DonchianChannelStrategy,
    ATRTrailingStopStrategy,
    MARKET_STRATEGIES,
)
from backtest.strategies.builtin.ma_cross import MaCrossStrategy
from backtest.strategies.builtin.bollinger import BollingerStrategy
from backtest.strategies.builtin.momentum import MomentumStrategy
from backtest.strategies.builtin.mean_revert import MeanRevertStrategy
from backtest.strategies.builtin.grid import GridStrategy

# ============================================================
# 策略类型常量
# ============================================================
STRATEGY_TYPES = {
    "趋势跟踪": "趋势跟踪",
    "均值回归": "均值回归",
    "震荡": "震荡",
    "突破": "突破",
    "多因子": "多因子",
    "共振": "共振",
    "实验性": "实验性",
}

SOURCE_LABELS = {
    "builtin": "外来策略",
    "imported": "外来策略",
    "manual": "外来策略",
    "mined": "挖掘策略",
    "custom": "自建策略",
    "experimental": "自建策略",
}

# ============================================================
# 内置策略
# ============================================================
BUILTIN_STRATEGIES = {
    "双均线交叉": {
        "class": MaCrossStrategy,
        "source": "builtin",
        "source_url": "",
        "desc": "快线上穿慢线买入，下穿卖出",
        "params": {"fast": 5, "slow": 20},
        "type": "趋势跟踪",
    },
    "布林带回归": {
        "class": BollingerStrategy,
        "source": "builtin",
        "source_url": "",
        "desc": "触下轨买入，回中轨卖出",
        "params": {"period": 20, "devfactor": 2.0},
        "type": "均值回归",
    },
    "动量策略": {
        "class": MomentumStrategy,
        "source": "builtin",
        "source_url": "",
        "desc": "买过去涨最多的，持有到期卖出",
        "params": {"lookback": 126, "hold_period": 21, "threshold": 0.03},
        "type": "趋势跟踪",
    },
    "均值回归": {
        "class": MeanRevertStrategy,
        "source": "builtin",
        "source_url": "",
        "desc": "超跌买入，回归均线卖出",
        "params": {"period": 20, "threshold": 2.0},
        "type": "均值回归",
    },
    "网格交易": {
        "class": GridStrategy,
        "source": "builtin",
        "source_url": "",
        "desc": "价格区间内低买高卖赚波动",
        "params": {"grid_count": 10, "upper_pct": 0.10, "lower_pct": -0.10},
        "type": "震荡",
    },
}

# ============================================================
# FEP 引擎无关策略
# ============================================================
FEP_STRATEGIES = {
    "双均线交叉(fep)": {
        "fep_package": "strategies_repo/custom/ma_cross",
        "source": "manual",
        "source_url": "quantp/fep",
        "desc": "[fep] 快线上穿慢线买入——引擎无关格式",
        "params": {"fast": 5, "slow": 20},
        "type": "趋势跟踪",
    },
    "布林带回归(fep)": {
        "fep_package": "strategies_repo/custom/bollinger",
        "source": "builtin",
        "source_url": "",
        "desc": "[fep] 触下轨超卖买入，回中轨卖出",
        "params": {"period": 20, "devfactor": 2.0},
        "type": "均值回归",
    },
    "动量策略(fep)": {
        "fep_package": "strategies_repo/custom/momentum",
        "source": "builtin",
        "source_url": "",
        "desc": "[fep] 过去N月涨超阈值买入，持M月卖出",
        "params": {"lookback": 126, "hold_period": 21, "threshold": 0.03},
        "type": "趋势跟踪",
    },
    "均值回归(fep)": {
        "fep_package": "strategies_repo/custom/mean_revert",
        "source": "builtin",
        "source_url": "",
        "desc": "[fep] Z-score超跌买入，回归均值卖出",
        "params": {"period": 20, "threshold": 2.0},
        "type": "均值回归",
    },
    "网格交易(fep)": {
        "fep_package": "strategies_repo/custom/grid",
        "source": "builtin",
        "source_url": "",
        "desc": "[fep] 价格区间低买高卖赚波动",
        "params": {"grid_count": 10, "upper_pct": 0.10, "lower_pct": -0.10},
        "type": "震荡",
    },
    "vnpy双均线(fep)": {
        "fep_package": "strategies_repo/custom/vnpy_double_ma",
        "source": "imported",
        "source_url": "vnpy 社区 / github.com/vnpy/vnpy",
        "desc": "[fep] 经典双均线交叉，金叉买入死叉卖出",
        "params": {"fast": 5, "slow": 20},
        "type": "趋势跟踪",
    },
    "海龟交易(fep)": {
        "fep_package": "strategies_repo/custom/turtle_trading",
        "source": "imported",
        "source_url": "Richard Dennis, 1983",
        "desc": "[fep] 突破20日高点买入，跌破10日低点卖出，金字塔加仓",
        "params": {"entry_period": 20, "exit_period": 10, "atr_period": 20},
        "type": "突破",
    },
    "Freqtrade RSI(fep)": {
        "fep_package": "strategies_repo/custom/freqtrade_rsi",
        "source": "imported",
        "source_url": "Freqtrade 社区 / github.com/freqtrade/freqtrade",
        "desc": "[fep] RSI<30超卖买入，RSI>70超买卖出",
        "params": {"rsi_period": 14, "oversold": 30, "overbought": 70},
        "type": "均值回归",
    },
    "聚宽MACD(fep)": {
        "fep_package": "strategies_repo/custom/joinquant_macd",
        "source": "imported",
        "source_url": "聚宽社区 / joinquant.com",
        "desc": "[fep] MACD金叉买入，死叉卖出",
        "params": {"fast": 12, "slow": 26, "signal": 9},
        "type": "趋势跟踪",
    },
    "Donchian通道(fep)": {
        "fep_package": "strategies_repo/custom/donchian_channel",
        "source": "imported",
        "source_url": "Backtrader 社区",
        "desc": "[fep] 突破N日最高价买入，跌破N日最低价卖出",
        "params": {"period": 20},
        "type": "突破",
    },
    "ATR移动止损(fep)": {
        "fep_package": "strategies_repo/custom/atr_trailing_stop",
        "source": "imported",
        "source_url": "量化经典模式",
        "desc": "[fep] 均线入场 + ATR动态止损",
        "params": {"ma_fast": 10, "ma_slow": 30, "atr_period": 14, "atr_mult": 2.0},
        "type": "趋势跟踪",
    },
}

# ============================================================
# 高级/实验性策略
# ============================================================
_ADVANCED_STRATEGIES = {}
try:
    from backtest.strategies.experimental.resonance import ResonanceStrategy
    _ADVANCED_STRATEGIES["多信号共振"] = {
        "class": ResonanceStrategy,
        "source": "experimental",
        "source_url": "",
        "desc": "多信号共振+区制过滤+因子嵌入，减少假信号",
        "params": {"require_signals": 2},
        "type": "共振",
    }
except ImportError:
    pass

try:
    from backtest.strategies.experimental.factor_strategy import FactorStrategy
    _ADVANCED_STRATEGIES["因子驱动策略"] = {
        "class": FactorStrategy,
        "source": "experimental",
        "source_url": "",
        "desc": "因子得分驱动交易，支持波动率目标仓位",
        "params": {},
        "type": "多因子",
    }
except ImportError:
    pass

try:
    from backtest.strategies.experimental.cross_section_strategy import CrossSectionStrategy
    _ADVANCED_STRATEGIES["截面多因子选股"] = {
        "class": CrossSectionStrategy,
        "source": "experimental",
        "source_url": "",
        "desc": "多股票面板截面排名，买Top N定期调仓",
        "params": {"top_n": 5, "rebalance_freq": 21},
        "type": "多因子",
    }
except ImportError:
    pass

# ============================================================
# Freqtrade 移植策略
# ============================================================
_FREQTRADE_PORT_STRATEGIES = {}
try:
    from backtest.strategies.imported.freqtrade_port import (
        VIDYACrossStrategy, STCStrategy, KelterBreakoutStrategy,
    )
    _FREQTRADE_PORT_STRATEGIES = {
        "VIDYA自适应均线": {
            "class": VIDYACrossStrategy,
            "source": "imported",
            "source_url": "Freqtrade 社区 / github.com/freqtrade/freqtrade",
            "desc": "Variable Index Dynamic Average 自适应交叉，CMO动态调平滑度",
            "params": {"vidya_period": 14, "smoothing": 9},
            "type": "趋势跟踪",
        },
        "STC趋势转折": {
            "class": STCStrategy,
            "source": "imported",
            "source_url": "Freqtrade 社区 / Schaff Trend Cycle",
            "desc": "Schaff Trend Cycle，MACD+Stochastic双重平滑，超卖<25买入/超买>75卖出",
            "params": {"stc_fast": 23, "stc_slow": 50, "stc_cycle": 10},
            "type": "均值回归",
        },
        "Keltner突破": {
            "class": KelterBreakoutStrategy,
            "source": "imported",
            "source_url": "Freqtrade 社区 / Triple Keltner Channel",
            "desc": "凯尔特纳通道突破，上轨突破买入/下轨跌破卖出",
            "params": {"ke_period": 20, "ke_atr": 10, "ke_mult": 1.5},
            "type": "突破",
        },
    }
except ImportError:
    pass

# ============================================================
# 合并所有策略 → ALL_STRATEGIES
# ============================================================
_ALL = {**FEP_STRATEGIES, **MARKET_STRATEGIES, **BUILTIN_STRATEGIES, **_ADVANCED_STRATEGIES, **_FREQTRADE_PORT_STRATEGIES}
if not is_crypto_enabled():
    _ALL = {k: v for k, v in _ALL.items()
            if "freqtrade" not in v.get("source_url", "").lower()}

# 解析 fep 格式策略的 class（通过适配器动态加载）
for _name, _info in list(_ALL.items()):
    if "class" not in _info:
        try:
            from backtest.engine.strategy_adapter import load_strategy_from_package
            _, _compute = load_strategy_from_package(_info["fep_package"])
            from backtest.engine.strategy_adapter import UniversalStrategy

            class _FepStrategy(UniversalStrategy):
                _fep_compute = _compute

            _info["class"] = _FepStrategy
            _info["_fep_loaded"] = True
            logger.info(f"fep 策略已解析: {_name} → UniversalStrategy")
        except Exception as e:
            logger.warning(f"fep 策略解析失败: {_name} ({e})，已从列表中排除")
            del _ALL[_name]

ALL_STRATEGIES = _ALL

__all__ = [
    "ALL_STRATEGIES", "BUILTIN_STRATEGIES", "FEP_STRATEGIES",
    "MARKET_STRATEGIES", "STRATEGY_TYPES", "SOURCE_LABELS",
]
