"""
BacktestResult — 统一回测结果协议（dataclass）

替代 run_backtest() 返回的裸 dict。
所有分析/报告/验证模块通过此协议消费回测结果，不再依赖隐式 key 约定。

用法
--------
>>> from backtest.engine.result import BacktestResult, TradeRecord
>>> result = BacktestResult(...)
>>> print(result.sharpe)       # 而非 result["sharpe"]
>>> print(result.summary())    # 一行文本概括
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TradeRecord:
    """单笔完整交易记录（买入→卖出配对）"""
    entry_date: str          # 买入日期 YYYY-MM-DD
    exit_date: str           # 卖出日期 YYYY-MM-DD
    entry_price: float
    exit_price: float
    size: int                # 股数/张数
    pnl: float               # 盈亏金额
    pnl_pct: float           # 盈亏百分比
    bars_held: int = 0       # 持仓天数


@dataclass
class BacktestResult:
    """
    回测结果统一协议。

    所有字段都有明确的类型和默认值。消费者无需猜 key、无需 hasattr 防御编程。
    """

    # ── 基础信息 ──
    strategy_name: str = ""
    symbol: str = ""
    start_date: str = ""
    end_date: str = ""
    total_bars: int = 0
    engine: str = "backtrader"

    # ── 收益指标 ──
    start_value: float = 100000.0
    end_value: float = 100000.0
    total_return: float = 0.0
    annual_return: float = 0.0

    # ── 风险指标 ──
    drawdown: float = 0.0           # 最大回撤（小数，如 0.15 = 15%）
    drawdown_duration: int = 0      # 最长回撤持续天数
    sharpe: Optional[float] = None  # 夏普比率
    calmar: Optional[float] = None  # 卡玛比率 = 年化收益/最大回撤
    sortino: Optional[float] = None

    # ── 交易统计 ──
    total_trades: int = 0
    win_trades: int = 0
    loss_trades: int = 0
    win_rate: Optional[float] = None     # 胜率
    avg_win: Optional[float] = None      # 平均盈利
    avg_loss: Optional[float] = None     # 平均亏损
    profit_factor: Optional[float] = None  # 利润因子 = 总盈利/总亏损
    avg_bars_held: float = 0.0

    # ── 详细数据 ──
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)  # [{date, equity}, ...]
    daily_returns: list[float] = field(default_factory=list)

    # ── 元信息 ──
    warnings: list[str] = field(default_factory=list)  # 回测过程中的警告（数据不足等）
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """一行文本概括"""
        s = self.sharpe
        sr = f"{s:.2f}" if s is not None else "N/A"
        return (
            f"[{self.strategy_name}] AR={self.annual_return:.1%} "
            f"DD={self.drawdown:.1%} SR={sr} T={self.total_trades}"
        )

    def is_valid(self) -> bool:
        """是否有足够的交易数据做统计推断"""
        return self.total_trades >= 3 and self.total_bars >= 100

    # Backward-compatible dict key mapping (DRY: single source of truth)
    _DICT_MAP = {
        "total_return": "total_return", "annual_return": "annual_return",
        "drawdown": "drawdown", "sharpe": "sharpe", "calmar": "calmar",
        "total_trades": "total_trades", "won_trades": "win_trades",
        "lost_trades": "loss_trades", "win_rate": "win_rate",
        "final_value": "end_value", "start_value": "start_value",
        "end_value": "end_value", "strategy_name": "strategy_name",
        "profit_factor": "profit_factor", "avg_win": "avg_win",
        "avg_loss": "avg_loss", "symbol": "symbol",
        "start_date": "start_date", "end_date": "end_date",
    }

    def __contains__(self, key: str) -> bool:
        """支持 'key' in result 检查"""
        return key in self._DICT_MAP

    def __getitem__(self, key: str):
        """向后兼容：result['sharpe'] 等同于 result.sharpe"""
        if key in ("trades_df",):
            import pandas as pd
            return pd.DataFrame([t.__dict__ for t in self.trades])
        if key in ("equity_df",):
            import pandas as pd
            return pd.DataFrame(self.equity_curve)
        if key in ("validation",):
            return None
        attr = self._DICT_MAP.get(key)
        if attr is not None:
            return getattr(self, attr, None)
        raise KeyError(key)

    def __setitem__(self, key: str, value):
        """向后兼容：result['key'] = value 等同于 setattr"""
        attr = self._DICT_MAP.get(key, key)
        object.__setattr__(self, attr, value)

    def get(self, key: str, default=None):
        """向后兼容 dict.get()"""
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict:
        """向后兼容：转为 dict（给尚未迁移的旧代码用）"""
        return {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "total_bars": self.total_bars,
            "start_value": self.start_value,
            "end_value": self.end_value,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "drawdown": self.drawdown,
            "sharpe": self.sharpe,
            "calmar": self.calmar,
            "total_trades": self.total_trades,
            "win_trades": self.win_trades,
            "loss_trades": self.loss_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "equity_curve": self.equity_curve,
            "trades": [t.__dict__ if hasattr(t, '__dict__') else t for t in self.trades],
            "warnings": self.warnings,
        }
