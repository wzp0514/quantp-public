"""
策略适配器 — 将 fep 格式策略包装为引擎相关策略类

支持:
  - Backtrader: UniversalStrategy(bt.Strategy) 包装 compute()
  - VectorBT: VectorBTAdapter 包装 compute() → entries/exits 数组（预留）

fep 格式策略:
  - fep.yaml: 元数据（名称/描述/参数/分类）
  - strategy.py: 纯函数 compute(data, context) → {action, ...}
"""

import logging
from pathlib import Path
from typing import Optional

import backtrader as bt
import pandas as pd
import yaml

from config.log import get_logger

logger = get_logger("strategy_adapter")


def load_strategy_from_package(package_dir: str) -> tuple[dict, callable]:
    """
    从 fep 格式策略包加载元数据和 compute 函数。

    参数
    ----------
    package_dir : str
        策略包目录，含 fep.yaml 和 strategy.py

    返回
    -------
    (meta: dict, compute: callable)
    """
    package_path = Path(package_dir)
    if not package_path.exists():
        raise FileNotFoundError(f"策略包不存在: {package_dir}")

    # 加载元数据
    fep_path = package_path / "fep.yaml"
    if not fep_path.exists():
        raise FileNotFoundError(f"fep.yaml 不存在: {fep_path}")

    with open(fep_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    # 加载 compute 函数
    strategy_path = package_path / meta.get("technical", {}).get("entryPoint", "strategy.py")
    if not strategy_path.exists():
        raise FileNotFoundError(f"策略入口不存在: {strategy_path}")

    import importlib.util
    spec = importlib.util.spec_from_file_location("_fep_strategy", strategy_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "compute"):
        raise AttributeError(f"策略 {strategy_path} 中缺少 compute 函数")

    return meta, module.compute


class UniversalStrategy(bt.Strategy):
    """
    把 compute(data, context) 包装成 Backtrader Strategy。

    用法
    --------
    >>> cerebro.addstrategy(UniversalStrategy,
    ...     compute_func=my_compute,
    ...     params={"fast": 5, "slow": 20},
    ... )
    """

    params = (
        ("compute_func", None),
        ("strategy_params", {}),
        ("a_share_mode", True),
        ("stop_loss_pct", 0.05),
    )

    def __init__(self):
        self.trades = []  # 交易记录（兼容 bt_runner 的 trades_df 生成）
        self._data_buffer = []
        self._strategy_state = {}  # 跨 Bar 持久状态，compute() 可通过 context["state"] 读写
        self._context = {
            "cash": self.broker.getcash(),
            "equity": self.broker.getvalue(),
            "position": None,
            "params": dict(self.p.strategy_params),
            "state": self._strategy_state,
        }
        self.log("UniversalStrategy 初始化 (fep 格式)")

    def log(self, msg: str):
        try:
            dt = self.datas[0].datetime.date(0)
            logger.info(f"[{dt}] Universal: {msg}")
        except (IndexError, AttributeError):
            logger.info(f"[INIT] Universal: {msg}")

    def next(self):
        # 累积当前 Bar 数据
        self._data_buffer.append({
            "date": self.datas[0].datetime.date(0),
            "open": self.data.open[0],
            "high": self.data.high[0],
            "low": self.data.low[0],
            "close": self.data.close[0],
            "volume": self.data.volume[0],
        })
        df = pd.DataFrame(self._data_buffer)

        # 更新 context（state 是引用，自动跨 Bar 持久）
        self._context["cash"] = self.broker.getcash()
        self._context["equity"] = self.broker.getvalue()
        self._context["params"] = dict(self.p.strategy_params)
        if self.position:
            self._context["position"] = {
                "size": self.position.size,
                "entry_price": self.position.price,
            }
        else:
            self._context["position"] = None

        # 调用 compute
        compute_func = self.p.compute_func
        if compute_func is None:
            return

        try:
            decision = compute_func(df, self._context)
        except Exception as e:
            logger.warning(f"compute 执行异常: {e}")
            return

        if not isinstance(decision, dict):
            return

        action = decision.get("action", "hold")

        if action == "buy":
            amount = decision.get("amount", 0)
            if amount > 0 and not self.position:
                price = decision.get("price", self.data.close[0])
                order = self.buy(size=amount)
                if order:
                    self.trades.append({
                        "date": self.datas[0].datetime.date(0),
                        "type": "buy",
                        "size": amount,
                        "price": price,
                        "value": amount * price,
                        "commission": amount * price * 0.00025,
                    })
                self.log(f"买入 {amount} 股: {decision.get('reason', '')}")

        elif action == "sell":
            if self.position:
                pct = decision.get("percent", 1.0)
                size = int(self.position.size * pct)
                if size > 0:
                    price = self.data.close[0]
                    order = self.sell(size=size)
                    if order:
                        self.trades.append({
                            "date": self.datas[0].datetime.date(0),
                            "type": "sell",
                            "size": size,
                            "price": price,
                            "value": size * price,
                            "commission": size * price * 0.00025 + size * price * 0.0005,
                        })
                    self.log(f"卖出 {size} 股: {decision.get('reason', '')}")


def run_backtest_universal(
    package_dir: str,
    df: pd.DataFrame,
    initial_cash: float = 100000.0,
    **strategy_params,
) -> dict:
    """
    用 fep 格式策略运行回测（零依赖策略类导入）。

    参数
    ----------
    package_dir : str
        策略包目录路径
    df : pd.DataFrame
        行情数据
    initial_cash : float
        初始资金
    **strategy_params :
        传递给 compute 的参数

    返回
    -------
    同 bt_runner.run_backtest()
    """
    from backtest.engine.bt_runner import run_backtest

    meta, compute_func = load_strategy_from_package(package_dir)

    strategy_params = strategy_params or {}
    defaults = meta.get("backtest", {})
    if not strategy_params:
        # 从 fep.yaml 的 identity 推断默认参数
        pass

    return run_backtest(
        UniversalStrategy,
        df,
        initial_cash=initial_cash or defaults.get("initialCapital", 100000),
        compute_func=compute_func,
        strategy_params=strategy_params,
    )


# ── VectorBT 适配器 ─────────────────────────────────────────

class VectorBTAdapter:
    """
    将 FEP compute() 转换为 VectorBT 可用的 entries/exits 信号数组。

    通过逐 Bar 回放 compute()，累积有状态决策，生成布尔信号数组。

    用法
    --------
    >>> adapter = VectorBTAdapter(compute_func, params={"fast": 5, "slow": 20})
    >>> entries, exits = adapter.to_signals(df)
    >>> portfolio = vbt.Portfolio.from_signals(close, entries, exits)
    """

    def __init__(self, compute_func: callable, params: dict = None,
                 initial_cash: float = 100000.0):
        self.compute_func = compute_func
        self.params = params or {}
        self.initial_cash = initial_cash

    def to_signals(self, df: pd.DataFrame) -> tuple:
        """
        将 FEP 策略转为 entry/exit 信号数组。

        返回
        -------
        (entries: np.ndarray[bool], exits: np.ndarray[bool])
        """
        n = len(df)
        entries = np.zeros(n, dtype=bool)
        exits = np.zeros(n, dtype=bool)

        cash = self.initial_cash
        position = None
        equity = self.initial_cash
        state = {}
        data_buffer = []

        for i in range(n):
            data_buffer.append({
                "date": df.index[i] if hasattr(df.index[i], 'date') else str(df.index[i]),
                "open": df["open"].iloc[i],
                "high": df["high"].iloc[i],
                "low": df["low"].iloc[i],
                "close": df["close"].iloc[i],
                "volume": df["volume"].iloc[i] if "volume" in df.columns else 0,
            })
            sub_df = pd.DataFrame(data_buffer)

            ctx = {
                "cash": cash,
                "equity": equity,
                "position": position,
                "params": self.params,
                "state": state,
            }

            try:
                decision = self.compute_func(sub_df, ctx)
            except Exception:
                continue

            state = ctx.get("state", {})
            action = decision.get("action", "hold")

            if action == "buy" and position is None:
                amount = decision.get("amount", 0)
                price = decision.get("price", df["close"].iloc[i])
                if amount > 0 and cash >= amount * price:
                    entries[i] = True
                    position = {"size": amount, "entry_price": price}
                    cash -= amount * price
                    equity = cash + amount * price

            elif action == "sell" and position is not None:
                pct = decision.get("percent", 1.0)
                size = int(position["size"] * pct)
                amount = decision.get("amount", size)
                if amount > 0 and position["size"] >= amount:
                    exits[i] = True
                    price = df["close"].iloc[i]
                    cash += amount * price
                    position["size"] -= amount
                    if position["size"] <= 0:
                        position = None
                    equity = cash

        return entries, exits


# ── 命令行测试 ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    pkg = "strategies_repo/custom/ma_cross"
    meta, compute_func = load_strategy_from_package(pkg)
    print(f"策略: {meta['identity']['name']}")
    print(f"描述: {meta['identity']['description']}")
    print(f"参数上限: {meta['classification']['maxParams']}")
    print(f"compute 函数: {compute_func.__name__}")

    # 快速单元测试：假数据验证 compute() 逻辑
    import numpy as np
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    price = 100 + np.cumsum(np.random.randn(30) * 2)
    test_data = pd.DataFrame({
        "date": dates,
        "open": price * 0.99,
        "high": price * 1.02,
        "low": price * 0.98,
        "close": price,
        "volume": np.random.randint(1000, 10000, 30),
    })
    ctx = {"cash": 100000, "equity": 100000, "position": None, "params": {"fast": 5, "slow": 20}}
    result = compute_func(test_data, ctx)
    print(f"决策: {result}")
    print("适配器 OK")
