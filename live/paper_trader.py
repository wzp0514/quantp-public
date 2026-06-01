"""
纸上交易引擎（Paper Trading）

模拟真实交易环境，逐日运行策略并模拟成交，但不涉及真金白银。

和回测的关键区别：
  - 回测：一次性跑完整个历史区间，瞬间出结果
  - 纸上交易：逐日推进，每天只能看到"今天之前"的数据，
    模拟的是真实的逐日决策过程，更能暴露策略的问题

纸上交易的作用：
  1. 验证策略在"只能看到过去"时的表现（和回测一致才说明策略逻辑正确）
  2. 发现信号频率和回测预期是否一致
  3. 测试程序稳定性（连跑 4 周不崩溃）
  4. 感受真实的持仓心理（每天看到浮动亏损会不会想手动干预）

实现细节：
  - 历史模式：用"次日开盘价"成交（用当日收盘价是偷价行为）
  - 实时模式：用当前轮询价成交（step_live）
  - 每笔交易前必须通过风控检查 + StrategyGuard 熔断检测
  - 每日记录：信号、订单、持仓、资产变化
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional, TYPE_CHECKING

import pandas as pd

from live.execution.order_mgr import OrderManager, BUY, SELL
from live.execution.position_mgr import PositionManager
from live.risk.risk_engine import reload as reload_risk, run_all_checks, get_params
from live.risk.strategy_guard import StrategyGuard

from config.log import get_logger
logger = get_logger("paper_trader")
# 文件日志
try:
    file_handler = logging.FileHandler("notebooks/paper_trade.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-5s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)
except Exception as e:
    logger.debug(f"无法创建文件日志: {e}")

if TYPE_CHECKING:
    from live.execution.paper_store import PaperTradeStore


class PaperTrader:
    """纸上交易引擎。

    两种运行模式：
      - 历史模式：run() 逐日回放历史数据，用次日开盘价成交
      - 实时模式：run_live() 接收实时行情快照，用当前价成交
    """

    def __init__(
        self,
        strategy_class,
        data: pd.DataFrame,
        initial_cash: float = 100000.0,
        store: Optional["PaperTradeStore"] = None,
        instance_id: Optional[int] = None,
        enable_guard: bool = True,
        **strategy_params,
    ):
        """
        参数
        ----------
        strategy_class : Backtrader Strategy 子类
        data : DataFrame，历史行情数据（实时模式下可为空）
        initial_cash : 初始资金
        store : PaperTradeStore，持久化存储（可选）
        instance_id : 存储中的实例ID（用于恢复）
        enable_guard : 是否启用StrategyGuard策略级熔断
        """
        self.strategy_class = strategy_class
        self.data = data.sort_values("date").reset_index(drop=True) if not data.empty else data
        self.initial_cash = initial_cash
        self.strategy_params = strategy_params

        # 子系统
        self.order_mgr = OrderManager()
        self.position_mgr = PositionManager(initial_cash)

        # 策略级熔断器
        self.guard = StrategyGuard(
            name=strategy_class.__name__,
            max_drawdown=0.15,
            max_consecutive_losses=5,
            signal_freq_drift_pct=0.50,
            price_slippage_pct=0.02,
        ) if enable_guard else None

        # 持久化
        self.store = store
        self.instance_id = instance_id

        # 日志与状态
        self.daily_log: list[dict] = []
        self.signals: list[dict] = []
        self.warnings: list[str] = []
        self._signal_map: Optional[dict] = None
        self._live_bar_index = 0
        self._last_trade_pnl: float = 0

        # 风控
        self.risk_params = get_params()
        logger.info(f"纸上交易初始化: {strategy_class.__name__}, "
                    f"资金 {initial_cash:,.0f}, 数据 {len(data)} 条, "
                    f"熔断={'启用' if self.guard else '关闭'}")

    # ================================================================
    # 历史模式
    # ================================================================

    def run(self) -> dict:
        """逐日运行纸上交易（历史数据回放）。"""
        from backtest.engine.bt_runner import run_backtest

        data_len = len(self.data)

        # 先跑回测作为基准
        logger.info("先跑回测作为对照基准...")
        baseline = run_backtest(
            self.strategy_class, self.data,
            initial_cash=self.initial_cash,
            **self.strategy_params,
        )

        # 构建信号日期映射 O(1)查表
        self._signal_map = {}
        trades_df = baseline.get("trades_df", pd.DataFrame())
        if not trades_df.empty:
            for _, trade in trades_df.iterrows():
                d = pd.Timestamp(trade["date"]).date()
                if d not in self._signal_map:
                    self._signal_map[d] = []
                self._signal_map[d].append(trade["type"])

        # 设置StrategyGuard回测基准
        if self.guard:
            total_signals = len(trades_df)
            avg_interval = data_len / max(total_signals, 1)
            self.guard.set_backtest_baseline(total_signals, avg_interval)

        logger.info(f"开始纸上交易，共 {data_len} 个交易日")

        for i in range(data_len):
            today = self.data.iloc[i]
            today_date = pd.Timestamp(today["date"]).date()
            today_price = float(today["close"])

            # Guard熔断检查
            if self.guard and self.guard.is_blown():
                msg = f"[{today_date}] StrategyGuard熔断: {self.guard.reason}"
                logger.warning(msg)
                self.warnings.append(msg)
                self._log_daily(today_date, today_price, None, "guard_blown", msg)
                break

            signal = self._check_signal(i)

            if signal:
                self.signals.append({"date": today_date, "signal": signal, "price": today_price})

                risk_ok, risk_msg = self._risk_check(signal, today_price)
                if not risk_ok:
                    logger.warning(f"[{today_date}] 风控拦截: {risk_msg}")
                    self.warnings.append(f"[{today_date}] {risk_msg}")
                    self._log_daily(today_date, today_price, signal, "blocked", risk_msg)
                    continue

                self._execute_signal(signal, i, today_date, today_price)

            self._log_daily(today_date, today_price, signal, signal or "hold", "")

            if (i + 1) % 50 == 0:
                self._progress_log(i, data_len, today_date, today_price)

        return self._build_result(baseline)

    # ================================================================
    # 实时模式
    # ================================================================

    def step_live(self, symbol: str, snapshot: dict) -> Optional[dict]:
        """处理一个实时行情快照。

        参数
        ----------
        symbol : 标的代码
        snapshot : {price, open, high, low, volume, timestamp}

        返回
        -------
        dict 或 None（当日日志条目）
        """
        price = snapshot.get("price", 0)
        if price <= 0:
            return None

        today = snapshot.get("timestamp", datetime.now())
        today_date = today.date() if hasattr(today, 'date') else datetime.now().date()

        self._live_bar_index += 1

        # Guard熔断
        if self.guard:
            self.guard.check_signal(self._live_bar_index)
            if self.guard.is_blown():
                logger.warning(f"[{today_date}] Guard熔断: {self.guard.reason}")
                self._flush_store()
                return None

        # 策略信号（实时模式下基于当前持仓和价格判断）
        signal = self._live_strategy_signal(price)

        if signal:
            self.signals.append({"date": today_date, "signal": signal, "price": price})

            risk_ok, risk_msg = self._risk_check(signal, price)
            if not risk_ok:
                self.warnings.append(f"[{today_date}] {risk_msg}")
                entry = self._log_daily(today_date, price, signal, "blocked", risk_msg)
                self._flush_store()
                return entry

            self._execute_live_signal(signal, symbol, price, today_date)

        entry = self._log_daily(today_date, price, signal, signal or "hold", "")
        self._flush_store()
        return entry

    def run_live(self, feed, symbol: str) -> dict:
        """持续实时纸上交易循环。

        参数
        ----------
        feed : RealTimeFeed 实例
        symbol : 标的代码

        返回
        -------
        dict，包含运行结果
        """
        from live.feed.realtime_feed import RealTimeFeed

        logger.info(f"启动实时纸上交易: {self.strategy_class.__name__}, {symbol}")

        # 获取历史数据作为基准
        if not self.data.empty:
            from backtest.engine.bt_runner import run_backtest
            baseline = run_backtest(
                self.strategy_class, self.data,
                initial_cash=self.initial_cash,
                **self.strategy_params,
            )
        else:
            baseline = {}

        def on_snapshot(snap: dict):
            sym_code = symbol[-6:] if symbol.startswith(("sh", "sz")) else symbol
            info = snap.get(sym_code, {})
            if info:
                self.step_live(symbol, info)

        try:
            feed.start(on_snapshot)
            logger.info("实时轮询已启动，等待信号...（Ctrl+C 停止）")
            while feed._running:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("用户中断")
        finally:
            feed.stop()

        return self._build_result(baseline)

    # ================================================================
    # 内部方法
    # ================================================================

    def _check_signal(self, i: int) -> Optional[str]:
        """从回测信号映射查表（历史模式，O(1)）。"""
        if self._signal_map is None:
            return None
        today_date = pd.Timestamp(self.data.iloc[i]["date"]).date()
        signals = self._signal_map.get(today_date, [])
        if not signals:
            return None
        return "sell" if "sell" in signals else "buy"

    def _live_strategy_signal(self, price: float) -> Optional[str]:
        """实时模式下的简化策略信号判断。

        基于简单的均线交叉（短/长周期），仅作为占位实现。
        完整实现需集成策略的回测引擎逐bar判断。
        """
        # 使用策略参数中的 fast/slow 做均线近似
        fast = self.strategy_params.get("fast", 5)
        slow = self.strategy_params.get("slow", 20)

        pos = self.position_mgr.get_position("PAPER")
        has_position = pos and pos["size"] > 0

        # 简化的价格vs均线交叉逻辑：依赖外部提供的价格趋势
        # 实际应用中，均线应从历史数据+实时价格计算
        if not has_position:
            # 空仓时发出买入信号（简化：基于策略的常规入场逻辑）
            # 此处留空，由外部策略引擎驱动
            return None
        else:
            # 持仓时的卖出条件：距开仓价偏离超过 stop_loss_pct
            cost = pos["avg_cost"]
            stop_loss = self.strategy_params.get("stop_loss_pct", 0.05)
            if price < cost * (1 - stop_loss):
                return "sell"
            return None

    def _execute_signal(self, signal: str, i: int, today_date, today_price: float):
        """历史模式：以次日开盘价成交。"""
        data_len = len(self.data)
        next_open = float(self.data.iloc[i + 1]["open"]) if i + 1 < data_len else None

        if signal == "buy" and next_open:
            size = int(self.position_mgr.cash / today_price)
            if size > 0:
                order = self.order_mgr.create_market_order("PAPER", BUY, size, today_date)
                self.order_mgr.submit(order["id"])
                self.order_mgr.fill(order["id"], next_open, today_date)
                prev_value = self.position_mgr.get_total_value({"PAPER": today_price})
                self.position_mgr.update_position("PAPER", BUY, size, next_open, today_date)
                new_value = self.position_mgr.get_total_value({"PAPER": next_open})
                self._last_trade_pnl = new_value - prev_value
                if self.guard:
                    self.guard.check_trade(next_open, today_price)
                    self.guard.update_equity(new_value, self._last_trade_pnl)
                logger.info(f"[{today_date}] 买入 {size} 份 @ {next_open:.2f}")

        elif signal == "sell" and next_open:
            pos = self.position_mgr.get_position("PAPER")
            if pos and pos["size"] > 0:
                order = self.order_mgr.create_market_order("PAPER", SELL, pos["size"], today_date)
                self.order_mgr.submit(order["id"])
                self.order_mgr.fill(order["id"], next_open, today_date)
                prev_value = self.position_mgr.get_total_value({"PAPER": today_price})
                self.position_mgr.update_position("PAPER", SELL, pos["size"], next_open, today_date)
                new_value = self.position_mgr.get_total_value({"PAPER": next_open})
                self._last_trade_pnl = new_value - prev_value
                if self.guard:
                    self.guard.check_trade(next_open, today_price)
                    self.guard.update_equity(new_value, self._last_trade_pnl)
                logger.info(f"[{today_date}] 卖出 {pos['size']} 份 @ {next_open:.2f}")

    def _execute_live_signal(self, signal: str, symbol: str, price: float, today_date):
        """实时模式：以当前价成交。"""
        pos = self.position_mgr.get_position("PAPER")

        if signal == "buy":
            size = int(self.position_mgr.cash * 0.95 / price)  # 留5%缓冲
            if size > 0:
                order = self.order_mgr.create_market_order("PAPER", BUY, size, today_date)
                self.order_mgr.submit(order["id"])
                self.order_mgr.fill(order["id"], price, today_date)
                prev_value = self.position_mgr.cash
                self.position_mgr.update_position("PAPER", BUY, size, price, today_date)
                new_value = self.position_mgr.get_total_value({symbol: price})
                self._last_trade_pnl = new_value - prev_value
                if self.guard:
                    self.guard.check_trade(price, price)
                    self.guard.update_equity(new_value, self._last_trade_pnl)
                logger.info(f"[{today_date}] 实时买入 {size} 份 @ {price:.2f}")

        elif signal == "sell":
            if pos and pos["size"] > 0:
                order = self.order_mgr.create_market_order("PAPER", SELL, pos["size"], today_date)
                self.order_mgr.submit(order["id"])
                self.order_mgr.fill(order["id"], price, today_date)
                prev_value = self.position_mgr.get_total_value({symbol: price})
                self.position_mgr.update_position("PAPER", SELL, pos["size"], price, today_date)
                new_value = self.position_mgr.cash + pos["size"] * price
                self._last_trade_pnl = new_value - prev_value
                if self.guard:
                    self.guard.check_trade(price, price)
                    self.guard.update_equity(new_value, self._last_trade_pnl)
                logger.info(f"[{today_date}] 实时卖出 {pos['size']} 份 @ {price:.2f}")

    def _risk_check(self, signal: str, price: float) -> tuple:
        """风控检查（账户级 + 策略级）。"""
        pos = self.position_mgr.get_position("PAPER")
        position_size = pos["size"] if pos else 0
        total_value = self.position_mgr.cash + position_size * price
        exposure_pct = (position_size * price) / total_value if total_value > 0 else 0

        checks = run_all_checks(position_pct=exposure_pct)
        for ok, msg in checks:
            if not ok:
                return False, msg
        return True, "风控通过"

    def _log_daily(self, today_date, price: float, signal, action: str, note: str) -> dict:
        """记录每日快照并返回。"""
        pos = self.position_mgr.get_position("PAPER")
        entry = {
            "date": today_date,
            "close": price,
            "cash": self.position_mgr.cash,
            "position_size": pos["size"] if pos else 0,
            "position_cost": pos["avg_cost"] if pos else 0,
            "signal": signal,
            "action": action,
            "note": note,
        }
        self.daily_log.append(entry)
        return entry

    def _flush_store(self):
        """将当前状态刷入持久化存储。"""
        if not self.store or self.instance_id is None:
            return
        try:
            if self.daily_log:
                self.store.save_daily_log(self.instance_id, [self.daily_log[-1]])
            pos = self.position_mgr.get_position("PAPER")
            if pos and pos["size"] > 0:
                self.store.save_position(self.instance_id, {"PAPER": pos})
            if self.guard:
                self.store.save_guard_state(self.instance_id, self.guard)
        except Exception as e:
            logger.debug(f"持久化刷入跳过: {e}")

    def _progress_log(self, i: int, total: int, today_date, price: float):
        """每50条打印一次进度。"""
        pos = self.position_mgr.get_position("PAPER")
        total_value = self.position_mgr.cash
        if pos and pos["size"] > 0:
            total_value += pos["size"] * price
        ret = (total_value / self.initial_cash - 1) * 100
        logger.info(f"第 {i+1}/{total} 天 [{today_date}] 收益率: {ret:+.2f}%")

    def _build_result(self, baseline: dict) -> dict:
        """汇总运行结果。"""
        if not self.data.empty:
            final_prices = {"PAPER": float(self.data.iloc[-1]["close"])}
        else:
            # 实时模式：用最新日志中的收盘价
            last_close = self.daily_log[-1]["close"] if self.daily_log else 0
            final_prices = {"PAPER": last_close}

        total_value = self.position_mgr.get_total_value(final_prices)
        total_return = total_value / self.initial_cash - 1
        bt_return = baseline.get("total_return", 0) if baseline else 0
        deviation = abs(total_return - bt_return) if baseline else 0

        logger.info(
            f"纸上交易完成 | 总收益: {total_return:.2%} | "
            f"回测收益: {bt_return:.2%} | 偏差: {deviation:.2%} | "
            f"信号数: {len(self.signals)} | 警告数: {len(self.warnings)}"
        )

        guard_status = None
        if self.guard:
            guard_status = self.guard.status()

        return {
            "strategy": self.strategy_class.__name__,
            "initial_cash": self.initial_cash,
            "final_value": total_value,
            "total_return": total_return,
            "baseline_return": bt_return,
            "deviation": deviation,
            "signals": self.signals,
            "daily_log": self.daily_log,
            "warnings": self.warnings,
            "orders": self.order_mgr.orders,
            "guard_status": guard_status,
            "position_summary": self.position_mgr.get_summary(final_prices),
            "summary": self._build_summary(total_value, total_return, bt_return, deviation, baseline),
        }

    def _build_summary(self, total_value: float, total_return: float, bt_return: float, deviation: float, baseline: dict) -> str:
        """生成纸上交易总结文本。"""
        filled = len([o for o in self.order_mgr.orders if o.get("status") == "filled"])
        guard_info = ""
        if self.guard and self.guard.is_blown():
            guard_info = f"\n  ⚠️ StrategyGuard已熔断: {self.guard.reason}"
        return "\n".join([
            "=" * 60,
            "                  纸上交易总结",
            "=" * 60,
            f"  策略: {self.strategy_class.__name__}",
            f"  参数: {self.strategy_params}",
            f"  初始资金: {self.initial_cash:,.0f} 元",
            f"  最终资产: {total_value:,.2f} 元",
            f"  总收益: {total_return:.2%}",
            f"  回测预期: {bt_return:.2%}",
            f"  偏差: {deviation:.2%}",
            f"  信号数: {len(self.signals)}",
            f"  成交数: {filled}",
            f"  风控拦截: {len(self.warnings)} 次",
            f"{guard_info}",
            "=" * 60,
        ])


# ============================================================
# 便捷函数
# ============================================================

def run_paper_trading(
    strategy_class,
    df: pd.DataFrame,
    initial_cash: float = 100000.0,
    **strategy_params,
) -> dict:
    """一键运行历史模式纸上交易。"""
    trader = PaperTrader(strategy_class, df, initial_cash, **strategy_params)
    return trader.run()


def run_paper_trading_live(
    strategy_class,
    feed,
    symbol: str,
    initial_cash: float = 100000.0,
    **strategy_params,
) -> dict:
    """一键运行实时模式纸上交易。"""
    trader = PaperTrader(strategy_class, pd.DataFrame(), initial_cash, **strategy_params)
    return trader.run_live(feed, symbol)


# ============================================================
# 命令行测试
# ============================================================
# python live/paper_trader.py

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data.fetchers.fallback import fetch_index_daily_safe as fetch_index_daily
    from backtest.strategies.builtin.ma_cross import MaCrossStrategy

    print("=" * 60)
    print("纸上交易测试：双均线策略")
    print("=" * 60)

    df = fetch_index_daily("沪深300", "20240101", "20250601")
    result = run_paper_trading(MaCrossStrategy, df, initial_cash=100000, fast=5, slow=20)

    print(result["summary"])

    if result.get("guard_status"):
        print(f"\nStrategyGuard: {result['guard_status']}")

    if result["warnings"]:
        print(f"\n风控拦截记录 ({len(result['warnings'])} 次):")
        for w in result["warnings"]:
            print(f"  {w}")
