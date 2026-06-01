"""
Backtrader 回测引擎封装

功能：
  - 策略基类（含日志、参数管理）
  - 数据喂入（从 DataFrame 转为 Backtrader 数据源）
  - 手续费/滑点/印花税建模
  - 回测运行 + 结果收集

使用 Backtrader 的事件驱动架构，逐 Bar 模拟交易，天然避免未来函数。

参考资料：
  Backtrader 官方文档: https://www.backtrader.com/docu/
"""

import logging
from datetime import datetime
from typing import Optional

import backtrader as bt
import pandas as pd

from config.log import get_logger
logger = get_logger("bt_runner")
from backtest.engine.result import BacktestResult, TradeRecord
# ============================================================
# A 股交易成本（佣金+印花税）
# ============================================================

class AShareCommission(bt.CommInfoBase):
    """
    A 股交易成本模型 — 支持股票/ETF/可转债。

      品种       佣金         最低   印花税       过户费
      stock      万2.5        5元    万5(仅卖)    万0.1(双)
      etf        万2.5        5元    免           万0.1(双)
      convertible 万0.5       1元    免           免

    用法:
      comm = AShareCommission.for_asset_type("etf")
      cerebro.broker.addcommissioninfo(comm)
    """
    params = (
        ("commission", 0.00025),   # 佣金率
        ("stamp_duty", 0.0005),    # 印花税（仅卖出，2023年8月起万5）
        ("transfer_fee", 0.00001), # 过户费（双向）
        ("min_commission", 5.0),   # 最低佣金
        ("percabs", True),
    )

    @staticmethod
    def for_asset_type(asset_type: str = "stock") -> "AShareCommission":
        """从配置读取品种参数，创建对应的佣金模型"""
        try:
            from config.loader import get_config
            cfg = get_config().get("asset_types", {}).get(asset_type, {})
        except Exception:
            cfg = {}
        return AShareCommission(
            commission=cfg.get("commission", 0.00025),
            stamp_duty=cfg.get("stamp_duty", 0.0005),
            transfer_fee=cfg.get("transfer_fee", 0.00001),
            min_commission=cfg.get("min_commission", 5.0),
        )

    def _getcommission(self, size, price, pseudoexec):
        value = abs(size) * price
        comm = max(value * self.p.commission, self.p.min_commission)
        comm += value * self.p.transfer_fee
        if size < 0:
            comm += value * self.p.stamp_duty
        return comm


# ============================================================
# 策略基类
# ============================================================

class BaseStrategy(bt.Strategy):
    """
    策略基类——所有自定义策略都继承它。

    已经帮你处理了：
      - 日志（记录每笔买卖的时机和价格）
      - 交易记录收集（存到 self.trades，回测后可导出分析）
      - A股约束（T+1/涨跌停板），通过 a_share_mode 参数开关
      - 基础仓位管理（max_position_pct=20%，不再全仓单票）

    执行假设（注明避免偷价争议）：
      - 信号在当日收盘后产生（使用 close[0]）
      - 订单在次日开盘执行（Backtrader 默认行为）
      - 仓位大小用 close[0] 估算（次日开盘价的最佳代理）
      - 止损在日内收盘价触发（保守假设：日内最不利情况）

    子类只需要实现：
      - __init__：定义指标（如移动平均线）
      - next：每个交易日的买卖逻辑
    """

    params = (
        ("a_share_mode", True),         # 默认启用A股约束（T+1+涨跌停）
        ("stop_loss_pct", 0.05),        # 默认5%止损（0=不设止损，不推荐）
        ("max_position_pct", 0.20),     # 单策略仓位上限（替代全仓单票）
    )

    def __init__(self):
        """初始化：子类必须调用 super().__init__()"""
        self.trades = []  # 记录每笔交易
        self.buy_dates = {}  # 记录每笔买入日期（用于T+1检查）
        self._stop_loss_orders = {}  # 跟踪止损单

        # A股约束
        if self.p.a_share_mode:
            from data.cleaner.a_share_constraints import AShareConstraints
            self.constraints = AShareConstraints()
        else:
            self.constraints = None

        # 止损警告
        if self.p.stop_loss_pct == 0:
            self.log("WARNING: stop_loss_pct=0, 无止损保护！参照05文档失败案例4/7")

    def _place_stop_loss(self, buy_order, entry_price: float):
        """买入后自动挂止损单。子类无需调用，buy() 自动处理。"""
        if self.p.stop_loss_pct <= 0 or not buy_order:
            return
        stop_price = entry_price * (1 - self.p.stop_loss_pct)
        sl = self.sell(exectype=bt.Order.Stop, price=stop_price,
                       size=buy_order.executed.size if hasattr(buy_order, 'executed') else buy_order.created.size)
        if sl:
            self._stop_loss_orders[id(buy_order)] = sl

    def log(self, msg: str):
        """记录日志（带日期和策略名）。__init__ 中调用时日期显示为 'INIT'"""
        try:
            dt = self.datas[0].datetime.date(0)
            logger.info(f"[{dt}] {self.__class__.__name__}: {msg}")
        except (IndexError, AttributeError):
            logger.info(f"[INIT] {self.__class__.__name__}: {msg}")

    def position_size(self, price: float = None) -> int:
        """
        计算合规仓位（不超过 max_position_pct 上限）。

        替代策略中手写的 size = int(cash / close[0])，
        避免全仓单票的非专业做法。
        """
        if price is None:
            price = self.data.close[0]
        cash = self.broker.getcash()
        max_cash = cash * self.p.max_position_pct
        return int(max_cash / price)

    def buy(self, *args, **kwargs):
        """买入（自动检查A股约束）"""
        if self.constraints:
            price = self.data.close[0]
            # 前收盘价 = close[-1]，不是 open[0]（open[0]是当天开盘价）
            prev_close = self.data.close[-1] if len(self.data) > 1 else price
            today = self.datas[0].datetime.date(0)
            ok, msg = self.constraints.can_buy("A_SHARE", price, prev_close, today)
            if not ok:
                self.log(f"买入被拒: {msg}")
                return None

        result = super().buy(*args, **kwargs)
        if result:
            today = self.datas[0].datetime.date(0)
            self.buy_dates[id(result)] = today
            # 自动挂止损单
            price = self.data.close[0]
            self._place_stop_loss(result, price)
        return result

    def sell(self, *args, **kwargs):
        """卖出（自动检查A股约束）"""
        if self.constraints:
            price = self.data.close[0]
            prev_close = self.data.close[-1]  # 前收盘价，不是当天开盘
            today = self.datas[0].datetime.date(0)
            # 找到对应仓位的买入日期
            buy_date = min(self.buy_dates.values()) if self.buy_dates else today
            ok, msg = self.constraints.can_sell("A_SHARE", price, prev_close, buy_date, today)
            if not ok:
                self.log(f"卖出被拒: {msg}")
                return None
        return super().sell(*args, **kwargs)

    def notify_order(self, order):
        """订单状态变化时 Backtrader 自动调用（不用自己写）"""
        if order.status in [order.Submitted, order.Accepted]:
            return  # 已提交/已接受，等待成交

        if order.status == order.Completed:
            if order.isbuy():
                self.log(f"买入 {order.executed.size} 股 @ {order.executed.price:.2f}")
            else:
                self.log(f"卖出 {order.executed.size} 股 @ {order.executed.price:.2f}")

            # 记录到交易列表
            self.trades.append({
                "date": self.datas[0].datetime.date(0),
                "type": "buy" if order.isbuy() else "sell",
                "size": order.executed.size,
                "price": order.executed.price,
                "value": order.executed.value,
                "commission": order.executed.comm,
            })

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"订单异常: {order.getstatusname()}")

    def notify_trade(self, trade):
        """一笔完整交易（买入→卖出）完成时自动调用"""
        if trade.isclosed:
            self.log(
                f"交易完成 | 毛利: {trade.pnl:.2f} | "
                f"净利: {trade.pnlcomm:.2f} | "
                f"持仓: {trade.baropen} → {trade.barclose}"
            )


# ============================================================
# 回测运行器
# ============================================================

def run_backtest(
    strategy_class=None,
    df: pd.DataFrame = None,
    initial_cash: float = 100000.0,
    commission: float = 0.00025,
    stamp_duty: float = 0.0005,
    slippage: float = 0.001,
    a_share_mode: bool = True,
    plot: bool = False,
    auto_save: bool = True,
    symbol: str = "",
    asset_type: str = "",
    auto_validate: bool = False,
    oos_split_date: str = "",
    use_universal: bool = False,
    package_dir: str = "",
    **strategy_params,
) -> dict:
    """
    运行一次回测。

    参数
    ----------
    strategy_class : bt.Strategy 子类
        策略类（不是实例），例如 MaCrossStrategy
    df : pd.DataFrame
        行情数据，必须包含 date/open/high/low/close/volume 列
    initial_cash : float
        初始资金（元），默认 10 万
    commission : float
        佣金率，默认万 2.5（0.00025）
    stamp_duty : float
        印花税率，默认万 5（0.0005），仅卖出收取
    slippage : float
        滑点比例，默认 0.1%（0.001）
    plot : bool
        是否画图（True 会弹出 Backtrader 自带的K线图）
    **strategy_params :
        传给策略的参数，例如 fast=5, slow=20

    返回
    -------
    dict，包含：
        - cerebro: Backtrader Cerebro 实例（可进一步分析）
        - strategy: 策略实例（含 self.trades 交易记录）
        - sharpe: 夏普比率
        - drawdown: 最大回撤（小数，0.15=15%）
        - total_return: 总收益率
        - annual_return: 年化收益率
        - final_value: 最终资金
        - trades_df: 交易记录 DataFrame
        - equity_df: 每日权益 DataFrame

    示例
    --------
    >>> from data.fetchers.akshare_fetch import fetch_index_daily
    >>> from backtest.engine.bt_runner import run_backtest
    >>> from backtest.strategies.builtin.ma_cross import MaCrossStrategy
    >>> df = fetch_index_daily("沪深300", "20200101", "20241231")
    >>> result = run_backtest(MaCrossStrategy, df, fast=5, slow=20)
    >>> print(f"年化收益: {result['annual_return']:.2%}")
    """
    # 0. 通用策略加载（fep 格式）
    universal_meta = None
    if use_universal and package_dir:
        from backtest.engine.strategy_adapter import load_strategy_from_package, UniversalStrategy
        universal_meta, compute_func = load_strategy_from_package(package_dir)
        strategy_class = UniversalStrategy
        user_params = dict(strategy_params) if strategy_params else {}
        strategy_params = {
            "compute_func": compute_func,
            "strategy_params": user_params,
        }

    # 0. 数据校验
    if df is None or df.empty:
        logger.error("回测数据为空，终止")
        return BacktestResult(errors=["数据为空"])
    if len(df) < 20:
        logger.warning(f"数据仅 {len(df)} 条（<20），回测结果不可靠")

    # 0a. 回测周期校验（牛熊覆盖）
    dates = pd.to_datetime(df["date"])
    years_span = (dates.max() - dates.min()).days / 365
    if years_span < 3:
        logger.warning(f"回测仅覆盖 {years_span:.1f} 年 (<3年)，可能未包含完整牛熊周期。"
                       f"建议 A 股从 2015 年至今覆盖股灾+熊市+修复。参照05文档陷阱6。")

    # 0b. 幸存者偏差提示
    logger.info("注意: 回测使用当前仍在交易的标的，存在幸存者偏差——收益可能被高估10-30%。"
                "实盘建议回测结果打7折。参照05文档陷阱3。")

    logger.info(f"开始回测: {strategy_class.__name__}, 初始资金 {initial_cash:,.0f} 元")
    logger.info(f"数据: {len(df)} 条, {df['date'].min()} ~ {df['date'].max()}")
    logger.info("注意: A股普通账户无法做空，回测/策略挖掘默认仅做多。若策略产生做空信号将被忽略。")

    # 1. 准备数据：把 DataFrame 转成 Backtrader 能识别的格式
    data = bt.feeds.PandasData(
        dataname=df,
        datetime="date",     # 日期列
        open="open",         # 开盘价
        high="high",         # 最高价
        low="low",           # 最低价
        close="close",       # 收盘价
        volume="volume",     # 成交量
        openinterest=-1,     # A 股没有持仓量概念
    )

    # 2. 创建回测引擎
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.broker.setcash(initial_cash)

    # 3. 设置交易成本（品种自动切换：stock/etf/convertible_bond）
    if not asset_type and symbol:
        sym_upper = symbol.upper()
        if "ETF" in sym_upper:
            asset_type = "etf"
        elif "转债" in symbol or "CONVERTIBLE" in sym_upper or "BOND" in sym_upper:
            asset_type = "convertible_bond"
        else:
            asset_type = "stock"
    if asset_type:
        comm_info = AShareCommission.for_asset_type(asset_type)
        logger.info(f"品种: {asset_type}, 佣金={comm_info.p.commission:.4%}, "
                    f"印花税={comm_info.p.stamp_duty:.4%}")
    else:
        comm_info = AShareCommission(
            commission=commission,
            stamp_duty=stamp_duty,
        )
    cerebro.broker.addcommissioninfo(comm_info)

    # 4. 设置滑点
    #    FixedPercSlippage(slippage) 表示每笔成交价在当前价格基础上滑动 slippage%
    #    例如 slippage=0.001 表示 10 元股票实际成交价可能是 10.01（买入）或 9.99（卖出）
    cerebro.broker.set_slippage_perc(slippage)

    # 5. 加密模式检测：symbol 含 "/" 自动切换
    is_crypto = "/" in symbol if symbol else False
    if is_crypto:
        a_share_mode = False
        stamp_duty = 0.0
        from backtest.engine.crypto_commission import CryptoCommission
        cc = CryptoCommission()
        commission = cc.taker_fee
    # 添加策略（a_share_mode 由 strategy 的 params 继承，通过 BaseStrategy 默认 True）
    cerebro.addstrategy(strategy_class, a_share_mode=a_share_mode, **strategy_params)

    # 6. 添加分析器（自动计算各种指标）
    # 无风险利率从配置读取，默认 2.5%（2026年中国10年国债收益率）
    try:
        from config.loader import get_config
        risk_free = get_config().get("backtest", {}).get("risk_free_rate", 0.025)
    except Exception:
        risk_free = 0.025
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                         riskfreerate=risk_free, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades_analyzer")

    # 7. 记录回测前资金
    start_value = cerebro.broker.getvalue()
    logger.info(f"回测前资金: {start_value:,.2f} 元")

    # 8. 运行回测
    logger.info("运行回测中...")
    results = cerebro.run()
    strategy_instance = results[0]  # 第一个（也是唯一一个）策略实例

    # 9. 记录回测后资金
    end_value = cerebro.broker.getvalue()
    logger.info(f"回测后资金: {end_value:,.2f} 元")

    # 10. 提取分析结果
    total_return = (end_value / start_value) - 1

    # 夏普比率
    sharpe_analyzer = strategy_instance.analyzers.sharpe.get_analysis()
    sharpe = sharpe_analyzer.get("sharperatio", None)

    # 最大回撤（Backtrader 返回的是百分数，如 24.5 表示 24.5%，转成小数 0.245）
    drawdown_analyzer = strategy_instance.analyzers.drawdown.get_analysis()
    max_drawdown = drawdown_analyzer.get("max", {}).get("drawdown", 0) / 100

    # 年化收益率
    returns_analyzer = strategy_instance.analyzers.returns.get_analysis()
    annual_return = returns_analyzer.get("rnorm100", 0) / 100

    # 交易分析
    trade_analysis = strategy_instance.analyzers.trades_analyzer.get_analysis()
    total_trades = trade_analysis.get("total", {}).get("total", 0)
    won_trades = trade_analysis.get("won", {}).get("total", 0)
    lost_trades = trade_analysis.get("lost", {}).get("total", 0)
    win_rate = won_trades / total_trades if total_trades > 0 else 0

    logger.info(
        f"回测完成 | 总收益: {total_return:.2%} | "
        f"年化: {annual_return:.2%} | "
        f"夏普: {sharpe if sharpe else 'N/A'} | "
        f"最大回撤: {max_drawdown:.2%} | "
        f"交易次数: {total_trades} | "
        f"胜率: {win_rate:.2%}"
    )

    # 回测质量警告
    if annual_return > 0.30 and max_drawdown < 0.10:
        logger.warning("[!] 年化>30%且回撤<10%——结果过于完美，建议检查: "
                       "(1)是否有未来函数 (2)是否只测了牛市 (3)样本外验证。参照05文档规则2。")
    logger.info("注意: 回测中止损以固定滑点0.1%成交，实盘极端行情下止损可能无法成交。参照05文档失败2/失败4。")

    # 11. 画图（可选）
    if plot:
        cerebro.plot(style="candlestick")

    # 12. 组装返回结果
    trades_df = pd.DataFrame(strategy_instance.trades) if strategy_instance.trades else pd.DataFrame()

    # 构建每日权益曲线（从 analyzers 提取）
    equity_list = []
    for analyzer in results:
        try:
            returns_data = analyzer.analyzers.returns.get_analysis()
            if "rtn" in returns_data:
                # rtn 是每期收益率的 dict，按日期索引
                cum_ret = 1.0
                for k, v in returns_data["rtn"].items():
                    cum_ret *= (1 + v)
                    equity_list.append({"date": k, "equity": initial_cash * cum_ret})
        except Exception as e:
            logger.warning(f"权益曲线构建失败: {e}")
    equity_df = pd.DataFrame(equity_list)

    result = BacktestResult(
        strategy_name=strategy_class.__name__,
        symbol=symbol,
        start_date=str(df["date"].min().date()) if "date" in df.columns else "",
        end_date=str(df["date"].max().date()) if "date" in df.columns else "",
        total_bars=len(df),
        start_value=initial_cash,
        end_value=end_value,
        total_return=total_return,
        annual_return=annual_return,
        drawdown=max_drawdown,
        sharpe=sharpe,
        calmar=annual_return / max_drawdown if max_drawdown > 0 else None,
        total_trades=total_trades,
        win_trades=won_trades,
        loss_trades=lost_trades,
        win_rate=win_rate,
        equity_curve=equity_df.to_dict("records") if not equity_df.empty else [],
    )

    # 自动存档
    if auto_save and result.total_trades > 0:
        try:
            from data.vault.backtest_store import BacktestStore
            bs = BacktestStore()
            bs.save(strategy=strategy_class.__name__, result=result.to_dict(),
                    symbol=symbol, params=strategy_params if strategy_params else None)
        except Exception as e:
            logger.debug(f"Auto-save skipped: {e}")

    # 自动验证
    if auto_validate and result.total_trades > 0:
        result_dict = result.to_dict()
        result_dict["validation"] = _run_auto_validate(
            strategy_class, df, result_dict, strategy_params,
            oos_split_date=oos_split_date,
        )

    return result


def _run_auto_validate(
    strategy_class,
    df: pd.DataFrame,
    backtest_result: dict,
    strategy_params: dict,
    oos_split_date: str = "",
) -> dict:
    """Auto-run validation after backtest: out-of-sample + statistical significance."""
    validation = {}

    # 1. Out-of-sample test
    try:
        from backtest.analysis.validate import out_of_sample_test
        if not oos_split_date:
            dates = pd.to_datetime(df["date"])
            split_idx = int(len(dates) * 0.8)
            oos_split_date = str(dates.iloc[split_idx].date())
        oos = out_of_sample_test(strategy_class, df, oos_split_date, **strategy_params)
        validation["out_of_sample"] = {
            "passed": oos.get("passed", False),
            "decay": oos.get("decay", 1.0),
            "train_return": oos.get("train_return", 0),
            "test_return": oos.get("test_return", 0),
            "detail": oos.get("detail", ""),
        }
    except Exception as e:
        validation["out_of_sample"] = {"passed": False, "error": str(e)}

    # 2. Statistical significance
    try:
        from backtest.analysis.accuracy import statistical_test
        equity = backtest_result.get("equity_df")
        if equity is not None and not equity.empty:
            daily_ret = equity["equity"].pct_change().dropna()
            st = statistical_test(daily_ret)
            validation["statistical"] = {
                "significant": st.get("significant", False),
                "p_value": st.get("p_value", 1.0),
                "prob_negative": st.get("prob_negative", 1.0),
                "interpretation": st.get("interpretation", ""),
            }
    except Exception as e:
        validation["statistical"] = {"significant": False, "error": str(e)}

    # 3. Overall
    oos_ok = validation.get("out_of_sample", {}).get("passed", False)
    sig_ok = validation.get("statistical", {}).get("significant", False)
    validation["overall"] = "ALL PASSED" if (oos_ok and sig_ok) else "PARTIAL"

    logger.info(f"Auto-validate: OOS={'PASS' if oos_ok else 'FAIL'}, "
                f"Stats={'PASS' if sig_ok else 'FAIL'} → {validation['overall']}")

    return validation
