"""
截面多因子选股策略 — 每日对所有股票排名 → 买Top N → 定期调仓

与 FactorStrategy（单标的、时间序列评分）互补：
  FactorStrategy: 一只指数/股票的因子评分 → 择时
  CrossSectionStrategy: 多只股票截面排名 → 选股
"""

import backtrader as bt
import numpy as np
import pandas as pd

from config.log import get_logger
logger = get_logger("cross_section_strategy")


class CrossSectionStrategy(bt.Strategy):
    """
    截面多因子选股策略。

    每日：计算所有股票的多因子综合评分 → 排名 → 买Top N只
    调仓：每 N 个交易日重新排名调仓
    """

    params = (
        ("factor_names", []),          # 使用的因子名列表
        ("top_n", 10),                 # 持有Top N只股票
        ("rebalance_freq", 21),        # 调仓频率（交易日）
        ("a_share_mode", True),
        ("stop_loss_pct", 0.05),
        ("max_position_pct", 0.20),    # 单票仓位上限
        ("verbose", False),
    )

    def __init__(self):
        self._bars_since_rebalance = 0
        self._entry_prices: dict[int, float] = {}   # data_idx → entry_price
        self.trades = []
        self._n_stocks = len(self.datas)  # 股票数量

    def log(self, msg: str):
        if self.p.verbose:
            try:
                dt = self.datas[0].datetime.date(0)
                logger.info(f"[{dt}] CS: {msg}")
            except Exception:
                pass

    def _get_rankings(self) -> list[tuple[int, float]]:
        """截面排名：对所有股票按因子综合评分排序，返回 [(data_idx, score), ...]"""
        scores = []
        for i, d in enumerate(self.datas):
            if len(d) < 20:
                continue
            factor_scores = []
            close = d.close[0]
            if close <= 0:
                continue

            # 简化版：用收益率、波动率、成交量做截面评分
            ret_5d = close / d.close[-5] - 1 if len(d) > 5 else 0
            ret_20d = close / d.close[-20] - 1 if len(d) > 20 else 0
            vol_20d = np.std([d.close[-j] / d.close[-j-1] - 1 for j in range(1, min(20, len(d)))]) if len(d) > 5 else 0.02
            vol_ratio = d.volume[0] / (np.mean([d.volume[-j] for j in range(1, min(20, len(d)))]) + 1) if len(d) > 5 else 1

            # 多因子等权评分（动量 + 低波 + 放量）
            momentum = ret_5d * 0.4 + ret_20d * 0.2
            low_vol = -vol_20d * 0.3
            liquidity = (vol_ratio - 1) * 0.1
            score = momentum + low_vol + liquidity
            scores.append((i, score))

        scores.sort(key=lambda x: -x[1])
        return scores

    def next(self):
        self._bars_since_rebalance += 1

        # 调仓日
        if self._bars_since_rebalance >= self.p.rebalance_freq:
            self._bars_since_rebalance = 0

            # 清仓所有持仓
            for i in range(self._n_stocks):
                pos = self.getposition(self.datas[i])
                if pos.size > 0:
                    self.close(data=self.datas[i])

            # 截面排名 → 买Top N
            rankings = self._get_rankings()
            n_buy = min(self.p.top_n, len(rankings))
            if n_buy == 0:
                return

            total_cash = self.broker.getcash()
            per_stock_cash = total_cash * self.p.max_position_pct

            bought = 0
            for data_idx, score in rankings:
                if bought >= n_buy:
                    break
                d = self.datas[data_idx]
                if len(d) < 20:
                    continue
                price = d.close[0]
                if price <= 0:
                    continue
                size = int(per_stock_cash / price)
                if size > 0:
                    self.buy(data=d, size=size)
                    self._entry_prices[data_idx] = price
                    bought += 1

            self.log(f"Rebalance: bought {bought}/{n_buy} stocks")

        # 每日止损检查
        for i in range(self._n_stocks):
            pos = self.getposition(self.datas[i])
            if pos.size > 0 and i in self._entry_prices:
                entry = self._entry_prices[i]
                price = self.datas[i].close[0]
                if price / entry - 1 < -self.p.stop_loss_pct:
                    self.close(data=self.datas[i])
                    del self._entry_prices[i]
                    self.log(f"Stop loss stock[{i}] @ {price:.2f}")
