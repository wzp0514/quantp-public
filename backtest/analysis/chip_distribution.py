"""
筹码分布 — 成交量在价格轴上的堆积估算。

简易版：日K (high-low) 区间分配成交量到价格桶，滚动 N 日衰减叠加。
若后续有 Level-2 逐笔数据可升级为精确版（移动成本分布算法）。

用法
--------
>>> from backtest.analysis.chip_distribution import ChipDistribution
>>> cd = ChipDistribution(df)
>>> result = cd.analyze()
>>> print(result["peak_price"])       # 筹码峰位置
>>> print(result["profit_ratio"])     # 获利比例
>>> print(result["concentration"])    # 成本集中度
"""

import logging

import numpy as np
import pandas as pd

from config.log import get_logger

logger = get_logger("chip_dist")


class ChipDistribution:
    """成交量价格堆积分析"""

    def __init__(self, df: pd.DataFrame, window: int = 90, n_bins: int = 100):
        """
        参数
        ----------
        df : DataFrame
            OHLCV 数据，需含 date/open/high/low/close/volume 列
        window : int
            滚动窗口天数，默认为 90（约一个季度）
        n_bins : int
            价格桶数量
        """
        self.df = df.copy()
        self.window = window
        self.n_bins = n_bins

    def analyze(self) -> dict:
        """
        计算筹码分布并输出分析指标。

        返回
        -------
        dict: peak_price, profit_ratio, concentration, resistance_levels, chip_profile
        """
        if len(self.df) < self.window:
            logger.warning(f"数据不足（{len(self.df)} < {self.window}），使用全部数据")
            window_data = self.df
        else:
            window_data = self.df.tail(self.window)

        # ── 1. 建立价格桶 ──
        price_min = window_data["low"].min()
        price_max = window_data["high"].max()
        price_grid = np.linspace(price_min, price_max, self.n_bins)
        bin_width = price_grid[1] - price_grid[0]
        chip_sum = np.zeros(self.n_bins)

        # ── 2. 成交量分配 + 滚动衰减 ──
        n = len(window_data)
        for i, (_, row) in enumerate(window_data.iterrows()):
            vol = row["volume"]
            low = row["low"]
            high = row["high"]
            if high <= low or vol <= 0:
                continue

            # 衰减权重：越久远权重越小（指数衰减，半衰期 window/2）
            decay = np.exp(-(n - 1 - i) / (self.window / 3))

            # 将当日成交量均匀分配到 [low, high] 区间内的价格桶
            low_idx = max(0, int((low - price_min) / bin_width))
            high_idx = min(self.n_bins - 1, int((high - price_min) / bin_width))
            if high_idx <= low_idx:
                high_idx = low_idx + 1

            vol_per_bin = vol * decay / max(high_idx - low_idx, 1)
            chip_sum[low_idx:high_idx + 1] += vol_per_bin

        # 归一化
        total = chip_sum.sum()
        if total > 0:
            chip_pct = chip_sum / total
        else:
            chip_pct = chip_sum

        # ── 3. 筹码峰 ──
        peak_idx = int(np.argmax(chip_pct))
        peak_price = float(price_grid[peak_idx])

        # ── 4. 获利比例（当前价以下筹码占比）──
        current_price = float(window_data["close"].iloc[-1])
        below_mask = price_grid <= current_price
        profit_ratio = float(chip_pct[below_mask].sum())

        # ── 5. 成本集中度（peak 附近 ±5% 的筹码占比）──
        half_width = max(int(self.n_bins * 0.05), 1)
        conc_start = max(peak_idx - half_width, 0)
        conc_end = min(peak_idx + half_width, self.n_bins)
        concentration = float(chip_pct[conc_start:conc_end].sum())

        # ── 6. 套牢盘压力位（当前价以上，筹码密集区）──
        above_mask = price_grid > current_price
        above_chips = pd.Series({
            "price": price_grid[above_mask],
            "chips": chip_pct[above_mask],
        })
        # 找上方筹码峰（连续高密度区）
        resistance_levels = []
        if above_chips["chips"].sum() > 0:
            # 用简单阈值：密度 > 均值的 2 倍即为压力位
            threshold = float(chip_pct.mean() * 2)
            for idx in np.where(chip_pct > threshold)[0]:
                p = float(price_grid[idx])
                if p > current_price:
                    # 合并相邻压力位
                    if not resistance_levels or p - resistance_levels[-1]["price"] > bin_width * 3:
                        resistance_levels.append({
                            "price": round(p, 2),
                            "density": round(float(chip_pct[idx]) * 100, 1),
                        })

        # ── 7. 筹码分布曲线（用于前端绘图）──
        chip_profile = [
            {"price": round(float(price_grid[i]), 2), "chips": round(float(chip_pct[i]) * 100, 2)}
            for i in range(self.n_bins)
        ]

        result = {
            "peak_price": round(peak_price, 2),
            "profit_ratio": round(profit_ratio * 100, 1),
            "concentration": round(concentration * 100, 1),
            "resistance_levels": resistance_levels[:5],  # Top 5
            "chip_profile": chip_profile,
            "current_price": round(current_price, 2),
        }

        logger.info(
            f"筹码分布: 峰={peak_price:.2f}, 获利={profit_ratio:.1%}, "
            f"集中度={concentration:.1%}, 压力位={len(resistance_levels)}"
        )

        return result

    def report(self) -> str:
        """生成可打印的分析报告"""
        r = self.analyze()
        lines = [
            "=" * 50,
            f"  筹码分布分析",
            "=" * 50,
            f"  当前价格: {r['current_price']}",
            f"  筹码峰: {r['peak_price']}",
            f"  获利比例: {r['profit_ratio']}%",
            f"  成本集中度 (peak±5%): {r['concentration']}%",
            f"  上方压力位: {len(r['resistance_levels'])} 个",
        ]
        if r["resistance_levels"]:
            lines.append("  ── 压力位 ──")
            for rl in r["resistance_levels"]:
                lines.append(f"    ¥{rl['price']} (密度 {rl['density']}%)")
        lines.append("=" * 50)
        return "\n".join(lines)
