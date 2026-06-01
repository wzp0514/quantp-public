"""
缠论 — 笔/线段/中枢/背驰识别，输出三类买卖点+级别标注。

实现链：包含关系处理 → 顶底分型 → 笔 → 线段 → 中枢 → 背驰(MACD面积比较)
输入 OHLCV 数据（日线起步，后续扩展 30min/5min）
输出买卖点信号 + 级别标注(1F/5F/30F/日线)

参考：缠中说禅博客原文 + 开源实现 Chanlun-X

用法
--------
>>> from backtest.analysis.chan_theory import ChanTheory
>>> ct = ChanTheory(df)
>>> signals = ct.analyze()
>>> for s in signals:
...     print(s["date"], s["type"], s["price"], s["level"])
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.log import get_logger

logger = get_logger("chan_theory")

# 级别映射
LEVEL_MAP = {"1F": 1, "5F": 5, "30F": 30, "D": 100, "W": 500}


class ChanTheory:
    """缠论分析器"""

    def __init__(self, df: pd.DataFrame, level: str = "D"):
        """
        参数
        ----------
        df : DataFrame
            OHLCV 数据，需含 date/open/high/low/close/volume 列
        level : str
            分析级别: "1F"/"5F"/"30F"/"D"/"W"
        """
        self.df = df.copy()
        self.level = level
        self._kline_merged = False

    # ═══════════════════════════════════════════════════════════
    # Step 1: 包含关系处理
    # ═══════════════════════════════════════════════════════════

    def _merge_containment(self) -> pd.DataFrame:
        """处理 K 线包含关系：上升趋势取高高/低高，下降趋势取高低/低低"""
        df = self.df.copy()
        if len(df) < 3:
            return df

        df["_dir"] = 0  # 1=上升, -1=下降
        for i in range(1, len(df)):
            if df["high"].iloc[i] > df["high"].iloc[i - 1] and df["low"].iloc[i] > df["low"].iloc[i - 1]:
                df.loc[df.index[i], "_dir"] = 1
            elif df["high"].iloc[i] < df["high"].iloc[i - 1] and df["low"].iloc[i] < df["low"].iloc[i - 1]:
                df.loc[df.index[i], "_dir"] = -1
            else:
                df.loc[df.index[i], "_dir"] = df["_dir"].iloc[i - 1]

        # 合并包含关系
        merged = []
        i = 0
        while i < len(df):
            row = df.iloc[i].to_dict()
            j = i + 1
            while j < len(df):
                next_row = df.iloc[j]
                h1, l1 = row["high"], row["low"]
                h2, l2 = next_row["high"], next_row["low"]
                # 检测包含关系
                if (h1 >= h2 and l1 <= l2) or (h2 >= h1 and l2 <= l1):
                    direction = df["_dir"].iloc[j] if df["_dir"].iloc[j] != 0 else 1
                    if direction == 1:
                        row["high"] = max(h1, h2)
                        row["low"] = max(l1, l2)
                    else:
                        row["high"] = min(h1, h2)
                        row["low"] = min(l1, l2)
                    # 合并后不更新 close/volume，保留第一根
                    j += 1
                else:
                    break
            merged.append(row)
            i = j

        result = pd.DataFrame(merged)
        result["date"] = pd.to_datetime(result["date"])
        self._kline_merged = True
        logger.debug(f"包含关系处理: {len(df)} → {len(result)} K线")
        return result

    # ═══════════════════════════════════════════════════════════
    # Step 2: 顶底分型识别
    # ═══════════════════════════════════════════════════════════

    def _find_fractals(self, df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
        """
        识别顶分型和底分型。
        顶分型：中间 high 高于两侧，且中间 low 高于两侧
        底分型：中间 low 低于两侧，且中间 high 低于两侧
        """
        tops, bottoms = [], []
        for i in range(1, len(df) - 1):
            h0, l0 = df["high"].iloc[i - 1], df["low"].iloc[i - 1]
            h1, l1 = df["high"].iloc[i], df["low"].iloc[i]
            h2, l2 = df["high"].iloc[i + 1], df["low"].iloc[i + 1]

            if h1 > h0 and h1 > h2 and l1 > l0 and l1 > l2:
                tops.append({"index": i, "date": str(df["date"].iloc[i]), "price": h1, "type": "top"})
            if l1 < l0 and l1 < l2 and h1 < h0 and h1 < h2:
                bottoms.append({"index": i, "date": str(df["date"].iloc[i]), "price": l1, "type": "bottom"})

        logger.debug(f"分型: {len(tops)} 顶, {len(bottoms)} 底")
        return tops, bottoms

    # ═══════════════════════════════════════════════════════════
    # Step 3: 笔（连接顶底分型）
    # ═══════════════════════════════════════════════════════════

    def _build_strokes(self, tops: list[dict], bottoms: list[dict], df: pd.DataFrame) -> list[dict]:
        """
        笔：顶底分型交替连接，相邻分型间至少 4 根 K 线（含两端）。
        不满足则取极值更高的顶 / 极值更低的底。
        """
        # 合并并按时序排列
        fractals = tops + bottoms
        fractals.sort(key=lambda x: x["index"])

        strokes = []
        pending = None

        for f in fractals:
            if pending is None:
                pending = f
                continue

            # 必须交替（顶→底 或 底→顶）
            if pending["type"] == f["type"]:
                # 同类型取更极端的
                if f["type"] == "top" and f["price"] > pending["price"]:
                    pending = f
                elif f["type"] == "bottom" and f["price"] < pending["price"]:
                    pending = f
                continue

            # 至少 4 根 K 线间隔
            if f["index"] - pending["index"] < 3:
                # 不满足间隔，保留更极端的
                if pending["type"] == "top" and f["price"] < pending["price"]:
                    continue  # 跳过这个底
                elif pending["type"] == "bottom" and f["price"] > pending["price"]:
                    continue  # 跳过这个顶

            strokes.append({
                "start": pending,
                "end": f,
                "direction": "down" if pending["type"] == "top" else "up",
            })
            pending = f

        logger.debug(f"笔: {len(strokes)} 根")
        return strokes

    # ═══════════════════════════════════════════════════════════
    # Step 4: 线段（至少 3 笔构成）
    # ═══════════════════════════════════════════════════════════

    def _build_segments(self, strokes: list[dict]) -> list[dict]:
        """线段：至少 3 笔，且第一笔被第三笔突破"""
        if len(strokes) < 3:
            return []

        segments = []
        for i in range(len(strokes) - 2):
            s1, s2, s3 = strokes[i], strokes[i + 1], strokes[i + 2]
            if s1["direction"] == s3["direction"]:
                ext = s3["end"]["price"] > s1["end"]["price"] if s1["direction"] == "up" else s3["end"]["price"] < s1["end"]["price"]
                if ext:
                    segments.append({
                        "start": s1["start"],
                        "end": s3["end"],
                        "direction": s1["direction"],
                        "stroke_count": 3,
                    })

        logger.debug(f"线段: {len(segments)} 条")
        return segments

    # ═══════════════════════════════════════════════════════════
    # Step 5: 中枢（三个连续线段重叠区域）
    # ═══════════════════════════════════════════════════════════

    def _find_pivots(self, segments: list[dict]) -> list[dict]:
        """中枢：连续三线段重叠区间"""
        if len(segments) < 3:
            return []

        pivots = []
        for i in range(len(segments) - 2):
            s1, s2, s3 = segments[i], segments[i + 1], segments[i + 2]
            high1 = max(s1["start"]["price"], s1["end"]["price"])
            low1 = min(s1["start"]["price"], s1["end"]["price"])
            high2 = max(s2["start"]["price"], s2["end"]["price"])
            low2 = min(s2["start"]["price"], s2["end"]["price"])
            high3 = max(s3["start"]["price"], s3["end"]["price"])
            low3 = min(s3["start"]["price"], s3["end"]["price"])

            # 重叠区：三线段高低点的交集
            zhongshu_high = min(high1, high2, high3)
            zhongshu_low = max(low1, low2, low3)

            if zhongshu_high > zhongshu_low:
                pivots.append({
                    "high": zhongshu_high,
                    "low": zhongshu_low,
                    "center": (zhongshu_high + zhongshu_low) / 2,
                    "width_pct": (zhongshu_high - zhongshu_low) / zhongshu_low * 100,
                    "start_date": s1["start"]["date"],
                    "end_date": s3["end"]["date"],
                })

        # 合并重叠中枢
        pivots = self._merge_overlapping_pivots(pivots)
        logger.debug(f"中枢: {len(pivots)} 个")
        return pivots

    def _merge_overlapping_pivots(self, pivots: list[dict]) -> list[dict]:
        if len(pivots) <= 1:
            return pivots
        pivots.sort(key=lambda x: x["low"])
        merged = [pivots[0]]
        for p in pivots[1:]:
            last = merged[-1]
            if p["low"] <= last["high"]:
                last["high"] = max(last["high"], p["high"])
                last["center"] = (last["high"] + last["low"]) / 2
            else:
                merged.append(p)
        return merged

    # ═══════════════════════════════════════════════════════════
    # Step 6: 背驰（MACD 面积比较）
    # ═══════════════════════════════════════════════════════════

    def _detect_divergence(self, df: pd.DataFrame, strokes: list[dict]) -> list[dict]:
        """
        MACD 面积背驰检测。
        比较相邻同向笔的 MACD 柱面积：新笔动能衰减 → 背驰。
        """
        # 计算 MACD
        close = df["close"].values
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_bar = 2 * (dif - dea)

        divergences = []
        for i in range(len(strokes) - 1):
            s1 = strokes[i]
            s2 = strokes[i + 1]
            if s1["direction"] != s2["direction"]:
                continue

            idx1_start = s1["start"]["index"]
            idx1_end = s1["end"]["index"]
            idx2_start = s2["start"]["index"]
            idx2_end = s2["end"]["index"]

            if idx2_start > idx1_end:
                area1 = abs(macd_bar.iloc[idx1_start:idx1_end + 1].sum())
                area2 = abs(macd_bar.iloc[idx2_start:idx2_end + 1].sum())

                if area1 > 0:
                    ratio = area2 / area1
                    if ratio < 0.5:  # 动能衰减超 50% = 背驰
                        divergences.append({
                            "date": s2["end"]["date"],
                            "direction": s2["direction"],
                            "area_ratio": round(ratio, 3),
                            "signal": "sell" if s2["direction"] == "up" else "buy",
                        })

        logger.debug(f"背驰: {len(divergences)} 处")
        return divergences

    # ═══════════════════════════════════════════════════════════
    # Step 7: 买卖点信号
    # ═══════════════════════════════════════════════════════════

    def _generate_signals(self, pivots: list[dict], divergences: list[dict],
                          df: pd.DataFrame) -> list[dict]:
        """
        三类买卖点：
          一类买点：背驰+底分型确认
          二类买点：回调不破一类买点低点
          三类买点：突破中枢上沿回踩不破
          （卖点对称反向）
        """
        signals = []
        current_price = float(df["close"].iloc[-1])

        # 一类：背驰信号直接映射
        for d in divergences:
            signals.append({
                "date": d["date"],
                "type": "一类买点" if d["signal"] == "buy" else "一类卖点",
                "price": current_price,
                "level": self.level,
                "source": "背驰",
                "confidence": round(1 - d["area_ratio"], 2),
            })

        # 二类：中枢附近价格回归
        for p in pivots:
            if current_price < p["high"] * 1.02 and current_price > p["low"]:
                signals.append({
                    "date": str(df["date"].iloc[-1]),
                    "type": "二类买点",
                    "price": round(p["low"], 2),
                    "level": self.level,
                    "source": "中枢支撑",
                    "confidence": 0.7,
                })
            elif current_price > p["low"] * 0.98 and current_price < p["high"]:
                signals.append({
                    "date": str(df["date"].iloc[-1]),
                    "type": "二类卖点",
                    "price": round(p["high"], 2),
                    "level": self.level,
                    "source": "中枢压力",
                    "confidence": 0.7,
                })

        # 三类：突破回踩
        for p in pivots:
            if current_price > p["high"] * 1.01:
                signals.append({
                    "date": str(df["date"].iloc[-1]),
                    "type": "三类买点",
                    "price": round(p["high"], 2),
                    "level": self.level,
                    "source": "突破回踩",
                    "confidence": 0.6,
                })

        signals.sort(key=lambda x: x["confidence"], reverse=True)
        return signals

    # ═══════════════════════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════════════════════

    def analyze(self) -> dict:
        """
        运行完整缠论分析链。

        返回
        -------
        dict: signals, strokes, pivots, divergences, fractal_count, summary
        """
        if len(self.df) < 10:
            logger.warning("数据不足（<10 条），无法进行缠论分析")
            return {"signals": [], "summary": "数据不足"}

        # Step 1: 包含关系处理
        merged = self._merge_containment()

        # Step 2: 顶底分型
        tops, bottoms = self._find_fractals(merged)

        # Step 3: 笔
        strokes = self._build_strokes(tops, bottoms, merged)

        # Step 4: 线段
        segments = self._build_segments(strokes)

        # Step 5: 中枢
        pivots = self._find_pivots(segments)

        # Step 6: 背驰
        divergences = self._detect_divergence(merged, strokes)

        # Step 7: 买卖点
        signals = self._generate_signals(pivots, divergences, merged)

        buy_count = sum(1 for s in signals if "买点" in s["type"])
        sell_count = sum(1 for s in signals if "卖点" in s["type"])

        summary = (
            f"缠论分析({self.level}级): {len(tops)}顶/{len(bottoms)}底分型, "
            f"{len(strokes)}笔, {len(segments)}线段, {len(pivots)}中枢, "
            f"{len(divergences)}背驰, {buy_count}买点/{sell_count}卖点"
        )
        logger.info(summary)

        return {
            "signals": signals,
            "stroke_count": len(strokes),
            "segment_count": len(segments),
            "pivot_count": len(pivots),
            "divergence_count": len(divergences),
            "fractal_count": len(tops) + len(bottoms),
            "pivots": [{"high": p["high"], "low": p["low"], "center": round(p["center"], 2),
                        "start": p["start_date"], "end": p["end_date"]} for p in pivots],
            "divergences": divergences,
            "summary": summary,
        }

    def report(self) -> str:
        """生成可打印分析报告"""
        r = self.analyze()
        lines = [
            "=" * 60,
            f"  缠论分析报告 — {self.level} 级别",
            "=" * 60,
            f"  分型: {r['fractal_count']} 个",
            f"  笔: {r['stroke_count']} 根",
            f"  线段: {r['segment_count']} 条",
            f"  中枢: {r['pivot_count']} 个",
            f"  背驰: {r['divergence_count']} 处",
        ]
        if r["pivots"]:
            lines.append("  ── 中枢 ──")
            for p in r["pivots"]:
                lines.append(f"    ¥{p['low']:.2f} - ¥{p['high']:.2f} (中心 ¥{p['center']:.2f})")
        if r["signals"]:
            lines.append("  ── 买卖点 ──")
            for s in r["signals"][:5]:
                lines.append(f"    {s['date']} {s['type']} @ ¥{s['price']} (置信度: {s['confidence']})")
        lines.append("=" * 60)
        return "\n".join(lines)
