"""
市场情绪基础指标 — 涨跌停家数/炸板率/封板率/连板高度等。

从 akshare 获取当日涨跌停列表统计，作为择时参考信号。

用法
--------
>>> from data.alternative.market_sentiment import MarketSentiment
>>> ms = MarketSentiment()
>>> result = ms.snapshot()
>>> print(result["limit_up_count"], result["bust_rate"])
"""

import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config.log import get_logger

logger = get_logger("market_sentiment")


class MarketSentiment:
    """A股市场情绪快照"""

    def __init__(self):
        self._cache = {}
        self._cache_time = None

    def snapshot(self, date: str = "") -> dict:
        """
        获取当日市场情绪快照。

        参数
        ----------
        date : str
            日期 YYYYMMDD，默认今天

        返回
        -------
        dict: limit_up_count, limit_down_count, bust_rate, seal_rate,
              max_board_height, up_3pct_count, up_5pct_count, up_8pct_count,
              down_3pct_count, down_5pct_count, down_8pct_count, summary
        """
        if not date:
            date = datetime.now().strftime("%Y%m%d")

        # 缓存当日结果
        if self._cache_time == date:
            return self._cache

        result = self._fetch_akshare(date)
        if result is None:
            result = self._empty_result(date)
        else:
            result["status"] = "ok"

        self._cache = result
        self._cache_time = date
        return result

    def _fetch_akshare(self, date: str) -> dict | None:
        """从 akshare 获取涨跌停数据"""
        try:
            import akshare as ak
            # 获取当日涨跌停列表
            limit_up_df = ak.stock_zt_pool_em(date=date)
            limit_down_df = ak.stock_zt_pool_dtgc_em(date=date)

            up_count = len(limit_up_df) if limit_up_df is not None else 0
            down_count = len(limit_down_df) if limit_down_df is not None else 0

            # 连板高度
            max_height = 0
            if limit_up_df is not None and not limit_up_df.empty and "连板数" in limit_up_df.columns:
                max_height = int(limit_up_df["连板数"].max())

            # 炸板率
            if limit_up_df is not None and not limit_up_df.empty and "炸板" in limit_up_df.columns:
                bust_count = int(limit_up_df["炸板"].sum()) if "炸板" in limit_up_df.columns else 0
                bust_rate = round(bust_count / max(up_count + bust_count, 1) * 100, 1)
            else:
                bust_rate = 0

            # 封板率
            seal_rate = round(100 - bust_rate, 1) if bust_rate > 0 else 100.0

            # 涨跌幅分布（用指数成分替代全市场，避免耗时）
            pct_dist = self._get_pct_distribution(date)

            result = {
                "date": date,
                "limit_up_count": up_count,
                "limit_down_count": down_count,
                "limit_ratio": round(up_count / max(down_count, 1), 2),
                "bust_rate": bust_rate,
                "seal_rate": seal_rate,
                "max_board_height": max_height,
                **pct_dist,
            }
            logger.info(f"市场情绪: {up_count}涨停/{down_count}跌停, 炸板率{bust_rate}%, 连板高度{max_height}")
            return result

        except ImportError:
            logger.debug("akshare 未安装", exc_info=False)
            return None
        except Exception as e:
            logger.warning(f"获取涨跌停数据失败: {e}")
            return None

    def _get_pct_distribution(self, date: str) -> dict:
        """获取涨跌幅分布（简化：用沪深300成分代替全市场）"""
        try:
            import akshare as ak
            df = ak.stock_zh_index_spot_em()
            if df is None or df.empty:
                return self._empty_pct_dist()
            pct_col = "涨跌幅" if "涨跌幅" in df.columns else None
            if pct_col is None:
                return self._empty_pct_dist()
            pct = pd.to_numeric(df[pct_col], errors="coerce").dropna()
            return {
                "up_3pct_count": int((pct > 3).sum()),
                "up_5pct_count": int((pct > 5).sum()),
                "up_8pct_count": int((pct > 8).sum()),
                "down_3pct_count": int((pct < -3).sum()),
                "down_5pct_count": int((pct < -5).sum()),
                "down_8pct_count": int((pct < -8).sum()),
                "total_samples": len(pct),
            }
        except Exception:
            return self._empty_pct_dist()

    @staticmethod
    def _empty_pct_dist() -> dict:
        return {"up_3pct_count": 0, "up_5pct_count": 0, "up_8pct_count": 0,
                "down_3pct_count": 0, "down_5pct_count": 0, "down_8pct_count": 0,
                "total_samples": 0}

    @staticmethod
    def _empty_result(date: str) -> dict:
        return {
            "date": date, "status": "unavailable",
            "limit_up_count": 0, "limit_down_count": 0, "limit_ratio": 0,
            "bust_rate": 0, "seal_rate": 100, "max_board_height": 0,
            "up_3pct_count": 0, "up_5pct_count": 0, "up_8pct_count": 0,
            "down_3pct_count": 0, "down_5pct_count": 0, "down_8pct_count": 0,
            "total_samples": 0,
        }

    def summary_text(self) -> str:
        """单行情绪摘要"""
        r = self.snapshot()
        if r.get("status") != "ok":
            return "市场情绪数据不可用"

        sentiment = "偏热" if r["limit_up_count"] > 100 else "偏冷" if r["limit_up_count"] < 30 else "正常"
        return (
            f"情绪{sentiment}: {r['limit_up_count']}涨停/{r['limit_down_count']}跌停, "
            f"炸板率{r['bust_rate']}%, 连板{r['max_board_height']}板"
        )

    @staticmethod
    def composite_fear_index(snapshot: dict) -> dict:
        """综合恐慌指数：涨跌停比(0.3)+波动率(0.4)+跌幅分布(0.3)→0-100。

        参数
        ----------
        snapshot : dict
            MarketSentiment.snapshot() 的输出

        返回
        -------
        dict: {fear_index, level, components, signal}
        """
        limit_up = snapshot.get("limit_up_count", 0)
        limit_down = snapshot.get("limit_down_count", 0)
        ratio = limit_up / max(limit_down, 1)
        limit_component = np.clip(100 - (ratio / 5) * 100, 0, 100)

        total = snapshot.get("total_samples", 1)
        down_8 = snapshot.get("down_8pct_count", 0)
        down_5 = snapshot.get("down_5pct_count", 0)
        down_3 = snapshot.get("down_3pct_count", 0)
        vol_score = (down_8 * 1.0 + down_5 * 0.6 + down_3 * 0.3) / max(total, 1) * 100
        vol_component = np.clip(vol_score, 0, 100)

        total_down = down_3 + down_5 + down_8
        if total_down > 0:
            dist_component = (down_8 / total_down) * 100
        else:
            dist_component = 0

        fear_index = (
            limit_component * 0.3 + vol_component * 0.4 + dist_component * 0.3
        )
        fear_index = np.clip(fear_index, 0, 100)

        if fear_index >= 80:
            level = "极度恐惧"
        elif fear_index >= 60:
            level = "恐惧"
        elif fear_index >= 40:
            level = "中性"
        elif fear_index >= 20:
            level = "贪婪"
        else:
            level = "极度贪婪"

        return {
            "fear_index": round(float(fear_index), 1),
            "level": level,
            "components": {
                "limit_ratio": round(float(limit_component), 1),
                "volatility": round(float(vol_component), 1),
                "distribution": round(float(dist_component), 1),
            },
            "signal": "减仓" if fear_index >= 60 else ("加仓" if fear_index <= 20 else "持有"),
        }
