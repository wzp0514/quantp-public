"""
另类数据统一管道 — 参考 Sattoro Hub 的 AI+GenAI 多源数据融合

汇总所有另类数据源 → 输出综合风险评分 → 集成到策略信号中。

权重可配置: config/settings.yaml → alternative_data 段
"""

from datetime import datetime

from config.loader import get_alternative_data_config
from config.log import get_logger
logger = get_logger("pipeline")


def _confidence_adj(weight: float, confidence: float, threshold: float) -> float:
    """低置信度源自动降权：conf < threshold → 权重减半"""
    if confidence < threshold:
        return weight * 0.5
    return weight


class AlternativeData:
    """
    另类数据中心 — 统一对外接口。

    用法
    --------
    >>> ad = AlternativeData()
    >>> result = ad.full_scan()
    >>> print(result["summary"])
    """

    def full_scan(self, symbol: str = "") -> dict:
        """
        全量扫描所有另类数据源。

        返回
        -------
        dict:
            risk_score: int (0-100)
            sentiment_score: float (-1~1)
            geopolitical_level: str
            should_trade: bool
            position_multiplier: float
            summary: str
        """
        from data.alternative.news_sentiment import NewsSentiment
        from data.alternative.geopolitical import GeopoliticalMonitor

        ad_cfg = get_alternative_data_config()
        conf_threshold = ad_cfg.get("confidence_threshold", 0.3)
        w_news = ad_cfg.get("news_sentiment", 0.25)
        w_geo = ad_cfg.get("geopolitical", 0.25)
        w_xq = ad_cfg.get("xueqiu_sentiment", 0.25)
        w_earn = ad_cfg.get("earnings_quality", 0.25)

        # ── 1. 新闻情绪 ─────────────────────────────────
        logger.info("扫描新闻情绪...")
        ns = NewsSentiment()
        sentiment = ns.analyze()
        sentiment_mult = ns.as_filter()
        news_conf = sentiment.get("confidence", 0.5)
        w_news_adj = _confidence_adj(w_news, news_conf, conf_threshold)

        # ── 2. 地缘政治 ─────────────────────────────────
        logger.info("扫描地缘风险...")
        gm = GeopoliticalMonitor()
        geo = gm.scan()
        _, geo_mult, geo_reason = gm.should_reduce_position()
        geo_conf = geo.get("confidence", 0.5)
        w_geo_adj = _confidence_adj(w_geo, geo_conf, conf_threshold)

        # ── 3. 雪球个股情绪 ──────────────────────────────
        xq_result = None
        xq_conf = 0.0
        w_xq_adj = 0.0
        if symbol:
            logger.info(f"扫描雪球个股情绪({symbol})...")
            try:
                from data.alternative.xueqiu_sentiment import XueqiuSentiment
                xs = XueqiuSentiment()
                xq_result = xs.stock_sentiment(symbol)
                xq_conf = xq_result.get("confidence", 0.3)
                w_xq_adj = _confidence_adj(w_xq, xq_conf, conf_threshold)
            except Exception as e:
                logger.warning(f"雪球情绪获取失败: {e}")

        # ── 4. 财报质量 ─────────────────────────────────
        earnings_result = None
        earn_conf = 0.0
        w_earn_adj = 0.0
        if symbol:
            logger.info(f"分析财报质量({symbol})...")
            try:
                from data.alternative.earnings_analysis import EarningsAnalyzer
                ea = EarningsAnalyzer()
                earnings_result = ea.analyze(symbol)
                earn_conf = earnings_result.get("confidence", 0.3)
                w_earn_adj = _confidence_adj(w_earn, earn_conf, conf_threshold)
            except Exception as e:
                logger.warning(f"财报分析失败: {e}")

        # ── 5. 恐慌指数综合指标 ──────────────────────────
        fear_risk = 0
        fear_index = None
        try:
            from data.alternative.market_sentiment import MarketSentiment
            ms = MarketSentiment()
            snap = ms.snapshot()
            if snap.get("status") == "ok":
                fi = MarketSentiment.composite_fear_index(snap)
                fear_index = fi["fear_index"]
                if fear_index >= 80:
                    fear_risk = 30 * (w_news_adj / max(total_w, 1e-9))
                elif fear_index >= 60:
                    fear_risk = 15 * (w_news_adj / max(total_w, 1e-9))
                logger.info(f"恐慌指数: {fear_index}/100 ({fi['level']})")
        except Exception as e:
            logger.debug(f"恐慌指数不可用: {e}")

        # ── 6. 综合风险评分（加权） ──────────────────────
        total_w = w_news_adj + w_geo_adj + w_xq_adj + w_earn_adj
        if total_w <= 0:
            total_w = 1.0

        sentiment_risk = (1 - sentiment_mult) * 40 * (w_news_adj / max(total_w, 1e-9))
        geo_risk_map = {"low": 0, "medium": 20, "high": 35, "critical": 40}
        geo_risk = geo_risk_map.get(geo["level"], 0) * (w_geo_adj / max(total_w, 1e-9))

        xq_risk = 0
        if xq_result and xq_result.get("sentiment") == "bearish":
            xq_risk = 10 * (w_xq_adj / max(total_w, 1e-9))

        earnings_risk = 0
        if earnings_result and earnings_result.get("rating") in ("C", "D"):
            earnings_risk = 10 * (w_earn_adj / max(total_w, 1e-9))

        risk_score = min(100, int(sentiment_risk + geo_risk + xq_risk + earnings_risk + fear_risk))

        # ── 7. 综合交易建议 ──────────────────────────────
        should_trade = geo["level"] != "critical" and sentiment["score"] > -0.5
        position_multiplier = min(sentiment_mult, geo_mult)

        if xq_result and xq_result.get("sentiment") == "bearish":
            position_multiplier *= 0.7
        if earnings_result and earnings_result.get("rating") == "D":
            position_multiplier *= 0.5
            should_trade = False

        risk_level = "低" if risk_score < 20 else ("中" if risk_score < 50 else "高")

        # ── 7. 生成摘要（含置信度标注） ──────────────────
        def _conf_tag(conf: float) -> str:
            if conf >= 0.7:
                return "高"
            elif conf >= 0.4:
                return "中"
            else:
                return "低"

        summary_lines = [
            "=" * 55,
            "  另类数据综合扫描（多源加权融合）",
            "=" * 55,
            f"  新闻情绪: {sentiment['score']:+.2f} ({sentiment['interpretation']}) "
            f"[置信度:{_conf_tag(news_conf)} 权重:{w_news_adj:.0%}]",
            f"  地缘风险: {geo['level']} (评分 {geo['score']}) "
            f"[置信度:{_conf_tag(geo_conf)} 权重:{w_geo_adj:.0%}]",
        ]
        if fear_index is not None:
            summary_lines.append(f"  恐慌指数: {fear_index}/100")
        if xq_result:
            summary_lines.append(
                f"  雪球情绪({symbol}): {xq_result.get('sentiment','?')} "
                f"({xq_result.get('score',0):+.2f}) "
                f"[置信度:{_conf_tag(xq_conf)} 权重:{w_xq_adj:.0%}]"
            )
        if earnings_result:
            summary_lines.append(
                f"  财报评级({symbol}): {earnings_result.get('rating','?')} "
                f"({earnings_result.get('score',0)}分) "
                f"[置信度:{_conf_tag(earn_conf)} 权重:{w_earn_adj:.0%}]"
            )
        summary_lines += [
            f"  {'-'*45}",
            f"  综合风险: {risk_score}/100 ({risk_level})",
            f"  仓位建议: {position_multiplier:.0%}",
            f"  交易建议: {'[OK] 可以交易' if should_trade else '[STOP] 建议暂停'}",
            "=" * 55,
        ]

        output = {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "fear_index": fear_index,
            "sentiment_score": sentiment["score"],
            "sentiment_detail": sentiment,
            "geopolitical_level": geo["level"],
            "geopolitical_detail": geo,
            "should_trade": should_trade,
            "position_multiplier": position_multiplier,
            "summary": "\n".join(summary_lines),
            "source_confidences": {
                "news_sentiment": {"confidence": news_conf, "weight": w_news_adj},
                "geopolitical": {"confidence": geo_conf, "weight": w_geo_adj},
                "xueqiu": {"confidence": xq_conf, "weight": w_xq_adj},
                "earnings": {"confidence": earn_conf, "weight": w_earn_adj},
            },
        }

        logger.info(f"综合风险: {risk_score}/100 ({risk_level}), 建议仓位: {position_multiplier:.0%}")
        return output

    def quick_check(self) -> bool:
        """快速检查：今天适合交易吗？"""
        result = self.full_scan()
        return result["should_trade"]


# ── 便捷函数 ──────────────────────────────────────────────

def should_trade_today() -> bool:
    """今天适合交易吗？（最简单用法）"""
    ad = AlternativeData()
    return ad.quick_check()


def get_position_multiplier() -> float:
    """获取建议仓位比例（0-1）"""
    ad = AlternativeData()
    result = ad.full_scan()
    return result["position_multiplier"]


# ── 命令行测试 ────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("另类数据综合扫描")
    print("=" * 60)

    ad = AlternativeData()
    result = ad.full_scan()
    print(result["summary"])
