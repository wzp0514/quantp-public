"""
新闻情绪分析 — 参考 Sattoro Hub 的 AI+GenAI 财报电话会议/新闻情绪分析

免费方案：RSS 新闻源 + 中文关键词情感打分 → 每日情绪指数。

不需要付费 API（Bloomberg/Refinitiv 年费 3-17 万），用公开 RSS 即可。

原理：
  1. 从多个中文财经 RSS 源拉取当日新闻标题
  2. 用关键词词典对每条标题打分（正面词+1，负面词-1，中性词0）
  3. 汇总所有标题 → 每日情绪指数（-1=极度悲观, +1=极度乐观）
  4. 可作为策略过滤器：情绪极差时减仓或不交易

用法
--------
>>> from data.alternative.news_sentiment import NewsSentiment
>>> ns = NewsSentiment()
>>> result = ns.analyze()              # 获取今日情绪
>>> print(f"情绪指数: {result['score']:.2f}")     # -1 ~ +1
>>> print(f"解读: {result['interpretation']}")    # 乐观/中性/悲观
"""

import logging
import re
import time
from datetime import date, datetime
from typing import Optional
from urllib.parse import urljoin

import urllib.request
import xml.etree.ElementTree as ET

from config.log import get_logger

logger = get_logger("news_sentiment")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

# ============================================================
# 免费 RSS 新闻源（中文财经）
# ============================================================

RSS_SOURCES = {
    "东方财富-要闻": "https://rsshub.app/eastmoney/search/要闻",
    "证券时报-快讯": "https://rsshub.app/stcn/kuaixun",
    "华尔街见闻-最新": "https://rsshub.app/wallstreetcn/news/global",
    # 备用（可能需要 VPN）:
    # "Reuters-中国": "https://rsshub.app/reuters/world/china",
}

# ============================================================
# 中文情感关键词词典
# ============================================================

# 正面词（出现在标题中 → +1）
POSITIVE_WORDS = {
    "大涨", "暴涨", "涨停", "利好", "突破", "创新高", "反弹", "走强",
    "增长", "盈利", "分红", "回购", "增持", "降息", "宽松", "刺激",
    "复苏", "改善", "超预期", "扩张", "加仓", "资金流入", "看好",
    "政策支持", "减税", "补贴", "协议达成", "谈判进展",
}

# 负面词（出现在标题中 → -1）
NEGATIVE_WORDS = {
    "大跌", "暴跌", "跌停", "利空", "崩盘", "恐慌", "抛售", "走弱",
    "亏损", "债务", "违约", "暴雷", "退市", "加息", "收紧", "制裁",
    "衰退", "恶化", "不及预期", "萎缩", "减仓", "资金流出", "看空",
    "贸易战", "关税", "制裁", "地缘", "冲突", "战争", "危机",
}

# 中性词（对冲掉一些极端判断）
NEUTRAL_WORDS = {
    "震荡", "波动", "盘整", "横盘", "观望",
}


def _fetch_rss(url: str, timeout: int = 15) -> list[str]:
    """从 RSS 源获取标题列表"""
    titles = []
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read().decode("utf-8", errors="replace")

        # 解析 XML（RSS 标准格式）
        root = ET.fromstring(content)

        # RSS 2.0: <channel><item><title>...</title></item></channel>
        for item in root.iter("item"):
            title_elem = item.find("title")
            if title_elem is not None and title_elem.text:
                titles.append(title_elem.text.strip())

        # Atom: <entry><title>...</title></entry>
        for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
            title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
            if title_elem is not None and title_elem.text:
                titles.append(title_elem.text.strip())

        logger.debug(f"RSS {url}: {len(titles)} 条标题")

    except Exception as e:
        logger.warning(f"RSS 获取失败: {url} — {e}")

    return titles


def _score_title(title: str) -> float:
    """单条标题情感打分"""
    score = 0.0
    for word in POSITIVE_WORDS:
        if word in title:
            score += 1.0
    for word in NEGATIVE_WORDS:
        if word in title:
            score -= 1.0
    # 中性词不改变分数，只记录
    return score


def _interpret(score: float) -> str:
    """将分数映射为人可读的解读"""
    if score > 0.3:
        return "乐观（市场情绪积极，利好做多）"
    elif score > 0.1:
        return "偏乐观（轻微正面，正常交易）"
    elif score > -0.1:
        return "中性（市场方向不明，谨慎交易）"
    elif score > -0.3:
        return "偏悲观（轻微负面，考虑减仓）"
    else:
        return "悲观（市场恐慌，建议暂停交易或减仓）"


class NewsSentiment:
    """新闻情绪分析器"""

    def __init__(self):
        self.last_result: Optional[dict] = None

    def analyze(self, sources: list[str] = None) -> dict:
        """
        分析当前市场情绪。

        参数
        ----------
        sources : list[str]
            要使用的 RSS 源名（RSS_SOURCES 的 key）。不填=全部

        返回
        -------
        dict: {score, interpretation, articles, positive_count, negative_count, timestamp}
        """
        src_names = sources or list(RSS_SOURCES.keys())
        all_titles = []

        for name in src_names:
            url = RSS_SOURCES.get(name)
            if not url:
                continue
            titles = _fetch_rss(url)
            all_titles.extend(titles)
            time.sleep(0.5)  # 源间短暂间隔

        if not all_titles:
            logger.warning("无新闻数据，返回中性情绪")
            return {
                "score": 0.0,
                "interpretation": "无数据（中性）",
                "articles": 0,
                "positive_count": 0,
                "negative_count": 0,
                "timestamp": datetime.now().isoformat(),
                "confidence": 0.1,  # 无数据 → 置信度极低
            }

        # 逐条打分
        pos_count = 0
        neg_count = 0
        total_score = 0.0
        scored_titles = []

        for title in all_titles:
            s = _score_title(title)
            if s > 0:
                pos_count += 1
            elif s < 0:
                neg_count += 1
            total_score += s
            if s != 0:
                scored_titles.append((s, title))

        # 归一化：总分 / 有情绪的文章数（避免大量中性文章稀释）
        emotional_count = pos_count + neg_count
        if emotional_count > 0:
            score = max(-1.0, min(1.0, total_score / emotional_count * 2))
        else:
            score = 0.0

        # 置信度：成功源比例 × 文章数量因子 (min(1, articles/20))
        src_success_rate = len(set(src_names)) / len(src_names) if src_names else 0.0
        article_factor = min(1.0, len(all_titles) / 20)
        confidence = round(src_success_rate * 0.6 + article_factor * 0.4, 2)

        result = {
            "score": round(score, 3),
            "interpretation": _interpret(score),
            "articles": len(all_titles),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "top_positive": [t for s, t in sorted(scored_titles, reverse=True)[:3] if s > 0],
            "top_negative": [t for s, t in sorted(scored_titles)[:3] if s < 0],
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence,
        }

        self.last_result = result
        logger.info(f"情绪指数: {score:.2f} ({pos_count}+/{neg_count}-/{len(all_titles)}条)")
        return result

    def should_trade(self, threshold: float = -0.3) -> tuple[bool, str]:
        """
        判断当前是否应该交易。

        情绪极差时不建议交易（可作为策略过滤器）。

        返回
        -------
        (是否应该交易, 原因)
        """
        if self.last_result is None:
            self.analyze()

        score = self.last_result["score"]
        if score <= threshold:
            return False, f"情绪指数 {score:.2f} ≤ {threshold}（极度悲观，暂停交易）"
        return True, f"情绪指数 {score:.2f} > {threshold}（正常交易）"

    def as_filter(self) -> float:
        """
        作为策略信号过滤器。

        返回 0~1 的乘数（情绪好=1.0 全仓，情绪差=0.0 空仓）。

        用法：在策略中乘以仓位大小。
        """
        if self.last_result is None:
            self.analyze()
        score = self.last_result["score"]
        # 映射 (-1, 1) → (0, 1)
        return max(0.0, min(1.0, (score + 1) / 2))


# ============================================================
# 命令行测试
# ============================================================
# python data/alternative/news_sentiment.py

if __name__ == "__main__":
    print("=" * 60)
    print("新闻情绪分析")
    print("=" * 60)

    ns = NewsSentiment()
    result = ns.analyze()

    print(f"\n情绪指数: {result['score']:.2f}")
    print(f"解读: {result['interpretation']}")
    print(f"新闻总数: {result['articles']}")
    print(f"正面: {result['positive_count']}, 负面: {result['negative_count']}")

    if result["top_positive"]:
        print("\n最正面标题:")
        for t in result["top_positive"]:
            print(f"  + {t}")
    if result["top_negative"]:
        print("\n最负面标题:")
        for t in result["top_negative"]:
            print(f"  - {t}")

    trade_ok, reason = ns.should_trade()
    print(f"\n交易建议: {'可以交易' if trade_ok else '建议暂停'} — {reason}")
