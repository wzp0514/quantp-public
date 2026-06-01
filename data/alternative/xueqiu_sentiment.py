"""
雪球情绪分析 — 中国本土替代 Twitter/Reddit 社交媒体情绪

参考 Sattoro Hub 的"社交媒体情绪"功能。
中国版：雪球（xueqiu.com）— 最大的中文投资社交平台。

原理：
  1. 爬取雪球热帖（免费，无需登录）
  2. 按股票代码过滤相关讨论
  3. 根据标题和评论内容做多空情绪打分
  4. 输出个股/市场情绪指数

数据来源（公开，免费）：
  - 雪球热帖: https://xueqiu.com/statuses/hot/list.json
  - 个股讨论: https://xueqiu.com/statuses/search.json?q={symbol}

用法
--------
>>> from data.alternative.xueqiu_sentiment import XueqiuSentiment
>>> xs = XueqiuSentiment()
>>> result = xs.stock_sentiment("600519")    # 贵州茅台的情绪
>>> print(f"情绪: {result['sentiment']}")     # bullish/neutral/bearish
"""

import json
import logging
import re
import time
import urllib.request
from datetime import datetime
from typing import Optional

# 雪球 API 需要 Cookie（但不需登录，访问一次首页即可获取）
from config.log import get_logger

logger = get_logger("xueqiu_sentiment")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://xueqiu.com/",
}

# ============================================================
# A股多空关键词（中文投资社区常见表达）
# ============================================================

BULLISH_KEYWORDS = [
    "起飞", "涨停", "翻倍", "抄底", "满仓", "梭哈", "牛",
    "突破", "主升浪", "大利好", "业绩爆发", "超预期", "低估值",
    "回购", "增持", "分红", "价值投资", "长期持有", "坚定看好",
    "已上车", "已建仓", "已重仓", "格局", "不卖",
]

BEARISH_KEYWORDS = [
    "跑路", "割肉", "清仓", "踩雷", "暴雷", "跌停", "崩",
    "套牢", "站岗", "接盘", "韭菜", "收割", "出货",
    "造假", "退市", "亏损", "业绩暴雷", "不及预期", "高估值",
    "减持", "解禁", "地雷", "已清仓", "已割肉", "止损",
]

# 股票代码到雪球 symbol 的映射
# 上交所: SH + 代码, 深交所: SZ + 代码
def _to_xq_symbol(code: str) -> str:
    """600519 → SH600519, 000001 → SZ000001"""
    code = code.strip()
    if code.startswith("6"):
        return f"SH{code}"
    elif code.startswith(("0", "3")):
        return f"SZ{code}"
    return code


# ============================================================
# 雪球 API 请求
# ============================================================

def _fetch_xq_api(url: str, timeout: int = 15) -> Optional[dict]:
    """请求雪球 JSON API"""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            return json.loads(text) if text else None
    except Exception as e:
        logger.debug(f"雪球API请求失败: {e}")
        return None


def _get_xq_cookie() -> None:
    """获取雪球 Cookie（访问首页即可获得）"""
    try:
        req = urllib.request.Request("https://xueqiu.com/", headers={
            "User-Agent": HEADERS["User-Agent"],
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            cookies = resp.getheader("Set-Cookie")
            if cookies:
                HEADERS["Cookie"] = cookies
    except Exception:
        pass


def fetch_hot_posts(count: int = 30) -> list[dict]:
    """
    获取雪球热帖。

    返回
    -------
    list[dict]: [{title, text, symbol, reply_count, like_count, created_at}, ...]
    """
    _get_xq_cookie()
    all_posts = []

    # 雪球热帖 API（分页，每页 20 条）
    for page in range(1, 4):
        url = f"https://xueqiu.com/statuses/hot/list.json?page={page}&count=20"
        data = _fetch_xq_api(url)
        if not data:
            break

        items = data.get("list", [])
        for item in items:
            post = {
                "id": item.get("id"),
                "title": item.get("title", ""),
                "text": item.get("text", "") or item.get("description", ""),
                "symbol": item.get("target", ""),  # 关联股票代码
                "reply_count": item.get("reply_count", 0),
                "like_count": item.get("like_count", 0),
                "retweet_count": item.get("retweet_count", 0),
                "created_at": datetime.fromtimestamp(
                    item.get("created_at", 0) / 1000
                ).isoformat() if item.get("created_at") else "",
            }
            all_posts.append(post)
            if len(all_posts) >= count:
                break

        if len(all_posts) >= count:
            break
        time.sleep(1)

    return all_posts[:count]


def fetch_stock_posts(symbol: str, count: int = 20) -> list[dict]:
    """
    获取某只股票的雪球讨论帖。

    参数
    ----------
    symbol : str
        A股代码，如 "600519"（贵州茅台）
    count : int
        获取数量
    """
    xq_symbol = _to_xq_symbol(symbol)
    _get_xq_cookie()

    posts = []
    for page in range(1, 4):
        url = (
            f"https://xueqiu.com/statuses/search.json?"
            f"q={xq_symbol}&count=15&page={page}&comment=0"
        )
        data = _fetch_xq_api(url)
        if not data:
            break

        items = data.get("list", [])
        for item in items:
            title = item.get("title", "") or item.get("description", "")
            text = item.get("text", "") or item.get("description", "")
            posts.append({
                "id": item.get("id"),
                "title": title,
                "text": text,
                "reply_count": item.get("reply_count", 0),
                "like_count": item.get("like_count", 0),
                "created_at": datetime.fromtimestamp(
                    item.get("created_at", 0) / 1000
                ).isoformat() if item.get("created_at") else "",
            })
            if len(posts) >= count:
                break

        if len(posts) >= count:
            break
        time.sleep(1)

    return posts[:count]


# ============================================================
# 情绪分析
# ============================================================

def _score_text(text: str) -> dict:
    """单条文本多空打分"""
    score = 0
    bull_matches = []
    bear_matches = []

    for kw in BULLISH_KEYWORDS:
        if kw in text:
            score += 1
            bull_matches.append(kw)

    for kw in BEARISH_KEYWORDS:
        if kw in text:
            score -= 1
            bear_matches.append(kw)

    return {"score": score, "bullish": bull_matches, "bearish": bear_matches}


def _interpret_sentiment(score: float) -> str:
    if score > 0.2:
        return "bullish"
    elif score < -0.2:
        return "bearish"
    return "neutral"


class XueqiuSentiment:
    """雪球情绪分析器"""

    def market_sentiment(self, count: int = 30) -> dict:
        """
        市场整体情绪。

        从热帖中提取市场情绪指数。
        """
        posts = fetch_hot_posts(count)
        if not posts:
            return {"sentiment": "neutral", "score": 0.0, "error": "无数据"}

        total_score = 0
        bull_count = 0
        bear_count = 0

        for p in posts:
            combined = p["title"] + " " + p["text"]
            s = _score_text(combined)
            total_score += s["score"]
            if s["score"] > 0:
                bull_count += 1
            elif s["score"] < 0:
                bear_count += 1

        # 按互动量加权
        weighted_score = total_score / max(len(posts), 1)
        normalized = max(-1.0, min(1.0, weighted_score / 3))

        result = {
            "sentiment": _interpret_sentiment(normalized),
            "score": round(normalized, 3),
            "total_posts": len(posts),
            "bullish_posts": bull_count,
            "bearish_posts": bear_count,
            "neutral_posts": len(posts) - bull_count - bear_count,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"雪球市场情绪: {normalized:.2f} ({_interpret_sentiment(normalized)}) "
            f"看多{bull_count}/看空{bear_count}/{len(posts)}帖"
        )
        return result

    def stock_sentiment(self, symbol: str, count: int = 20) -> dict:
        """
        个股情绪。

        参数
        ----------
        symbol : str
            A股代码，如 "600519"
        """
        posts = fetch_stock_posts(symbol, count)
        if not posts:
            return {
                "symbol": symbol,
                "sentiment": "neutral",
                "score": 0.0,
                "error": "无数据",
            }

        total_score = 0
        bull_count = 0
        bear_count = 0
        top_bull = []
        top_bear = []

        for p in posts:
            combined = p["title"] + " " + p["text"]
            s = _score_text(combined)
            total_score += s["score"]
            if s["score"] > 0:
                bull_count += 1
                top_bull.append((s["score"], p["title"][:60]))
            elif s["score"] < 0:
                bear_count += 1
                top_bear.append((s["score"], p["title"][:60]))

        weighted_score = total_score / max(len(posts), 1)
        normalized = max(-1.0, min(1.0, weighted_score / 3))

        result = {
            "symbol": symbol,
            "sentiment": _interpret_sentiment(normalized),
            "score": round(normalized, 3),
            "total_posts": len(posts),
            "bullish_posts": bull_count,
            "bearish_posts": bear_count,
            "top_bullish": [t for s, t in sorted(top_bull, reverse=True)[:3]],
            "top_bearish": [t for s, t in sorted(top_bear)[:3]],
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"雪球个股情绪({symbol}): {normalized:.2f} "
            f"看多{bull_count}/看空{bear_count}/{len(posts)}帖"
        )
        return result

    def as_multiplier(self, sentiment_result: dict) -> float:
        """
        将情绪结果转为仓位乘数。

        bullish=1.0, neutral=0.7, bearish=0.3
        """
        mapping = {"bullish": 1.0, "neutral": 0.7, "bearish": 0.3}
        return mapping.get(sentiment_result.get("sentiment", "neutral"), 0.7)


# ============================================================
# 命令行测试
# ============================================================
# python data/alternative/xueqiu_sentiment.py

if __name__ == "__main__":
    print("=" * 60)
    print("雪球情绪分析")
    print("=" * 60)

    import sys
    sys.path.insert(0, ".")

    xs = XueqiuSentiment()

    print("\n[市场整体情绪]")
    market = xs.market_sentiment(20)
    print(f"  情绪: {market.get('sentiment', 'N/A')}")
    print(f"  评分: {market.get('score', 0):.2f}")
    print(f"  帖子: {market.get('total_posts', 0)}")
