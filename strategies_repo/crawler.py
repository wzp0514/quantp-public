"""
策略爬虫 — 从量化社区平台爬取公开策略

支持来源：
  - TradingView Pine Script 仓库（无需登录，公开可访问）
  - 聚宽 (JoinQuant) 策略广场（需登录 session 才能查看代码）

TradingView 爬取策略：
  列表页 → 提取 __NEXT_DATA__ JSON → 逐个脚本详情页 → 提取 Pine Script 源码 → 入库

聚宽爬取策略：
  首页获取 cookie → 策略列表页(HTML) → 逐个详情页 → 提取 Python 代码 → 入库
  注：查看策略代码需要登录。可通过 cookie_str 参数传入浏览器 cookie。

重要提醒：
  - 仅爬取公开策略
  - 请求间隔 >= 2 秒（避免给服务器造成压力）
  - 仅供个人学习研究，不可商用

用法
--------
>>> from strategies_repo.crawler import crawl_tradingview_top, crawl_joinquant
>>> crawl_tradingview_top(top_n=10)
>>> crawl_joinquant(max_pages=3, cookie_str="PHPSESSID=xxx")  # 需登录cookie
"""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

from strategies_repo.repo import StrategyRepo
from config.log import get_logger

logger = get_logger("crawler")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

REPO_ROOT = Path(__file__).resolve().parent

# 代理配置（如需翻墙访问 TradingView 等境外站点，请设置）
# 格式: {"http": "http://<proxy_host>:<proxy_port>", "https": "http://<proxy_host>:<proxy_port>"}
PROXY = None

CrawlStats = dict  # {"success": int, "no_code": int, "auth_required": int, "network_error": int, "parse_error": int}


def set_proxy(http_proxy: str = "", https_proxy: str = ""):
    """设置爬虫代理"""
    global PROXY
    if http_proxy or https_proxy:
        PROXY = {}
        if http_proxy:
            PROXY["http"] = http_proxy
        if https_proxy:
            PROXY["https"] = https_proxy
    else:
        PROXY = None


def _new_stats() -> CrawlStats:
    return {"success": 0, "no_code": 0, "auth_required": 0, "network_error": 0, "parse_error": 0}


def _make_session(cookie_str: str = "") -> requests.Session:
    """创建带 cookie 和代理的 requests Session"""
    session = requests.Session()
    session.headers.update(HEADERS)
    if PROXY:
        session.proxies.update(PROXY)
    if cookie_str:
        for item in cookie_str.split(";"):
            item = item.strip()
            if "=" in item:
                k, v = item.split("=", 1)
                session.cookies.set(k.strip(), v.strip())
    return session


def _fetch(session: requests.Session, url: str, timeout: int = 30,
           max_retries: int = 3) -> Optional[str]:
    """HTTP GET 请求，带指数退避重试"""
    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 403:
                logger.warning(f"请求被拒(403): {url[:80]}")
                return None
            if resp.status_code == 404:
                logger.debug(f"页面不存在(404): {url[:80]}")
                return None
            resp.raise_for_status()
            return resp.text
        except requests.Timeout:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.debug(f"超时，{wait}s 后重试({attempt + 1}/{max_retries}): {url[:60]}")
                time.sleep(wait)
            else:
                logger.warning(f"请求超时(已重试{max_retries}次): {url[:80]}")
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.warning(f"请求失败: {url[:80]} — {e}")
    return None


def _post_json(session: requests.Session, url: str, data: dict,
               timeout: int = 30, max_retries: int = 3) -> Optional[dict]:
    """HTTP POST JSON 请求，带重试"""
    for attempt in range(max_retries):
        try:
            resp = session.post(url, json=data,
                headers={**session.headers,
                         "Content-Type": "application/json",
                         "X-Requested-With": "XMLHttpRequest"},
                timeout=timeout)
            if resp.status_code == 403:
                logger.warning(f"POST 被拒(403): {url[:80]}")
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.warning(f"POST 超时: {url[:80]}")
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.warning(f"POST 失败: {url[:80]} — {e}")
        except json.JSONDecodeError:
            logger.warning(f"POST 返回非 JSON: {url[:80]}")
            return None
    return None


# ============================================================
# TradingView 爬虫（无需登录）
# ============================================================

TV_LIST_URL = "https://www.tradingview.com/scripts/"
TV_SCRIPT_URL = "https://www.tradingview.com/script/{script_id}-{slug}/"


def crawl_tradingview(
    max_scripts: int = 30,
    sort_by: str = "rating",
    repo: StrategyRepo = None,
    category: str = "market",
) -> tuple[int, CrawlStats]:
    """
    从 TradingView 公开脚本库爬取 Pine Script 策略。

    TradingView 使用 Next.js SSR，脚本数据嵌入在
    <script id="__NEXT_DATA__" type="application/json"> 中。
    无需登录即可获取公开脚本。

    参数
    ----------
    max_scripts : int   最多获取多少个脚本
    sort_by : str       排序: "rating" / "popular" / "trending"
    repo : StrategyRepo
    category : str

    返回
    -------
    (imported_count, stats)
    """
    if repo is None:
        repo = StrategyRepo()

    stats = _new_stats()
    logger.info(f"TradingView 爬虫启动: 最多 {max_scripts} 个, 排序={sort_by}")

    session = _make_session()

    # Step 1: 获取脚本列表页
    list_url = f"{TV_LIST_URL}?sort={sort_by}"
    html = _fetch(session, list_url)
    if html is None:
        logger.error("TradingView 列表页请求失败，请检查网络")
        return 0, stats

    script_ids = _extract_tv_script_ids(html)
    if not script_ids:
        logger.error("未找到脚本 ID，TradingView 页面结构可能已变化")
        return 0, stats

    logger.info(f"找到 {len(script_ids)} 个脚本")

    # Step 2: 逐个获取脚本详情
    imported = 0
    for i, (sid, slug) in enumerate(script_ids[:max_scripts]):
        try:
            detail_url = TV_SCRIPT_URL.format(script_id=sid, slug=slug)
            detail_html = _fetch(session, detail_url)
            if detail_html is None:
                stats["network_error"] += 1
                time.sleep(2)
                continue

            info = _extract_tv_script_info(detail_html, sid)
            if info is None:
                stats["parse_error"] += 1
                time.sleep(2)
                continue

            name = f"tv_{info['title']}"[:80]
            safe_name = name.replace("/", "_").replace("\\", "_")
            if (repo.root / category / safe_name).exists():
                logger.debug(f"  跳过(已存在): {info['title'][:40]}")
                time.sleep(1)
                continue

            code = _extract_tv_pine_code(detail_html)
            if not code or len(code) < 50:
                stats["no_code"] += 1
                logger.debug(f"  跳过 {info['title'][:40]}: 无有效代码")
                time.sleep(2)
                continue

            repo.import_from_market(
                name=name,
                source="TradingView社区",
                source_url=f"https://www.tradingview.com/script/{sid}-{slug}/",
                stype=info.get("type", "unknown"),
                desc=info.get("desc", ""),
                params={},
                strategy_code=(
                    f"// Pine Script v5\n"
                    f"// TradingView ID: {sid}\n"
                    f"// Author: {info.get('author', 'unknown')}\n"
                    f"// Rating: {info.get('rating', 'N/A')}\n\n{code}"
                ),
                category=category,
                tags=["tradingview", "pinescript"] + (info.get("tags", []) or []),
            )
            _update_meta_rating(repo, category, name, info.get("rating", 0))

            imported += 1
            stats["success"] += 1
            logger.info(f"  [{imported}] {info['title'][:40]} | [*]{info.get('rating', '?')}")

            time.sleep(2)

        except Exception as e:
            stats["parse_error"] += 1
            logger.debug(f"  跳过 {sid}: {e}")
            time.sleep(2)

    logger.info(f"TradingView 爬取完成: 成功 {stats['success']}, "
                f"无代码 {stats['no_code']}, 网络错误 {stats['network_error']}")
    return imported, stats


def crawl_tradingview_top(
    top_n: int = 20,
    min_rating: float = 3.5,
    repo: StrategyRepo = None,
) -> int:
    """获取评分最高的 N 个 TradingView 策略"""
    if repo is None:
        repo = StrategyRepo()
    logger.info(f"TradingView Top {top_n} 策略（最低评分 {min_rating}[*]）")
    n, _ = crawl_tradingview(max_scripts=top_n * 3, sort_by="rating", repo=repo)
    return n


def _extract_tv_script_ids(html: str) -> list[tuple[str, str]]:
    """
    从 TradingView 脚本列表页提取脚本 ID 和 slug。

    TradingView 脚本 URL 格式: /script/{id}-{slug}/
    列表页是服务端渲染 HTML，脚本链接嵌在 <a href> 中。

    返回 [(script_id, slug), ...]
    """
    results = []

    # 方法1: 从 HTML <a href> 中提取 /script/XXXXX-slug/ 模式
    matches = re.findall(r'/script/([a-zA-Z0-9]+)-([a-zA-Z0-9-]+?)(?:/|")', html)
    if matches:
        seen = set()
        for sid, slug in matches:
            if sid not in seen and len(sid) > 3:
                seen.add(sid)
                results.append((sid, slug))
        if results:
            return results[:100]

    # 方法2: __NEXT_DATA__ JSON (Next.js 渲染时)
    m = re.search(r'<script\s+id="__NEXT_DATA__"[^>]*>\s*({.*?})\s*</script>', html, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))

            def _find_scripts(obj):
                if isinstance(obj, dict):
                    sid = obj.get("id") or obj.get("scriptId")
                    slug = obj.get("slug") or obj.get("name") or ""
                    if sid and isinstance(sid, str) and len(sid) > 3:
                        slug = slug.lower().replace(" ", "-") if slug else sid
                        results.append((sid, slug))
                    for v in obj.values():
                        _find_scripts(v)
                elif isinstance(obj, list):
                    for item in obj:
                        _find_scripts(item)

            _find_scripts(data)
        except json.JSONDecodeError:
            pass

    if results:
        seen = set()
        unique = []
        for sid, slug in results:
            if sid not in seen:
                seen.add(sid)
                unique.append((sid, slug))
        return unique[:100]

    # 方法3: JSON 中的 scriptId 字段
    matches = re.findall(r'"scriptId"\s*:\s*"([^"]+)"', html)
    if matches:
        return [(sid, sid) for sid in list(dict.fromkeys(matches))[:100]]

    return []


def _extract_tv_script_info(html: str, script_id: str) -> Optional[dict]:
    """
    从 TradingView 脚本详情页提取元信息。

    TradingView 脚本详情页是服务端渲染 HTML，
    优先从 __NEXT_DATA__ JSON 提取，回退到 HTML 解析。
    """
    info = {"script_id": script_id}

    m = re.search(r'<script\s+id="__NEXT_DATA__"[^>]*>\s*({.*?})\s*</script>', html, re.DOTALL)
    next_data = None
    if m:
        try:
            next_data = json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    if next_data:
        props = next_data.get("props", {}).get("pageProps", {})
        script_data = props.get("script", props)

        info["title"] = (script_data.get("title") or script_data.get("name")
                         or props.get("title") or f"TV_{script_id}")[:80]
        author = script_data.get("author", {})
        info["author"] = (author.get("username") if isinstance(author, dict) else author) or "unknown"
        rating = script_data.get("rating") or script_data.get("votes", {}).get("rating")
        info["rating"] = float(rating) if rating else 0.0
        info["desc"] = (script_data.get("description") or script_data.get("short_description")
                        or f"TradingView script: {info['title']}")[:200]
    else:
        # 回退：HTML 正则提取
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.text if title_tag else script_id
        title = re.sub(r'\s*[—–-]\s*TradingView.*$', '', title).strip()
        info["title"] = title[:80] if title else script_id

        m = re.search(r'"username"\s*:\s*"([^"]+)"', html)
        info["author"] = m.group(1) if m else "unknown"
        m = re.search(r'"rating"\s*:\s*([\d.]+)', html)
        info["rating"] = float(m.group(1)) if m else 0.0
        m = re.search(r'"description"\s*:\s*"([^"]+)"', html)
        info["desc"] = (m.group(1)[:200] if m else f"TradingView script: {info['title']}")

    # 猜测策略类型
    text = (info["title"] + " " + info["desc"]).lower()
    if any(w in text for w in ["ma ", "sma", "ema", "moving average", "均线"]):
        info["type"] = "趋势跟踪"; info["tags"] = ["均线"]
    elif any(w in text for w in ["rsi", "macd", "bollinger", "布林"]):
        info["type"] = "均值回归"; info["tags"] = ["指标"]
    elif any(w in text for w in ["breakout", "突破", "channel", "通道"]):
        info["type"] = "突破"; info["tags"] = ["突破"]
    elif any(w in text for w in ["grid", "网格"]):
        info["type"] = "震荡"; info["tags"] = ["网格"]
    else:
        info["type"] = "unknown"; info["tags"] = []

    return info


def _extract_tv_pine_code(html: str) -> Optional[str]:
    """
    从 TradingView 脚本详情页提取 Pine Script 源码。

    源码在 __NEXT_DATA__ JSON 的 source 字段，或页面内嵌的 JSON blob 中。
    """
    # 方法1: __NEXT_DATA__ JSON 递归搜索 source 字段
    m = re.search(r'<script\s+id="__NEXT_DATA__"[^>]*>\s*({.*?})\s*</script>', html, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))

            def _find_source(obj):
                if isinstance(obj, dict):
                    for k in ("source", "code", "scriptSource", "content"):
                        if k in obj and isinstance(obj[k], str) and len(obj[k]) > 50:
                            # 确认是 Pine Script（包含 indicator/strategy 关键字）
                            if any(kw in obj[k] for kw in
                                   ("indicator", "strategy", "plot", "//@version")):
                                return obj[k]
                    for v in obj.values():
                        r = _find_source(v)
                        if r:
                            return r
                elif isinstance(obj, list):
                    for item in obj:
                        r = _find_source(item)
                        if r:
                            return r
                return None

            code = _find_source(data)
            if code:
                return code
        except json.JSONDecodeError:
            pass

    # 方法2: <code> / <pre> 标签中的 Pine Script
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["code", "pre"]):
        text = tag.get_text()
        if len(text) > 100 and any(kw in text.lower()
                                   for kw in ("//@version", "indicator", "strategy",
                                              "plot(", "ta.", "input.")):
            return text

    # 方法3: JSON 字符串中的 source 字段
    m = re.search(r'"source"\s*:\s*"((?:[^"\\]|\\.)*)"', html)
    if m:
        code = m.group(1)
        code = code.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
        if len(code) > 50:
            return code

    return None


# ============================================================
# 聚宽 (JoinQuant) 爬虫（需登录 session）
# ============================================================

JQ_HOME = "https://www.joinquant.com"
JQ_LIST = f"{JQ_HOME}/algorithm/list"
JQ_DETAIL = f"{JQ_HOME}/algorithm/index/edit"  # ?algorithmId=XXX
JQ_APISHARE_LIST = f"{JQ_HOME}/algorithm/apishare/list"
JQ_APISHARE_GET = f"{JQ_HOME}/algorithm/apishare/get"  # ?apiId=XXX


def crawl_joinquant(
    max_pages: int = 10,
    repo: StrategyRepo = None,
    category: str = "market",
    cookie_str: str = "",
) -> tuple[int, CrawlStats]:
    """
    从聚宽策略广场爬取公开策略。

    聚宽策略广场是 React SPA，策略列表页返回 HTML，
    详情页 URL 为 /algorithm/index/edit?algorithmId=XXX。

    **查看策略代码需要登录**。请从浏览器获取 cookie 后传入 cookie_str。
    获取方法：F12 → Application → Cookies → 复制 PHPSESSID 等关键 cookie。

    参数
    ----------
    max_pages : int      爬取页数
    repo : StrategyRepo
    category : str
    cookie_str : str     浏览器 cookie（登录后复制，格式 "key1=val1; key2=val2"）

    返回
    -------
    (imported_count, stats)
    """
    if repo is None:
        repo = StrategyRepo()

    stats = _new_stats()

    if not cookie_str:
        logger.warning("聚宽未提供 cookie，将只爬取无需登录的公开信息")
        logger.warning("获取 cookie: F12 → Application → Cookies → 复制 PHPSESSID 等")

    logger.info(f"聚宽爬虫启动: 最多 {max_pages} 页")

    session = _make_session(cookie_str)

    # Step 0: 访问首页获取基础 cookie
    home_html = _fetch(session, JQ_HOME)
    if home_html is None:
        logger.error("聚宽首页无法访问，请检查网络")
        return 0, stats

    # 检查是否已登录
    logged_in = not _needs_login(home_html)
    if logged_in:
        logger.info("检测到登录状态")
    else:
        logger.warning("未登录，策略代码将无法获取。请提供登录后的 cookie。")

    imported = 0

    for page in range(1, max_pages + 1):
        logger.info(f"第 {page}/{max_pages} 页...")

        try:
            # 聚宽策略列表页（GET，返回 HTML）
            list_url = f"{JQ_LIST}?page={page}&count=20&type=1"
            list_html = _fetch(session, list_url)

            if list_html is None:
                stats["network_error"] += 1
                break

            # 从列表 HTML 中提取策略 ID
            algo_ids = _extract_jq_algo_ids(list_html)
            if not algo_ids:
                logger.info(f"第 {page} 页无策略数据，可能已到底")
                break

            logger.info(f"  本页 {len(algo_ids)} 个策略")

            for algo_id in algo_ids:
                safe_name = f"jq_{algo_id}"
                if (repo.root / category / safe_name).exists():
                    continue

                # 聚宽策略详情页
                detail_url = f"{JQ_DETAIL}?algorithmId={algo_id}"
                detail_html = _fetch(session, detail_url)

                if detail_html is None:
                    stats["network_error"] += 1
                    time.sleep(2)
                    continue

                if _needs_login(detail_html):
                    stats["auth_required"] += 1
                    logger.debug(f"  需登录: algo_id={algo_id}")
                    time.sleep(2)
                    continue

                # 提取策略名和描述
                algo_name = _extract_jq_name(detail_html) or f"聚宽策略{algo_id}"
                algo_desc = _extract_jq_desc(detail_html) or ""

                # 提取 Python 代码
                code = _extract_code_joinquant(detail_html)
                if not code or len(code) < 100:
                    stats["no_code"] += 1
                    time.sleep(2)
                    continue

                params = _extract_params_joinquant(detail_html)

                repo.import_from_market(
                    name=f"聚宽_{algo_name}",
                    source="聚宽社区",
                    source_url=detail_url,
                    stype="unknown",
                    desc=algo_desc[:200] if algo_desc else algo_name,
                    params=params,
                    strategy_code=code,
                    category=category,
                    tags=["joinquant", "聚宽", "A股"],
                )
                imported += 1
                stats["success"] += 1
                logger.info(f"  导入: {algo_name}")

                time.sleep(2)

        except Exception as e:
            logger.warning(f"第 {page} 页爬取异常: {e}")
            stats["parse_error"] += 1
            break

        time.sleep(3)

    logger.info(f"聚宽爬取完成: 成功 {stats['success']}, 无代码 {stats['no_code']}, "
                f"需登录 {stats['auth_required']}, 网络错误 {stats['network_error']}")
    return imported, stats


def _needs_login(html: str) -> bool:
    """判断页面是否需要登录"""
    indicators = [
        "请先登录", "登录后查看", "需要登录",
        "login required", "please login", "sign in",
        "unauthorized", "尚未登录", "请登录",
        "window.location.href='/user/login'",
    ]
    text_lower = html.lower()
    return any(ind.lower() in text_lower for ind in indicators)


def _extract_jq_algo_ids(html: str) -> list[str]:
    """从聚宽策略列表页提取策略 ID 列表"""
    ids = []

    # 方法1: 从嵌入式 JSON/JS 数据提取
    for pattern in [
        r'"algorithmId"\s*:\s*(\d+)',
        r'"id"\s*:\s*(\d+)',
        r'/algorithm/index/edit\?algorithmId=(\d+)',
    ]:
        matches = re.findall(pattern, html)
        ids.extend(matches)

    # 方法2: 从 HTML 链接提取
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = re.search(r'algorithmId=(\d+)', href)
        if m:
            ids.append(m.group(1))

    # 去重保序
    return list(dict.fromkeys(ids))


def _extract_jq_name(html: str) -> Optional[str]:
    """从聚宽策略详情页提取策略名称"""
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)
        title = re.sub(r'\s*[-–—|].*$', '', title).strip()
        if len(title) > 2:
            return title[:80]

    m = re.search(r'"name"\s*:\s*"([^"]+)"', html)
    if m:
        return m.group(1)[:80]

    m = re.search(r'"title"\s*:\s*"([^"]+)"', html)
    if m:
        return m.group(1)[:80]

    return None


def _extract_jq_desc(html: str) -> Optional[str]:
    """从聚宽策略详情页提取描述"""
    m = re.search(r'"description"\s*:\s*"([^"]+)"', html)
    if m:
        return m.group(1)[:200]
    m = re.search(r'"desc"\s*:\s*"([^"]+)"', html)
    if m:
        return m.group(1)[:200]
    return None


def _extract_code_joinquant(html: str) -> Optional[str]:
    """
    从聚宽策略详情页提取 Python 代码。

    代码可能嵌入在：
      1. window.__INITIAL_STATE__ 或类似 JS 全局变量中的 JSON
      2. <script> 标签的 JSON blob
      3. <textarea> 或 CodeMirror 编辑器
      4. <pre>/<code> 标签
    """
    # 方法1: __INITIAL_STATE__ 或全局 JS 变量
    for var_name in ["__INITIAL_STATE__", "__DATA__", "__PRELOADED_STATE__"]:
        m = re.search(
            r'(?:window\.)?' + var_name + r'\s*=\s*({.*?});\s*(?:</script>|var\s|window\.)',
            html, re.DOTALL
        )
        if not m:
            m = re.search(
                r'(?:window\.)?' + var_name + r'\s*=\s*({.*?})\s*</script>',
                html, re.DOTALL
            )
        if m:
            try:
                data = json.loads(m.group(1))
                code = _search_json_for_code(data)
                if code:
                    return code
            except json.JSONDecodeError:
                pass

    # 方法2: 直接在 HTML 中搜索 JSON 里的 code 字段
    for field in ["code", "sourceCode", "algorithmCode", "source"]:
        m = re.search(
            r'"' + field + r'"\s*:\s*"((?:[^"\\]|\\.)*?)"\s*[,}\]]',
            html
        )
        if m:
            code = m.group(1)
            code = (code.replace("\\n", "\n").replace("\\t", "\t")
                    .replace("\\r", "").replace('\\"', '"').replace("\\\\", "\\"))
            if len(code) > 100 and ("def " in code or "import " in code):
                return code

    # 方法3: CodeMirror 编辑器行
    lines = re.findall(r'<pre[^>]*CodeMirror-line[^>]*[^>]*>(.*?)</pre>', html)
    if lines:
        import html as html_mod
        code = "\n".join(html_mod.unescape(l) for l in lines)
        if len(code) > 100:
            return code

    # 方法4: <textarea> 或 <pre>/<code> 标签
    soup = BeautifulSoup(html, "html.parser")
    for tag_name in ["textarea", "pre", "code"]:
        for tag in soup.find_all(tag_name):
            text = tag.get_text()
            if len(text) > 100 and ("def " in text or "import " in text
                                    or "class " in text):
                return text

    return None


def _search_json_for_code(data) -> Optional[str]:
    """递归搜索 JSON 对象中的策略代码字符串"""
    if isinstance(data, dict):
        for k in ("code", "sourceCode", "algorithmCode", "source"):
            if k in data and isinstance(data[k], str) and len(data[k]) > 100:
                if "def " in data[k] or "import " in data[k]:
                    return data[k]
        for v in data.values():
            r = _search_json_for_code(v)
            if r:
                return r
    elif isinstance(data, list):
        for item in data:
            r = _search_json_for_code(item)
            if r:
                return r
    return None


def _extract_params_joinquant(html: str) -> dict:
    """从聚宽策略详情页提取参数"""
    params = {}
    # 尝试从 JSON 中提取 params
    m = re.search(r'"params"\s*:\s*({[^}]+})', html)
    if m:
        try:
            params = json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    if isinstance(params, dict):
        return {k: v for k, v in params.items()
                if isinstance(v, (int, float, str, bool))}
    return {}


# ============================================================
# 评分更新
# ============================================================

def _update_meta_rating(repo, category: str, name: str, rating: float):
    try:
        safe_name = name.replace("/", "_").replace("\\", "_")
        meta_path = repo.root / category / safe_name / "meta.yaml"
        if meta_path.exists():
            import yaml
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = yaml.safe_load(f) or {}
            meta["external_rating"] = rating
            with open(meta_path, "w", encoding="utf-8") as f:
                yaml.dump(meta, f, allow_unicode=True)
    except Exception as e:
        logger.warning(f"无法写入 meta.yaml: {e}")


# ============================================================
# 本地缓存
# ============================================================

def load_cached_results() -> list[dict]:
    cache_file = REPO_ROOT / ".crawl_cache.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_crawl_results(results: list[dict]):
    cache_file = REPO_ROOT / ".crawl_cache.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)


# ============================================================
# 一键爬取
# ============================================================

def crawl_all(repo: StrategyRepo = None) -> dict:
    """一键从所有 web 来源爬取（当前仅 TradingView）"""
    if repo is None:
        repo = StrategyRepo()
    results = {}
    logger.info("=" * 50)
    logger.info("开始爬取所有 web 来源...")

    logger.info("\n[1/1] TradingView...")
    try:
        n, stats = crawl_tradingview(max_scripts=30, sort_by="rating", repo=repo)
        results["tradingview"] = {"imported": n, "stats": stats}
    except Exception as e:
        logger.error(f"TradingView 爬取失败: {e}")
        results["tradingview"] = {"imported": 0, "error": str(e)}

    total = sum(r.get("imported", 0) for r in results.values())
    logger.info(f"\n全部爬取完成: 共导入 {total} 个策略")
    return results


# ============================================================
# 命令行
# ============================================================
# python strategies_repo/crawler.py

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    print("=" * 60)
    print("策略爬虫")
    print("=" * 60)
    print("[1] TradingView — Pine Script 策略（需代理翻墙）")
    print("[0] 退出")
    print("注: 聚宽已放弃(API不可用)，GitHub/Gitee 策略请用 importer.py")
    choice = input("选择: ").strip()

    if choice == "1":
        top_n = input("获取 Top N 个高分策略 [20]: ").strip()
        top_n = int(top_n) if top_n.isdigit() else 20
        n = crawl_tradingview_top(top_n=top_n, min_rating=0)
        print(f"TradingView 爬取完成: {n} 个策略")
