"""
通知模块 — 实盘异常告警

支持六种通知渠道（全部免费）：
  1. 钉钉机器人 — 群 webhook，免费不限量，国内首选
  2. 飞书机器人 — 群 webhook，免费不限量
  3. PushPlus — 微信推送，免费 200条/天
  4. Server酱 — 微信推送，免费
  5. 企业微信机器人 — 群 webhook，免费
  6. Telegram Bot — 免费，需翻墙

短信需要付费（阿里云约 ¥0.045/条），不作为免费渠道内置。
如需短信，可在告警模板中对接阿里云/腾讯云 SMS SDK。

配置放在 config/settings.local.yaml：
  notification:
    dingtalk_webhook: "https://oapi.dingtalk.com/robot/send?access_token=xxx"
    feishu_webhook: "https://open.feishu.cn/open-apis/bot/v2/hook/xxx"
    pushplus_token: "xxx"
    server_chan_sendkey: "xxx"
    wecom_webhook: "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx"
    telegram_bot_token: "xxx"
    telegram_chat_id: "xxx"

钉钉机器人开通（1 分钟）：
  1. 钉钉PC端 → 群设置 → 智能群助手 → 添加机器人 → 自定义
  2. 机器人名字随便填，复制 webhook URL
  3. 安全设置选「自定义关键词」填 "QuantP"
  4. 填入配置文件即可

飞书机器人开通（1 分钟）：
  1. 飞书PC端 → 群设置 → 群机器人 → 添加机器人 → 自定义机器人
  2. 复制 webhook URL
  3. 安全设置选「自定义关键词」填 "QuantP"
  4. 填入配置文件即可

PushPlus 开通（1 分钟）：
  1. 微信扫码登录 https://www.pushplus.plus/
  2. 发送消息 → 一键复制 Token
  3. 填入配置文件即可（免费 200条/天）

Server酱 开通（1 分钟）：
  1. 微信扫码登录 https://sct.ftqq.com/
  2. 获取 SendKey
  3. 填入配置文件即可
"""

import logging
import urllib.request
import urllib.error
import urllib.parse
import json
from datetime import datetime
from typing import Optional

from config.loader import get_notification_config

from config.log import get_logger
logger = get_logger("notifier")
# ============================================================
# 配置读取
# ============================================================

def _load_notify_config() -> dict:
    """读取通知配置（委托给 config.loader）。"""
    return get_notification_config()


# ============================================================
# Telegram
# ============================================================

def send_telegram(message: str) -> bool:
    """
    发送 Telegram 消息。

    参数
    ----------
    message : str
        消息内容（支持 Markdown 格式）

    返回
    -------
    bool : 是否发送成功
    """
    cfg = _load_notify_config()
    token = cfg.get("telegram_bot_token", "")
    chat_id = cfg.get("telegram_chat_id", "")

    if not token or not chat_id:
        logger.warning("Telegram 配置未完成，请在 settings.local.yaml 中填写 token 和 chat_id")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if result.get("ok"):
                logger.info("Telegram 消息已发送")
                return True
            else:
                logger.error(f"Telegram 发送失败: {result}")
                return False
    except Exception as e:
        logger.error(f"Telegram 发送异常: {e}")
        return False


# ============================================================
# 钉钉机器人（群 webhook，免费，国内首选）
# ============================================================

def send_dingtalk(message: str, title: str = "QuantP") -> bool:
    """
    发送钉钉群机器人消息。

    完全免费，不限量，国内首选。支持 Markdown 格式。

    开通: 钉钉群 → 群设置 → 智能群助手 → 添加机器人 → 自定义
    安全设置建议选「自定义关键词」，填 "QuantP"。

    参数
    ----------
    message : str
        消息内容（支持 Markdown）
    title : str
        消息标题

    返回
    -------
    bool
    """
    cfg = _load_notify_config()
    webhook = cfg.get("dingtalk_webhook", "")

    if not webhook:
        logger.debug("钉钉 webhook 未配置")
        return False

    data = {
        "msgtype": "markdown",
        "markdown": {
            "title": title,
            "text": message,
        },
    }

    try:
        req = urllib.request.Request(
            webhook,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if result.get("errcode") == 0:
                logger.info("钉钉消息已发送")
                return True
            else:
                logger.error(f"钉钉发送失败: {result}")
                return False
    except Exception as e:
        logger.error(f"钉钉发送异常: {e}")
        return False


# ============================================================
# 飞书机器人（群 webhook，免费）
# ============================================================

def send_feishu(message: str, title: str = "QuantP") -> bool:
    """
    发送飞书群机器人消息。

    完全免费，不限量。支持富文本格式。

    开通: 飞书群 → 群设置 → 群机器人 → 添加机器人 → 自定义机器人
    安全设置建议选「自定义关键词」，填 "QuantP"。

    参数
    ----------
    message : str
        消息内容（纯文本，飞书不支持 Markdown 标题）
    title : str
        消息标题

    返回
    -------
    bool
    """
    cfg = _load_notify_config()
    webhook = cfg.get("feishu_webhook", "")

    if not webhook:
        logger.debug("飞书 webhook 未配置")
        return False

    # 飞书富文本格式，把 title 加粗放第一行
    content = [[{"tag": "text", "text": f"【{title}】\n{message}"}]]

    data = {
        "msg_type": "post",
        "content": {
            "post": {
                "zh_cn": {
                    "title": title,
                    "content": content,
                }
            }
        },
    }

    try:
        req = urllib.request.Request(
            webhook,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if result.get("code") == 0:
                logger.info("飞书消息已发送")
                return True
            else:
                logger.error(f"飞书发送失败: {result}")
                return False
    except Exception as e:
        logger.error(f"飞书发送异常: {e}")
        return False


# ============================================================
# 企业微信
# ============================================================

def send_wecom(message: str) -> bool:
    """
    发送企业微信机器人消息。

    参数
    ----------
    message : str
        消息内容（纯文本，不支持 Markdown）

    返回
    -------
    bool
    """
    cfg = _load_notify_config()
    webhook = cfg.get("wecom_webhook", "")

    if not webhook:
        logger.warning("企业微信 webhook 未配置，请在 settings.local.yaml 中填写")
        return False

    data = {
        "msgtype": "text",
        "text": {"content": message},
    }

    try:
        req = urllib.request.Request(
            webhook,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if result.get("errcode") == 0:
                logger.info("企业微信消息已发送")
                return True
            else:
                logger.error(f"企业微信发送失败: {result}")
                return False
    except Exception as e:
        logger.error(f"企业微信发送异常: {e}")
        return False


# ============================================================
# PushPlus（微信推送，免费，推荐国内使用）
# ============================================================

def send_pushplus(message: str, title: str = "QuantP") -> bool:
    """
    通过 PushPlus 发送微信消息。

    免费额度: 200条/天，无需翻墙，国内首选。

    参数
    ----------
    message : str
        消息内容（支持 Markdown）
    title : str
        消息标题

    返回
    -------
    bool
    """
    cfg = _load_notify_config()
    token = cfg.get("pushplus_token", "")

    if not token:
        logger.debug("PushPlus Token 未配置")
        return False

    url = "http://www.pushplus.plus/send"
    data = {
        "token": token,
        "title": title,
        "content": message,
        "template": "markdown",
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if result.get("code") == 200:
                logger.info("PushPlus 消息已发送")
                return True
            else:
                logger.error(f"PushPlus 发送失败: {result}")
                return False
    except Exception as e:
        logger.error(f"PushPlus 发送异常: {e}")
        return False


# ============================================================
# Server酱（微信推送，免费，备选）
# ============================================================

def send_server_chan(message: str, title: str = "QuantP") -> bool:
    """
    通过 Server酱 发送微信消息。

    参数
    ----------
    message : str
        消息内容（支持 Markdown）
    title : str
        消息标题

    返回
    -------
    bool
    """
    cfg = _load_notify_config()
    sendkey = cfg.get("server_chan_sendkey", "")

    if not sendkey:
        logger.debug("Server酱 SendKey 未配置")
        return False

    url = f"https://sctapi.ftqq.com/{sendkey}.send"
    data = {
        "title": title,
        "desp": message,
    }

    try:
        req = urllib.request.Request(
            url,
            data=urllib.parse.urlencode(data).encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if result.get("code") == 0:
                logger.info("Server酱 消息已发送")
                return True
            else:
                logger.error(f"Server酱 发送失败: {result}")
                return False
    except Exception as e:
        logger.error(f"Server酱 发送异常: {e}")
        return False


# ============================================================
# 告警模板
# ============================================================

def alert_trade(
    symbol: str,
    side: str,
    size: int,
    price: float,
    pnl: float = 0,
) -> str:
    """交易告警模板"""
    sign = "[BUY]" if side == "buy" else "[SELL]"
    return (
        f"{sign} *交易通知*\n"
        f"标的: `{symbol}`\n"
        f"方向: {'买入' if side == 'buy' else '卖出'}\n"
        f"数量: {size}\n"
        f"价格: {price:.4f}\n"
        f"{f'盈亏: {pnl:+.2f}' if side == 'sell' and pnl else ''}\n"
        f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


def alert_risk(violation: str, detail: str) -> str:
    """风控告警模板"""
    return (
        f"[ALERT] *风控告警*\n"
        f"违规项: {violation}\n"
        f"详情: {detail}\n"
        f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"\n[!] 已自动停止交易，请检查。"
    )


def alert_error(error_type: str, message: str, positions: str = "") -> str:
    """异常告警模板（程序崩溃/API中断等）"""
    return (
        f"[ERROR] *异常告警*\n"
        f"类型: {error_type}\n"
        f"详情: {message}\n"
        f"{f'当前持仓: {positions}' if positions else ''}\n"
        f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


def alert_daily_summary(
    total_return: float,
    daily_pnl: float,
    trades: int,
    signals: int,
    warnings: int,
) -> str:
    """每日总结模板"""
    return (
        f"[DAILY] *每日总结*\n"
        f"日期: {datetime.now().strftime('%Y-%m-%d')}\n"
        f"总收益率: {total_return:+.2%}\n"
        f"今日盈亏: {daily_pnl:+.2f}\n"
        f"成交: {trades} 笔\n"
        f"信号: {signals} 次\n"
        f"{f'[!] 风控预警: {warnings} 次' if warnings > 0 else '风控: NORMAL'}"
    )


# ============================================================
# 统一发送接口
# ============================================================

def send(message: str, channel: str = "auto", title: str = "QuantP") -> bool:
    """
    发送通知（自动选择可用渠道）

    参数
    ----------
    message : str
        消息内容
    channel : str
        "dingtalk" / "feishu" / "pushplus" / "serverchan" /
        "telegram" / "wecom" / "auto"
    title : str
        消息标题

    返回
    -------
    bool
    """
    if channel == "dingtalk":
        return send_dingtalk(message, title)
    elif channel == "feishu":
        return send_feishu(message, title)
    elif channel == "pushplus":
        return send_pushplus(message, title)
    elif channel == "serverchan":
        return send_server_chan(message, title)
    elif channel == "telegram":
        return send_telegram(message)
    elif channel == "wecom":
        return send_wecom(message)
    else:
        # auto: 按优先级尝试所有可用渠道
        # 钉钉(国内首选) → 飞书 → PushPlus → Server酱 → Telegram → 企业微信
        if send_dingtalk(message, title):
            return True
        if send_feishu(message, title):
            return True
        if send_pushplus(message, title):
            return True
        if send_server_chan(message, title):
            return True
        if send_telegram(message):
            return True
        return send_wecom(message)


# ============================================================
# 命令行测试
# ============================================================
# python live/gateway/notifier.py

if __name__ == "__main__":
    print("通知模块测试")
    print("=" * 50)

    cfg = _load_notify_config()

    # 全部 6 个渠道
    channels = [
        ("钉钉", cfg.get("dingtalk_webhook")),
        ("飞书", cfg.get("feishu_webhook")),
        ("PushPlus", cfg.get("pushplus_token")),
        ("Server酱", cfg.get("server_chan_sendkey")),
        ("Telegram", cfg.get("telegram_bot_token") and cfg.get("telegram_chat_id")),
        ("企业微信", cfg.get("wecom_webhook")),
    ]

    available = []
    for name, configured in channels:
        status = "已配置" if configured else "未配置"
        print(f"  {name}: {status}")
        if configured:
            available.append(name)

    print(f"\n可用渠道: {', '.join(available) if available else '无（请在 settings.local.yaml 配置）'}")

    if available:
        print("\n发送测试消息...")
        ok = send("[QuantP] 通知测试 — 所有渠道正常", channel="auto", title="QuantP 测试")
        print(f"发送结果: {'成功' if ok else '失败'}")

    # 打印开通指南（只打印未配置的）
    guides = {
        "dingtalk": (
            "\n钉钉机器人开通（1分钟）:",
            "  1. 钉钉群 → 群设置 → 智能群助手 → 添加机器人 → 自定义",
            "  2. 安全设置选「自定义关键词」，填: QuantP",
            "  3. 复制 webhook URL 填入 settings.local.yaml:",
            "     notification:",
            '       dingtalk_webhook: "https://oapi.dingtalk.com/robot/send?access_token=xxx"',
        ),
        "feishu": (
            "\n飞书机器人开通（1分钟）:",
            "  1. 飞书群 → 群设置 → 群机器人 → 添加机器人 → 自定义机器人",
            "  2. 安全设置选「自定义关键词」，填: QuantP",
            "  3. 复制 webhook URL 填入 settings.local.yaml:",
            "     notification:",
            '       feishu_webhook: "https://open.feishu.cn/open-apis/bot/v2/hook/xxx"',
        ),
        "pushplus": (
            "\nPushPlus 开通（1分钟，微信推送）:",
            "  1. 微信扫码登录 https://www.pushplus.plus/",
            "  2. 发送消息 → 一键复制 Token",
            "  3. 填入 settings.local.yaml:",
            "     notification:",
            '       pushplus_token: "你的Token"',
        ),
        "server_chan": (
            "\nServer酱 开通（1分钟，微信推送）:",
            "  1. 微信扫码登录 https://sct.ftqq.com/",
            "  2. 获取 SendKey",
            "  3. 填入 settings.local.yaml:",
            "     notification:",
            '       server_chan_sendkey: "你的SendKey"',
        ),
    }

    for key, lines in guides.items():
        # 检查是否已配置
        if key == "dingtalk" and not cfg.get("dingtalk_webhook"):
            print("\n".join(lines))
        elif key == "feishu" and not cfg.get("feishu_webhook"):
            print("\n".join(lines))
        elif key == "pushplus" and not cfg.get("pushplus_token"):
            print("\n".join(lines))
        elif key == "server_chan" and not cfg.get("server_chan_sendkey"):
            print("\n".join(lines))

    if not cfg.get("telegram_bot_token"):
        print("\nTelegram 开通（需翻墙）:")
        print("  1. 搜索 @BotFather，发送 /newbot")
        print("  2. 拿到 token 和 chat_id 填入 settings.local.yaml")

    if not cfg.get("wecom_webhook"):
        print("\n企业微信机器人开通:")
        print("  1. 企业微信群 → 群设置 → 群机器人 → 添加")
        print("  2. 复制 webhook URL 填入 settings.local.yaml")

    # 模板示例
    print("\n" + "=" * 50)
    print("告警模板示例:")
    print(alert_trade("BTC/USDT", "buy", 100, 65432.10))
    print()
    print(alert_risk("单笔亏损超限", "亏损 2.5% > 上限 2.0%"))
    print()
    print(alert_daily_summary(0.015, -50.0, 3, 5, 0))
