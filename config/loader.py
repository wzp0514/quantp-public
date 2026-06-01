"""
统一配置加载器 — R4: 密钥注入三通道

项目中唯一读取配置的地方。其他模块通过 get_config() 获取。

加载优先级（R4: 三通道密钥注入）：
  1. 环境变量（最高优先级，QUANTP_ 前缀） — 适用于 CI/CD / Docker
  2. settings.local.yaml（本地密钥） — 适用于本地开发
  3. settings.yaml（默认值） — 适用于公开仓库

用法
--------
>>> from config.loader import get_config
>>> cfg = get_config()
>>> print(cfg["database"]["host"])
"""

import os
from pathlib import Path
from typing import Optional
import yaml

# 配置根目录（相对于项目根）
_CONFIG_DIR = Path(__file__).resolve().parent

# 缓存（单例，读一次）
_config_cache: Optional[dict] = None

# ============================================================
# R4: 环境变量 → 配置项映射
# ============================================================
# 优先级: 环境变量 > settings.local.yaml > settings.yaml
# 支持的环境变量（QUANTP_ 前缀）：
#   QUANTP_TUSHARE_TOKEN       → data_sources.tushare.token
#   QUANTP_LLM_API_KEY         → llm.api_key
#   QUANTP_TELEGRAM_TOKEN      → notification.telegram_bot_token
#   QUANTP_WECOM_WEBHOOK       → notification.wecom_webhook
#   QUANTP_DINGTALK_WEBHOOK    → notification.dingtalk_webhook
#   QUANTP_FEISHU_WEBHOOK      → notification.feishu_webhook
#   QUANTP_DEEPSEEK_KEY        → llm.api_key (同 LLM)
#   QUANTP_CCXT_API_KEY        → ccxt.api_key
#   QUANTP_CCXT_SECRET         → ccxt.secret
# ============================================================

_ENV_MAP = {
    "QUANTP_TUSHARE_TOKEN":    ("data_sources", "tushare", "token"),
    "QUANTP_LLM_API_KEY":      ("llm", "api_key"),
    "QUANTP_DEEPSEEK_KEY":     ("llm", "api_key"),
    "QUANTP_TELEGRAM_TOKEN":   ("notification", "telegram_bot_token"),
    "QUANTP_WECOM_WEBHOOK":    ("notification", "wecom_webhook"),
    "QUANTP_DINGTALK_WEBHOOK": ("notification", "dingtalk_webhook"),
    "QUANTP_FEISHU_WEBHOOK":   ("notification", "feishu_webhook"),
    "QUANTP_CCXT_API_KEY":     ("ccxt", "api_key"),
    "QUANTP_CCXT_SECRET":      ("ccxt", "secret"),
}


def _inject_env_vars(config: dict) -> dict:
    """R4: 从环境变量注入敏感配置（最高优先级）"""
    for env_var, path in _ENV_MAP.items():
        val = os.environ.get(env_var, "")
        if val:
            section = path[0]
            sub_key = path[1]
            if section not in config:
                config[section] = {}
            # 确保嵌套 dict 存在
            if not isinstance(config[section], dict):
                config[section] = {}
            if sub_key not in config[section]:
                config[section][sub_key] = {}
            # 对于有 3 级路径的情况(path[2]存在)，放入嵌套dict
            if len(path) >= 3:
                config[section][sub_key][path[2]] = val
            else:
                config[section][sub_key] = val
    return config


def get_config() -> dict:
    """获取完整配置（环境变量 > settings.local.yaml > settings.yaml）"""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    config = {}

    # 1. 公开配置（最低优先级）
    public_path = _CONFIG_DIR / "settings.yaml"
    if public_path.exists():
        with open(public_path, "r", encoding="utf-8") as f:
            config.update(yaml.safe_load(f) or {})

    # 2. 本地配置覆盖（中等优先级）
    local_path = _CONFIG_DIR / "settings.local.yaml"
    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            _deep_merge(config, yaml.safe_load(f) or {})

    # 3. R4: 环境变量覆盖（最高优先级，适用于 CI/CD / Docker）
    config = _inject_env_vars(config)

    _config_cache = config
    return config


def get_risk_config() -> dict:
    """获取风控配置"""
    return get_config().get("risk", {})


def get_llm_config() -> dict:
    """获取 LLM 配置"""
    return get_config().get("llm", {})


def get_tushare_token() -> str:
    """获取 Tushare Token"""
    return get_config().get("data_sources", {}).get("tushare", {}).get("token", "")


def get_notification_config() -> dict:
    """获取通知配置"""
    return get_config().get("notification", {})


def get_a_share_config() -> dict:
    """获取A股数据源配置（含降级链、质量检查、缓存策略）"""
    return get_config().get("data_sources", {}).get("a_share", {})


def get_fallback_chain() -> list[str]:
    """获取A股降级链顺序，如 ['akshare', 'tushare', 'baostock']"""
    cfg = get_a_share_config()
    chain = cfg.get("fallback_chain", ["akshare", "tushare", "baostock"])
    if isinstance(chain, list) and len(chain) > 0:
        return chain
    return ["akshare", "tushare", "baostock"]


def is_quality_check_enabled() -> bool:
    """是否启用拉取后自动质量检查"""
    return get_a_share_config().get("quality_check", True)


def get_cache_days() -> int:
    """数据缓存天数（同一天内不重复拉取）"""
    return get_a_share_config().get("cache_days", 1)


def is_crypto_enabled() -> bool:
    """加密市场是否启用（默认关闭）"""
    return get_config().get("markets", {}).get("crypto", False)


def get_paper_trading_config() -> dict:
    """获取纸上交易配置"""
    return get_config().get("paper_trading", {})


def get_kronos_config() -> dict:
    """获取 Kronos 模型配置"""
    defaults = {
        "model": "small",
        "device": "auto",
        "max_context": 512,
        "cache_ttl": 3600,
        "source_enabled": False,
        "factor_registration": False,
        "l4_agent": False,
        "risk_guard": False,
        "webui_enabled": False,
        "token_analysis": False,
        "pred_len": 20,
        "T": 1.0,
        "top_p": 0.9,
        "top_k": 0,
        "sample_count": 5,
        "lookback": 60,
        "ic_threshold": 0.03,
        "tokenizer_path": "",
        "model_path": "",
        "project_path": "",
    }
    cfg = get_config().get("kronos", {})
    defaults.update(cfg)
    return defaults


def get_alternative_data_config() -> dict:
    """获取另类数据源权重配置"""
    defaults = {
        "news_sentiment": 0.25,
        "geopolitical": 0.25,
        "xueqiu_sentiment": 0.25,
        "earnings_quality": 0.25,
        "confidence_threshold": 0.3,
    }
    cfg = get_config().get("alternative_data", {})
    defaults.update(cfg)
    return defaults


def get_encoding_config() -> dict:
    """获取编码配置"""
    defaults = {"default": "utf-8", "fallback": "gbk", "error_mode": "replace"}
    cfg = get_config().get("encoding", {})
    defaults.update(cfg)
    return defaults


def get_default_encoding() -> str:
    """获取系统默认编码"""
    return get_encoding_config()["default"]


def detect_and_decode(data: bytes, encodings: list = None) -> str:
    """尝试多种编码解码字节数据，返回第一个成功的结果。

    Args:
        data: 原始字节
        encodings: 尝试的编码列表，默认从配置获取

    Returns:
        解码后的字符串
    """
    if encodings is None:
        cfg = get_encoding_config()
        encodings = [cfg["default"], cfg["fallback"], "utf-8", "gbk", "gb2312", "latin-1"]
    for enc in encodings:
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, UnicodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def reload_config() -> dict:
    """强制重新加载配置（修改配置文件后调用）"""
    global _config_cache
    _config_cache = None
    return get_config()


def _deep_merge(base: dict, override: dict) -> None:
    """深度合并，override 覆盖 base 同名键"""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


# ============================================================
# 命令行测试
# ============================================================
# python config/loader.py

if __name__ == "__main__":
    cfg = get_config()
    print("风控:", get_risk_config())
    print("LLM:", "已配置" if get_llm_config().get("api_key") else "未配置")
