"""
系统配置 — 统一入口：配置加载、校验、日志。

用法
--------
>>> from config import get_config, get_logger, validate_config
>>> cfg = get_config()
>>> logger = get_logger(__name__)
"""

from config.loader import (
    get_config,
    get_risk_config,
    get_llm_config,
    get_tushare_token,
    get_notification_config,
    reload_config,
)
from config.validator import validate_config, print_issues
from config.log import get_logger

__all__ = [
    "get_config",
    "get_risk_config",
    "get_llm_config",
    "get_tushare_token",
    "get_notification_config",
    "reload_config",
    "validate_config",
    "print_issues",
    "get_logger",
]
