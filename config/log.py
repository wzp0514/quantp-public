"""
共享日志配置 — 全局只配一次 root logger，各模块直接拿子 logger。

用法
--------
>>> from config.log import get_logger
>>> logger = get_logger(__name__)
>>> logger.info("hello")
"""

import logging
import sys

_initialized = False


def _init_root() -> None:
    global _initialized
    if _initialized:
        return
    _initialized = True
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S"
    ))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console)


def get_logger(name: str) -> logging.Logger:
    """返回 logger，首次调用时自动配置 root handler。"""
    _init_root()
    return logging.getLogger(name)
