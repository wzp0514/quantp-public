"""
配置校验器 — 参考 vnpy 的启动检查

启动时一次性检查所有配置项，给出具体修复建议。
不让用户运行时才发现"哎呀忘配密码了"。

用法
--------
>>> from config.validator import validate_config
>>> issues = validate_config()
>>> if issues:
...     for i in issues:
...         print(f"[{i['level']}] {i['message']}")
... else:
...     print("配置完整 [OK]")
"""

import logging
from pathlib import Path
from typing import Optional

import yaml

CONFIG_DIR = Path(__file__).resolve().parent


def validate_config() -> list[dict]:
    """
    检查所有配置。

    返回问题列表，每个问题:
      {level: "error"|"warning"|"info", key: "xxx", message: "xxx", fix: "xxx"}
    """
    issues = []

    # 1. 检查文件存在性
    required_files = ["settings.yaml", "strategies.yaml"]
    for fn in required_files:
        path = CONFIG_DIR / fn
        if not path.exists():
            issues.append({
                "level": "error",
                "key": fn,
                "message": f"配置文件不存在: {fn}",
                "fix": f"创建 {path}",
            })

    # 2. 加载 settings.yaml
    settings = {}
    settings_path = CONFIG_DIR / "settings.yaml"
    if settings_path.exists():
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            issues.append({
                "level": "error",
                "key": "settings.yaml",
                "message": f"YAML 格式错误: {e}",
                "fix": "检查缩进和冒号后的空格",
            })

    # 3. 检查 settings.local.yaml
    local_path = CONFIG_DIR / "settings.local.yaml"
    if not local_path.exists():
        issues.append({
            "level": "warning",
            "key": "settings.local.yaml",
            "message": "本地配置文件不存在",
            "fix": f"cp {CONFIG_DIR/'settings.local.yaml.example'} {local_path}",
        })
    else:
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                local_settings = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            issues.append({
                "level": "error",
                "key": "settings.local.yaml",
                "message": f"YAML 格式错误: {e}",
                "fix": "检查缩进和冒号后的空格",
            })

    # 4. 关键字段检查
    checks = [
        ("risk.max_position_pct", "风控仓位上限未设置", "在 settings.yaml 中填写 risk.max_position_pct"),
        ("risk.max_single_loss_pct", "风控止损线未设置", "在 settings.yaml 中填写 risk.max_single_loss_pct"),
    ]

    for key_path, msg, fix in checks:
        if not _get_nested(settings, key_path):
            issues.append({
                "level": "warning",
                "key": key_path,
                "message": msg,
                "fix": fix,
            })

    # 5. 本地密钥检查
    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            local = yaml.safe_load(f) or {}

        # 检查 LLM API key
        llm_key = local.get("llm", {}).get("api_key", "")
        if llm_key and llm_key != "你的DeepSeek API Key":
            pass  # 已配置
        else:
            issues.append({
                "level": "info",
                "key": "llm.api_key",
                "message": "LLM API Key 未配置（LLM因子挖掘需要）",
                "fix": "在 settings.local.yaml 的 llm.api_key 中填入 DeepSeek API Key",
            })

    # 6. 依赖检查
    try:
        import akshare
    except ImportError:
        issues.append({
            "level": "error",
            "key": "dependency",
            "message": "akshare 未安装",
            "fix": "pip install akshare",
        })

    try:
        import backtrader
    except ImportError:
        issues.append({
            "level": "error",
            "key": "dependency",
            "message": "backtrader 未安装",
            "fix": "pip install backtrader",
        })

    return issues


def print_issues(issues: list[dict]):
    """打印检查结果"""
    if not issues:
        print("[OK] 配置完整，可以启动。")
        return

    errors = [i for i in issues if i["level"] == "error"]
    warnings = [i for i in issues if i["level"] == "warning"]
    infos = [i for i in issues if i["level"] == "info"]

    print(f"配置检查: {len(errors)} 个错误, {len(warnings)} 个警告, {len(infos)} 个提示\n")

    for i in errors + warnings + infos:
        emoji = {"error": "[FAIL]", "warning": "[WARN]", "info": "[INFO]"}
        print(f"  {emoji[i['level']]} [{i['key']}] {i['message']}")
        print(f"     → {i['fix']}\n")


def _get_nested(d: dict, key_path: str):
    """获取嵌套字典值，如 'database.host' → d['database']['host']"""
    keys = key_path.split(".")
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k)
        else:
            return None
    return d


# ============================================================
# 命令行
# ============================================================
# python config/validator.py

if __name__ == "__main__":
    print("=" * 50)
    print("配置检查")
    print("=" * 50)
    issues = validate_config()
    print_issues(issues)
