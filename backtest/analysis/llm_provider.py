"""
LLM Provider 统一接口 — 支持 Anthropic / OpenAI / DeepSeek。

自动检测环境变量中的 API Key，兼容 settings.yaml 的 llm 段配置。

用法
--------
>>> from backtest.analysis.llm_provider import call_llm
>>> resp = call_llm(provider="deepseek", system="你是分析师", prompt="分析这只股票")
>>> print(resp)
"""

import json
import logging
import os
import urllib.request
from typing import Optional

from config.log import get_logger

logger = get_logger("llm_provider")

# 已安装的 SDK 缓存
_available_providers: Optional[dict] = None


def _detect_providers() -> dict:
    """检测哪些 LLM SDK 可用，加上 API Key 是否存在"""
    global _available_providers
    if _available_providers is not None:
        return _available_providers

    providers = {}

    # Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from anthropic import Anthropic  # noqa: F401
            providers["anthropic"] = {"sdk": True, "key": True}
        except ImportError:
            providers["anthropic"] = {"sdk": False, "key": True}
    else:
        providers["anthropic"] = {"sdk": True, "key": False}

    # OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI  # noqa: F401
            providers["openai"] = {"sdk": True, "key": True}
        except ImportError:
            providers["openai"] = {"sdk": False, "key": True}
    else:
        providers["openai"] = {"sdk": True, "key": False}

    # DeepSeek (uses urllib, no SDK needed)
    try:
        from config.loader import get_llm_config
        cfg = get_llm_config()
        key = cfg.get("api_key", "")
        if key and key != "你的DeepSeek API Key":
            providers["deepseek"] = {"sdk": True, "key": True}
        else:
            providers["deepseek"] = {"sdk": True, "key": False}
    except Exception:
        providers["deepseek"] = {"sdk": True, "key": False}

    _available_providers = providers
    return providers


def _get_default_config() -> dict:
    """从 settings.yaml 读取 LLM 默认配置"""
    try:
        from config.loader import get_llm_config
        return get_llm_config()
    except Exception:
        return {}


def call_llm(
    system_prompt: str,
    user_prompt: str,
    provider: str = "",
    model: str = "",
    max_tokens: int = 1000,
    temperature: float = 0.3,
    response_format: Optional[dict] = None,
) -> str:
    """
    统一的 LLM 调用接口。

    参数
    ----------
    system_prompt : str
        系统提示词
    user_prompt : str
        用户提示词
    provider : str
        anthropic / openai / deepseek，为空自动选择
    model : str
        模型名，为空用默认
    max_tokens : int
        最大输出 token 数
    temperature : float
        采样温度（决策场景建议 0.3）
    response_format : dict
        JSON schema 等格式约束（部分 provider 支持）

    返回
    -------
    str: 模型输出文本，失败返回空字符串
    """
    providers = _detect_providers()
    cfg = _get_default_config()

    # 自动选择 provider
    if not provider:
        provider = cfg.get("provider", "")
    if not provider:
        # 按优先级自动选
        for p in ["deepseek", "anthropic", "openai"]:
            if providers.get(p, {}).get("key"):
                provider = p
                break
    if not provider:
        logger.warning("无可用 LLM provider（所有 API Key 均未配置）")
        return ""

    if not model:
        model = cfg.get("model", "")
    if not model:
        defaults = {"deepseek": "deepseek-chat", "anthropic": "claude-haiku-4-5-20251001", "openai": "gpt-4o-mini"}
        model = defaults.get(provider, "gpt-4o-mini")

    if not max_tokens:
        max_tokens = cfg.get("max_tokens", 1000)
    if not temperature and temperature != 0:
        temperature = cfg.get("temperature", 0.3)

    logger.debug(f"LLM调用: provider={provider}, model={model}")

    try:
        if provider == "deepseek":
            return _call_deepseek(system_prompt, user_prompt, model, max_tokens, temperature)
        elif provider == "anthropic":
            return _call_anthropic(system_prompt, user_prompt, model, max_tokens, temperature)
        elif provider == "openai":
            return _call_openai(system_prompt, user_prompt, model, max_tokens, temperature, response_format)
        else:
            logger.warning(f"未知 LLM provider: {provider}")
            return ""
    except Exception as e:
        logger.warning(f"LLM调用失败 ({provider}/{model}): {e}")
        return ""


def _call_deepseek(system_prompt: str, user_prompt: str,
                   model: str, max_tokens: int, temperature: float) -> str:
    cfg = _get_default_config()
    api_key = cfg.get("api_key", "")
    url = "https://api.deepseek.com/v1/chat/completions"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    data = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        msg = result["choices"][0]["message"]
        return msg.get("content", "") or ""


def _call_anthropic(system_prompt: str, user_prompt: str,
                    model: str, max_tokens: int, temperature: float) -> str:
    from anthropic import Anthropic
    api_key = os.environ["ANTHROPIC_API_KEY"]
    client = Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return msg.content[0].text


def _call_openai(system_prompt: str, user_prompt: str,
                 model: str, max_tokens: int, temperature: float,
                 response_format: Optional[dict] = None) -> str:
    from openai import OpenAI
    api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if response_format:
        kwargs["response_format"] = response_format
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""
