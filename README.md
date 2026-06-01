# QuantP

<div align="center">

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**量化研究管线** — 数据获取 → 因子研究 → 策略回测 → 模拟交易

[English](README_EN.md) | [中文](README_zh.md)

</div>

## 这是什么

QuantP 是一套量化交易研究工具，覆盖从数据到模拟交易的全流程。以 Backtrader 为回测引擎，集成了因子研究、策略挖掘、过拟合控制和模拟交易等功能。

详细文档见 [中文文档](README_zh.md) 或 [English Documentation](README_EN.md)。

## 快速开始

```bash
git clone https://github.com/quantp/quantp.git
cd quantp
python -m venv venv
source venv/Scripts/activate  # Windows (Git Bash)
pip install -r requirements.txt
python interactive.py
```

需要 Python 3.12+。更多安装说明和功能细节见 [中文文档](README_zh.md#快速开始)。

## 免责声明

本项目仅供学习研究使用，不构成投资建议。量化交易存在重大亏损风险。详见 [完整免责声明](README_zh.md#免责声明)。

## License

[MIT](LICENSE)
