"""
pytest 配置文件（T1）

运行:
  pytest tests/ -v              # 全部测试
  pytest tests/ -x -m "not slow"   # 跳过慢测试
  pytest tests/ --cov=backtest --cov=live --cov-report=html  # 覆盖率
"""

import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_or_fetch_parquet(name: str, index_cn: str, start: str, end: str):
    """加载缓存Parquet，若无则拉取并保存。返回DataFrame。"""
    import pandas as pd

    fpath = FIXTURES_DIR / name
    if fpath.exists():
        df = pd.read_parquet(fpath)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    from data.fetchers.fallback import fetch_index_daily_safe
    df = fetch_index_daily_safe(index_cn, start, end)
    if not df.empty:
        df.to_parquet(fpath, index=False)
    return df


@pytest.fixture(scope="session")
def csi300_2023():
    """沪深300 2023全年日线（真实数据，AkShare→Parquet缓存）"""
    return _load_or_fetch_parquet("csi300_2023.parquet", "沪深300", "20230101", "20231231")


@pytest.fixture(scope="session")
def csi300_2015_2023():
    """沪深300 2015-2023 多年日线（用于 WFA/auto_recover 等需长数据的测试）"""
    return _load_or_fetch_parquet("csi300_2015_2023.parquet", "沪深300", "20150101", "20231231")


@pytest.fixture(scope="session")
def csi500_2023():
    """中证500 2023全年日线（真实数据，AkShare→Parquet缓存）"""
    return _load_or_fetch_parquet("csi500_2023.parquet", "中证500", "20230101", "20231231")


@pytest.fixture(scope="session")
def sample_ohlcv_df():
    """生成标准OHLCV测试数据（250个交易日）"""
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    n = 250
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    ret = np.random.randn(n) * 0.012 + 0.0003
    price = 3500 * np.cumprod(1 + ret)

    return pd.DataFrame({
        "date": dates,
        "open": price * 0.998,
        "high": price * 1.012,
        "low": price * 0.988,
        "close": price,
        "volume": np.random.randint(100, 500, n).astype(float) * 1e6,
    })


@pytest.fixture(scope="session")
def sample_multistock_panel():
    """生成多股票面板测试数据（10只×252日）"""
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    n_stocks = 10
    n_days = 252
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")

    panel = {}
    for i in range(n_stocks):
        ret = np.random.randn(n_days) * 0.015 + 0.0002
        price = 20 * np.cumprod(1 + ret)
        panel[f"STOCK_{i:03d}"] = pd.DataFrame({
            "date": dates,
            "open": price * 0.998,
            "high": price * 1.015,
            "low": price * 0.985,
            "close": price,
            "volume": np.random.randint(1000, 50000, n_days).astype(float) * 100,
        })
    return panel


@pytest.fixture
def sample_strategy_class():
    """返回双均线交叉策略类（用于大多数测试）"""
    from backtest.strategies.builtin.ma_cross import MaCrossStrategy
    return MaCrossStrategy
