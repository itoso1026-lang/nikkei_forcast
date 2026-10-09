import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.align import available_at_utc  # noqa: E402
from src.config import load_config  # noqa: E402


@pytest.fixture
def cfg():
    return load_config()


def make_daily(ticker, dates, close, cfg, **extra):
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=idx)
    for k, v in extra.items():
        df[k] = v
    df["available_at_utc"] = available_at_utc(df.index, ticker, cfg)
    df.index.name = "date"
    return df


def real_cache_available() -> bool:
    return (ROOT / "data" / "raw" / "IDX_N225.parquet").exists()
