"""データ取得元の抽象化。

DataSource.fetch() は bar日付（tz なし）を index に持ち、
open, high, low, close, volume, available_at_utc（, contract_month）列の DataFrame を返す。
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from .align import available_at_utc
from .config import JST, load_config

OHLCV = ["open", "high", "low", "close", "volume"]


class DataSource(ABC):
    @abstractmethod
    def fetch(self, symbol: str, start: str | pd.Timestamp, end: str | pd.Timestamp | None = None) -> pd.DataFrame:
        ...


def _normalize_daily(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=str.lower)
    if "close" not in df.columns:
        return pd.DataFrame(columns=OHLCV)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    df = df[OHLCV].astype(float)
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(subset=["close"])


class YFinanceSource(DataSource):
    def __init__(self, cfg: dict | None = None, retries: int = 4, pause: float = 3.0):
        self.cfg = cfg or load_config()
        self.retries = retries
        self.pause = pause

    def _history(self, symbol: str, **kw) -> pd.DataFrame:
        import yfinance as yf

        last_err = None
        for i in range(self.retries):
            try:
                df = yf.Ticker(symbol).history(**kw)
                if df is not None and len(df):
                    return df
                last_err = RuntimeError("empty")
            except Exception as e:  # yfinance はネットワーク・レート制限で例外を投げる
                last_err = e
            time.sleep(self.pause * (i + 1))
        raise RuntimeError(f"{symbol} の取得に失敗しました: {last_err}")

    def fetch(self, symbol, start, end=None) -> pd.DataFrame:
        adj = symbol in self.cfg.get("auto_adjust_tickers", [])
        raw = self._history(symbol, start=pd.Timestamp(start).strftime("%Y-%m-%d"),
                            end=None if end is None else pd.Timestamp(end).strftime("%Y-%m-%d"),
                            interval="1d", auto_adjust=adj)
        df = _normalize_daily(raw)
        df["available_at_utc"] = available_at_utc(df.index, symbol, self.cfg)
        return df

    def fetch_hourly(self, symbol: str, period: str = "7d") -> pd.DataFrame:
        """1時間足。index は足の開始時刻（UTC）。"""
        adj = symbol in self.cfg.get("auto_adjust_tickers", [])
        raw = self._history(symbol, period=period, interval="1h", auto_adjust=adj)
        df = raw.rename(columns=str.lower)[OHLCV].astype(float)
        df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
        df.index.name = "bar_start_utc"
        return df.dropna(subset=["close"])


class OSENightCSVSource(DataSource):
    """OSE 夜間先物の CSV（未入手。抽象化と読み込みのみ）。

    想定列：datetime（JST）, open, high, low, close, volume, contract_month
    夜間セッション（〜JST 6:00 終了）ごとに日足へ集計する。bar日付は「セッションが終わる JST の日付」。
    この bar 日付は予測対象日 t と同じ日になるため、align.assert_no_leak の same_day_ok に指定して使う。
    """

    def __init__(self, csv_path: str | Path, session_end: str = "06:00", bar_minutes: int = 1):
        self.csv_path = Path(csv_path)
        h, m = session_end.split(":")
        self.session_end = pd.Timedelta(hours=int(h), minutes=int(m))
        self.bar_minutes = bar_minutes

    def fetch(self, symbol="OSE_NK225_NIGHT", start="2000-01-01", end=None) -> pd.DataFrame:
        raw = pd.read_csv(self.csv_path, parse_dates=["datetime"])
        raw["datetime"] = pd.DatetimeIndex(raw["datetime"]).tz_localize(JST) \
            if pd.DatetimeIndex(raw["datetime"]).tz is None else raw["datetime"].dt.tz_convert(JST)
        local = raw["datetime"].dt.tz_localize(None)
        # 夜間セッション：前日 16:30 頃〜当日 session_end。日中セッションの足は除外する。
        tod = local - local.dt.normalize()
        night = (tod <= self.session_end) | (tod >= pd.Timedelta(hours=16, minutes=30))
        raw = raw[night].copy()
        local = local[night]
        raw["session"] = (local - self.session_end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)).dt.normalize()
        # セッションごとに出来高が最大の限月（期近）を採用する
        vol = raw.groupby(["session", "contract_month"])["volume"].sum().reset_index()
        front = vol.sort_values(["session", "volume"]).groupby("session").tail(1)[["session", "contract_month"]]
        raw = raw.merge(front, on=["session", "contract_month"])
        raw = raw.sort_values("datetime")
        g = raw.groupby("session")
        df = pd.DataFrame({
            "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
            "close": g["close"].last(), "volume": g["volume"].sum(),
            "contract_month": g["contract_month"].last(),
            "available_at_utc": (g["datetime"].max() + pd.Timedelta(minutes=self.bar_minutes)).dt.tz_convert("UTC"),
        })
        df.index = pd.DatetimeIndex(df.index)
        df.index.name = "date"
        df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]
        return df
