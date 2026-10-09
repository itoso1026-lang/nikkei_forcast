"""直近のデータの補い方（merge_hourly・session_close_from_quote）のテスト。

2026-10-09 の朝に実際に起きたこと：
- yfinance の NIY=F の「10/8 の日足」が、次の取引（JST 7:00〜）の値で上書きされ続けた
- ^N225 の 10/8 の日足が遅れ、1時間足の最後の値（引けのオークションを含まない）と正式な終値がずれた
"""
import numpy as np
import pandas as pd

from conftest import make_daily
from src.fetch_data import merge_hourly, session_close_from_quote


def _cme_hourly(start_utc: str, end_utc: str) -> pd.DataFrame:
    idx = pd.date_range(start_utc, end_utc, freq="1h", tz="UTC")
    idx = idx[idx.hour != 21]                      # 16:00〜17:00 CDT はメンテナンスで足がない
    close = 68000.0 + np.arange(len(idx))
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=idx)


def test_cme_last_bars_rebuilt_from_hourly(cfg):
    now = pd.Timestamp("2026-10-09 02:00", tz="UTC")             # JST 11:00
    daily = make_daily("NIY=F", ["2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08"],
                       [1.0, 2.0, 3.0, 99999.0], cfg)           # 10/8 の足は次の取引の値で上書きされている
    daily["synth"], daily["synth_src"] = False, ""
    hourly = _cme_hourly("2026-10-04 22:00", "2026-10-09 01:00")   # 日曜 17:00 CDT から
    out = merge_hourly(daily, hourly, "NIY=F", cfg, now)
    # 10/8 の日足の終値は、15:00 CDT に終わる足（19:00 UTC 開始）。日足の清算値の時刻に合わせる
    assert out.loc["2026-10-08", "close"] == hourly.loc["2026-10-08 19:00", "close"]
    assert out.loc["2026-10-08", "synth_src"] == "hourly"
    # 10/9 の取引（JST 7:00〜）はまだ終わっていないので作らない
    assert pd.Timestamp("2026-10-09") not in out.index
    # 1時間足の最初の取引日（10/5）は途中からしかないので、日足のまま
    assert out.loc["2026-10-05", "close"] == 1.0
    assert (out["close"] != 99999.0).all()


def test_tse_synth_uses_official_close(cfg):
    now = pd.Timestamp("2026-10-09 02:30", tz="UTC")
    daily = make_daily("^N225", ["2026-10-06", "2026-10-07"], [70683.98, 70035.71], cfg)
    daily["synth"], daily["synth_src"] = False, ""
    idx = pd.DatetimeIndex([f"2026-10-0{d} {h:02d}:00" for d in (7, 8) for h in range(0, 7)], tz="UTC")
    hourly = pd.DataFrame({"open": 69300.0, "high": 69500.0, "low": 69100.0, "close": 69221.06, "volume": 1.0}, index=idx)
    out = merge_hourly(daily, hourly, "^N225", cfg, now, quote=(pd.Timestamp("2026-10-08"), 69042.11))
    assert out.loc["2026-10-08", "close"] == 69042.11
    assert out.loc["2026-10-08", "low"] == 69042.11
    assert out.loc["2026-10-08", "synth_src"] == "hourly+quote"
    # 気配情報が取れないときは 1時間足の値のまま（予測時に警告を出す）
    out2 = merge_hourly(daily, hourly, "^N225", cfg, now, quote=None)
    assert out2.loc["2026-10-08", "synth_src"] == "hourly"


def _ts(s):
    return int(pd.Timestamp(s, tz="Asia/Tokyo").timestamp())


def test_session_close_from_quote(cfg):
    days = pd.bdate_range("2026-09-01", "2026-10-08")
    # 取引中（前場）：前の取引日の終値
    q = {"regularMarketTime": _ts("2026-10-09 11:16"), "marketState": "REGULAR",
         "regularMarketPrice": 68457.56, "regularMarketPreviousClose": 69042.11}
    assert session_close_from_quote(q, days, cfg) == (pd.Timestamp("2026-10-08"), 69042.11)
    # 寄りの前（朝 7:50）：最後の約定は前日の引け → その値が前日の終値
    q = {"regularMarketTime": _ts("2026-10-08 15:30"), "marketState": "CLOSED",
         "regularMarketPrice": 69042.11, "regularMarketPreviousClose": 70035.71}
    assert session_close_from_quote(q, days, cfg) == (pd.Timestamp("2026-10-08"), 69042.11)
    # 月曜の前場：前の取引日は金曜
    q = {"regularMarketTime": _ts("2026-10-05 10:00"), "marketState": "REGULAR",
         "regularMarketPrice": 1.0, "regularMarketPreviousClose": 2.0}
    assert session_close_from_quote(q, days, cfg) == (pd.Timestamp("2026-10-02"), 2.0)
    assert session_close_from_quote({}, days, cfg) is None
