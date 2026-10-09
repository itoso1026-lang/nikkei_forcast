"""確定時刻の付与と、予測対象日カレンダーへの時刻合わせ（リーク防止）。

ルール：予測対象日 t に使える行は available_at_utc < (JST t日 cutoff) のものだけ。
"""
from __future__ import annotations

from datetime import time, timedelta

import numpy as np
import pandas as pd

from .config import JST, load_config


class LeakError(AssertionError):
    pass


def _rule(ticker: str, cfg: dict) -> dict:
    av = cfg["availability"]
    group = av["ticker_group"].get(ticker)
    if group is None:
        raise KeyError(f"availability に {ticker} の定義がありません")
    return av["groups"][group]


def _parse_hm(s: str) -> timedelta:
    h, m = s.split(":")
    return timedelta(hours=int(h), minutes=int(m))


def close_offset(ticker: str, d: pd.Timestamp | None = None, cfg: dict | None = None) -> timedelta:
    """bar日付 D の 00:00（現地）から確定時刻までの時間（day_offset 込み、マージンなし）。"""
    cfg = cfg or load_config()
    r = _rule(ticker, cfg)
    close = r["close"]
    if d is not None:
        for since, c in sorted((r.get("since") or {}).items()):
            if pd.Timestamp(d) >= pd.Timestamp(since):
                close = c
    return _parse_hm(close) + timedelta(days=int(r.get("day_offset", 0)))


def bar_close_offset(ticker: str, cfg: dict | None = None) -> timedelta:
    """1時間足を日足にまとめるときの区切り（bar_close。未設定なら close と同じ）。

    CME の日足の終値はシカゴ時間 15:00 の清算値なので、区切りは 15:00。確定時刻（close）は安全側の 16:00。
    """
    cfg = cfg or load_config()
    r = _rule(ticker, cfg)
    if "bar_close" not in r:
        return close_offset(ticker, None, cfg)
    return _parse_hm(r["bar_close"]) + timedelta(days=int(r.get("day_offset", 0)))


def ticker_tz(ticker: str, cfg: dict | None = None) -> str:
    return _rule(ticker, cfg or load_config())["tz"]


def available_at_utc(dates: pd.DatetimeIndex, ticker: str, cfg: dict | None = None) -> pd.DatetimeIndex:
    """bar日付（tz なしの日付）ごとの確定時刻（UTC、マージン込み）。"""
    cfg = cfg or load_config()
    r = _rule(ticker, cfg)
    dates = pd.DatetimeIndex(dates).normalize()
    offsets = pd.TimedeltaIndex([close_offset(ticker, d, cfg) for d in dates]) if r.get("since") \
        else pd.TimedeltaIndex([close_offset(ticker, None, cfg)] * len(dates))
    local = (dates + offsets).tz_localize(r["tz"], ambiguous=True, nonexistent="shift_forward")
    margin = pd.Timedelta(minutes=cfg["availability"]["margin_min"])
    return (local.tz_convert("UTC") + margin)


def cutoff_utc(target_dates, cfg: dict | None = None) -> pd.DatetimeIndex:
    """予測対象日 t の JST cutoff（初期値 8:00）を UTC で返す。"""
    cfg = cfg or load_config()
    t = pd.DatetimeIndex(pd.to_datetime(target_dates)).normalize()
    local = (t + _parse_hm(cfg["cutoff_jst"])).tz_localize(JST)
    return local.tz_convert("UTC")


def assign_trade_date(bar_start_utc: pd.DatetimeIndex, ticker: str, bar_minutes: int = 60,
                      cfg: dict | None = None, close: timedelta | None = None) -> pd.DatetimeIndex:
    """1時間足を、その足が属する日足の bar 日付に割り当てる。

    確定時刻 c（bar日付 00:00 からの時間）に対し、足の終了時刻 e が (c_{D-1}, c_D] に入る D を返す。
    D = date(e - c + 24h - ε)
    """
    cfg = cfg or load_config()
    tz = ticker_tz(ticker, cfg)
    c = close if close is not None else bar_close_offset(ticker, cfg)
    end_local = (pd.DatetimeIndex(bar_start_utc).tz_convert(tz) + pd.Timedelta(minutes=bar_minutes)).tz_localize(None)
    return (end_local - c + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)).normalize()


def asof_align(source: pd.DataFrame, target_dates, prefix: str, cfg: dict | None = None,
               value_cols: list[str] | None = None) -> pd.DataFrame:
    """source（bar日付 index、available_at_utc 列あり）の各列を、予測対象日 t に揃える。

    t ごとに available_at_utc < cutoff(t) を満たす最新の行を選ぶ。
    戻り値には {prefix}_src_date, {prefix}_src_avail も含める（リーク検査用）。
    """
    cfg = cfg or load_config()
    t = pd.DatetimeIndex(pd.to_datetime(target_dates)).normalize()
    cols = value_cols or [c for c in source.columns if c != "available_at_utc"]
    src = source[cols].copy()
    src[f"{prefix}_src_date"] = source.index
    src["_avail"] = pd.DatetimeIndex(source["available_at_utc"]).tz_convert("UTC")
    src = src.dropna(subset=["_avail"]).sort_values("_avail")
    left = pd.DataFrame({"_t": t, "_cut": cutoff_utc(t, cfg)}).sort_values("_cut")
    merged = pd.merge_asof(left, src, left_on="_cut", right_on="_avail",
                           direction="backward", allow_exact_matches=False)
    merged = merged.set_index("_t").reindex(t)
    merged.index.name = "date"
    merged = merged.rename(columns={c: f"{prefix}_{c}" for c in cols})
    merged[f"{prefix}_src_avail"] = merged.pop("_avail")
    return merged.drop(columns=["_cut"])


def assert_no_leak(df: pd.DataFrame, cfg: dict | None = None, same_day_ok: tuple[str, ...] = ()) -> None:
    """全ての *_src_avail < cutoff(t)、*_src_date < t を検査する。"""
    cfg = cfg or load_config()
    t = pd.DatetimeIndex(df.index)
    cut = cutoff_utc(t, cfg)
    for c in df.columns:
        if c.endswith("_src_avail"):
            av = pd.DatetimeIndex(df[c])
            ok = av.isna() | (av < cut)
            if not np.all(ok):
                bad = df.index[~np.asarray(ok)][:5]
                raise LeakError(f"{c}: 確定時刻が cutoff 以降の行があります: {list(bad)}")
        if c.endswith("_src_date") and c[: -len("_src_date")] not in same_day_ok:
            sd = pd.DatetimeIndex(df[c])
            ok = sd.isna() | (sd < t)
            if not np.all(ok):
                bad = df.index[~np.asarray(ok)][:5]
                raise LeakError(f"{c}: 特徴量の日付が目的変数の日付以降の行があります: {list(bad)}")
