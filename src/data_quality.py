"""データ品質レポート（欠損・異常値・限月切り替え）。

出力：
- reports/data_quality.csv          ティッカー別の要約
- reports/data_quality_flags.csv    フラグが立った行（異常値・stale など）
- reports/roll_analysis.csv         メジャーSQ前後の先物ベーシスと NIY/NKD の差

usage: python -m src.data_quality
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .align import asof_align
from .calendar_jp import TSECalendar, nyse_holidays
from .check_bar_boundaries import all_tickers
from .config import load_config, path
from .fetch_data import load_raw

US_GROUPS = {"us_stock", "cme"}


def flag_rows(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """行ごとのフラグ（features でも同じ判定を使う）。"""
    dq = cfg["data_quality"]
    r = np.log(df["close"] / df["close"].shift(1))
    sd = r.rolling(250, min_periods=60).std().shift(1)
    out = pd.DataFrame(index=df.index)
    out["logret"] = r
    out["outlier"] = (r.abs() > dq["outlier_sigma"] * sd) | (r.abs() > dq["outlier_abs_logret"])
    out["stale"] = df["close"] == df["close"].shift(1)
    out["zero_volume"] = df["volume"].fillna(0) == 0
    out["flat_ohlc"] = (df["open"] == df["close"]) & (df["high"] == df["low"])
    out["bad_price"] = (df["close"] <= 0) | df["close"].isna()
    return out


def expected_days(ticker: str, cfg: dict, cal: TSECalendar, start, end) -> pd.DatetimeIndex:
    group = cfg["availability"]["ticker_group"][ticker]
    if group.startswith("tse"):
        return cal.days(start, end)
    days = pd.bdate_range(start, end)
    if group in US_GROUPS:
        days = days[[d.date() not in nyse_holidays(d.year) for d in days]]
    return days


def summarize(ticker: str, df: pd.DataFrame, cfg: dict, cal: TSECalendar) -> tuple[dict, pd.DataFrame]:
    f = flag_rows(df, cfg)
    exp = expected_days(ticker, cfg, cal, df.index.min(), df.index.max())
    missing = exp.difference(df.index)
    extra = df.index.difference(exp) if cfg["availability"]["ticker_group"][ticker].startswith("tse") else pd.DatetimeIndex([])
    by_year = pd.Series(1, index=missing).groupby(missing.year).sum().to_dict() if len(missing) else {}
    s = {
        "ticker": ticker, "start": df.index.min().date(), "end": df.index.max().date(), "rows": len(df),
        "missing_days": len(missing), "missing_by_year": by_year,
        "extra_days_on_tse_holidays": len(extra),
        "zero_volume_days": int(f["zero_volume"].sum()),
        "flat_ohlc_days": int(f["flat_ohlc"].sum()),
        "stale_days": int(f["stale"].sum()),
        "outlier_days": int(f["outlier"].sum()),
        "max_abs_logret": round(float(f["logret"].abs().max()), 4),
        "synth_rows": int(df.get("synth", pd.Series(False)).sum()),
    }
    flags = f[f[["outlier", "stale", "flat_ohlc", "bad_price"]].any(axis=1)].copy()
    flags.insert(0, "ticker", ticker)
    flags["close"] = df.loc[flags.index, "close"]
    extra_df = pd.DataFrame({"ticker": ticker, "extra_on_tse_holiday": True}, index=extra)
    return s, pd.concat([flags, extra_df])


def roll_analysis(cfg: dict, cal: TSECalendar) -> pd.DataFrame:
    """メジャーSQ日からの営業日オフセットごとに、T2 型の残差（N225 / 先物）と NIY・NKD の差を平均する。"""
    n225 = load_raw(cfg["tickers"]["n225"], cfg)
    t = cal.history
    rows = []
    for key in ("niy", "nkd"):
        # NKD はドル建てだが価格は指数ポイント（クオント）なので換算しない
        fut = load_raw(cfg["tickers"][key], cfg)
        a = asof_align(fut, t, key, cfg, value_cols=["close"])
        res = np.log(n225["close"].reindex(t) / a[f"{key}_close"])
        res.name = key
        rows.append(res)
    df = pd.concat(rows, axis=1)
    df["niy_minus_nkd"] = df["nkd"] - df["niy"]
    sqs = [cal.sq_date(y, m) for y in range(t.min().year, t.max().year + 1) for m in (3, 6, 9, 12)]
    pos = pd.Series(np.arange(len(t)), index=t)
    out = []
    for sq in sqs:
        if sq not in pos.index:
            continue
        i = pos[sq]
        for k in range(-5, 6):
            j = i + k
            if 0 <= j < len(t):
                d = t[j]
                out.append({"sq": sq.date(), "month": sq.month, "offset_bdays": k,
                            "basis_niy_bp": df.loc[d, "niy"] * 1e4, "basis_nkd_bp": df.loc[d, "nkd"] * 1e4})
    o = pd.DataFrame(out)
    return o.groupby(["offset_bdays"])[["basis_niy_bp", "basis_nkd_bp"]].agg(["mean", "median"]).round(1)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    cfg = load_config()
    n225 = load_raw(cfg["tickers"]["n225"], cfg)
    cal = TSECalendar(n225.index, cfg)
    summaries, flags = [], []
    for tk in all_tickers(cfg):
        try:
            df = load_raw(tk, cfg)
        except FileNotFoundError as e:
            print(f"[skip] {e}")
            continue
        s, f = summarize(tk, df, cfg, cal)
        summaries.append(s)
        flags.append(f)
    rep = path("reports", cfg)
    summ = pd.DataFrame(summaries)
    summ.to_csv(rep / "data_quality.csv", index=False, encoding="utf-8-sig")
    pd.concat(flags).to_csv(rep / "data_quality_flags.csv", encoding="utf-8-sig")
    roll = roll_analysis(cfg, cal)
    roll.to_csv(rep / "roll_analysis.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(summ.drop(columns=["missing_by_year"]).to_string(index=False))
    print("\n先物のベーシス（log(N225_t / 先物_{t-1}) の bp）：メジャーSQ日からの営業日オフセット別")
    print(roll.to_string())
    for tk in ("NIY=F", "NKD=F"):
        s = summ[summ.ticker == tk]
        if len(s):
            print(f"\n{tk} 欠損の年別内訳: {s.iloc[0]['missing_by_year']}")


if __name__ == "__main__":
    main()
