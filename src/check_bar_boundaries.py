"""yfinance 日足の区切り時刻を 1時間足と突き合わせて確認する。

候補の確定時刻 c（bar日付 00:00 からの時間、0:30〜48:00 を 30分刻み）ごとに、
1時間足を「終了時刻が (c_{D-1}, c_D] に入る bar 日付 D」に集計し、最後の足の終値と日足の終値の
相対誤差の中央値を求める。中央値が最小の c を、その日足の区切りと判定する。
（日足の終値は引けのオークション・清算値のことがあり、1時間足の最後の値と完全には一致しない。
 bar 日付を 1日ずらすと誤差が 1桁大きくなるので、区切りは判別できる）
比較は分割・配当の調整をしない値で行う。

制約：yfinance の 1時間足は直近約 730 日分しかない。それより前は同じ区切りと仮定する。

usage: python -m src.check_bar_boundaries [--tickers ^N225 NIY=F ...]
"""
from __future__ import annotations

import argparse
from datetime import timedelta

import pandas as pd

from .align import assign_trade_date, close_offset, ticker_tz
from .config import load_config, path
from .datasource import YFinanceSource

TOL = 2e-4


def all_tickers(cfg: dict) -> list[str]:
    ts = list(cfg["tickers"].values())
    if cfg["adr"]["enabled"]:
        for p in cfg["adr"]["pairs"]:
            ts += [p["adr"], p["tokyo"]]
    return ts


def _fmt(td: timedelta) -> str:
    mins = int(td.total_seconds() // 60)
    d, rem = divmod(mins, 24 * 60)
    return f"{rem // 60:02d}:{rem % 60:02d}" + (f" (+{d}d)" if d else "")


def check_one(src: YFinanceSource, ticker: str, cfg: dict) -> dict:
    hourly = src.fetch_hourly(ticker, period="730d")
    daily = src.fetch(ticker, start=hourly.index.min().tz_convert(None).normalize() - pd.Timedelta(days=3))
    tz = ticker_tz(ticker, cfg)
    rows = []
    for k in range(1, 97):
        c = timedelta(minutes=30 * k)
        td = assign_trade_date(hourly.index, ticker, close=c, cfg=cfg)
        last = hourly.assign(trade_date=td).groupby("trade_date")["close"].last()
        common = daily.index.intersection(last.index)
        # 1時間足の最初と最後の日は集計が不完全なので除く
        common = common[(common > last.index.min()) & (common < last.index.max())]
        if len(common) < 20:
            rows.append((c, float("inf"), 0.0, len(common)))
            continue
        rel = (last.loc[common] / daily.loc[common, "close"] - 1).abs()
        rows.append((c, float(rel.median()), float((rel < TOL).mean()), len(common)))
    res = pd.DataFrame(rows, columns=["c", "med_err", "rate", "n"])
    best_err = res["med_err"].min()
    tied = res[res["med_err"] <= best_err * 1.05 + 1e-6]
    c_best = tied["c"].min()
    best = float(tied["rate"].max())
    # 採用した区切りで、各 bar 日付の最後の 1時間足の終了時刻（現地時刻）
    td = assign_trade_date(hourly.index, ticker, close=c_best, cfg=cfg)
    end_local = (hourly.index.tz_convert(tz) + pd.Timedelta(hours=1)).tz_localize(None)
    last_end = pd.Series(end_local, index=hourly.index).groupby(td).max()
    last_end_tod = (last_end - last_end.index).map(_fmt)
    configured = close_offset(ticker, daily.index.max(), cfg)
    return {
        "ticker": ticker, "tz": tz, "n_days": int(tied["n"].max()),
        "median_rel_err_bp": round(best_err * 1e4, 2),
        "match_rate": round(best, 4),
        "boundary_earliest": _fmt(tied["c"].min()), "boundary_latest": _fmt(tied["c"].max()),
        "last_bar_end_mode": last_end_tod.mode().iloc[0] if len(last_end_tod) else "",
        "configured_close": _fmt(configured),
        # 設定した確定時刻が、日足に入る最後のデータ時刻（最頻値）以降なら OK
        "configured_ok": bool(configured >= (last_end - last_end.index).mode().iloc[0]) if len(last_end) else False,
        "hourly_from": str(hourly.index.min().date()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="*")
    args = ap.parse_args(argv)
    cfg = dict(load_config())
    cfg["auto_adjust_tickers"] = []   # 調整なしで比較する
    src = YFinanceSource(cfg)
    out = []
    for t in args.tickers or all_tickers(cfg):
        try:
            r = check_one(src, t, cfg)
        except Exception as e:
            r = {"ticker": t, "error": str(e)}
        print(r)
        out.append(r)
    df = pd.DataFrame(out)
    dest = path("reports", cfg) / "bar_boundaries.csv"
    df.to_csv(dest, index=False, encoding="utf-8-sig")
    print(f"\n保存しました: {dest}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
