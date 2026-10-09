"""データ取得と parquet キャッシュ（差分更新）。

- 日足：data/raw/{name}.parquet（index=bar日付、列=open..volume, available_at_utc, synth）
- 1時間足：data/raw/{name}_1h.parquet（直近分のみ。日足の補完と暫定 B1 に使う）
- yfinance の日足は最新日が遅れて出ることがある（例：^N225）。1時間足に日足より新しい
  確定済みの bar 日付があれば、1時間足から日足を作って補う（synth=True）。

usage: python -m src.fetch_data [--full] [--tickers ...] [--no-hourly]
"""
from __future__ import annotations

import argparse
import io

import pandas as pd

from .align import assign_trade_date, available_at_utc
from .check_bar_boundaries import all_tickers
from .config import load_config, path, safe_name
from .datasource import OHLCV, DataSource, YFinanceSource
from .locks import atomic_write_bytes, lock_for

REFETCH_DAYS = 15
HOURLY_KEEP_DAYS = 30


def raw_path(ticker: str, cfg: dict | None = None, hourly: bool = False):
    return path("raw", cfg) / f"{safe_name(ticker)}{'_1h' if hourly else ''}.parquet"


def _write_parquet(df: pd.DataFrame, dest) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf)
    with lock_for(dest):
        atomic_write_bytes(dest, buf.getvalue())


def load_raw(ticker: str, cfg: dict | None = None, hourly: bool = False) -> pd.DataFrame:
    p = raw_path(ticker, cfg, hourly)
    if not p.exists():
        raise FileNotFoundError(f"{p} がありません。先に python -m src.fetch_data を実行してください")
    with lock_for(p):
        return pd.read_parquet(p)


def synth_from_hourly(daily: pd.DataFrame, hourly: pd.DataFrame, ticker: str, cfg: dict,
                      now_utc: pd.Timestamp | None = None) -> pd.DataFrame:
    """日足より新しい bar 日付を 1時間足から作る。確定時刻を過ぎた bar 日付だけを対象にする。"""
    if hourly.empty:
        return daily
    now_utc = now_utc or pd.Timestamp.now(tz="UTC")
    td = assign_trade_date(hourly.index, ticker, cfg=cfg)
    last_daily = daily.index.max() if len(daily) else pd.Timestamp("1900-01-01")
    h = hourly.assign(trade_date=td)
    h = h[h["trade_date"] > last_daily]
    if h.empty:
        return daily
    g = h.sort_index().groupby("trade_date")
    s = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "volume": g["volume"].sum()})
    s.index = pd.DatetimeIndex(s.index)
    s.index.name = "date"
    s["available_at_utc"] = available_at_utc(s.index, ticker, cfg)
    s = s[s["available_at_utc"] <= now_utc]
    # 土日など、日足の取引日にならない日付は作らない
    s = s[s.index.weekday < 5]
    if s.empty:
        return daily
    s["synth"] = True
    return pd.concat([daily, s]).sort_index()


def update_ticker(ticker: str, src: DataSource, cfg: dict, full: bool = False, hourly: bool = True,
                  now_utc: pd.Timestamp | None = None) -> pd.DataFrame:
    p = raw_path(ticker, cfg)
    old = None
    if p.exists() and not full:
        old = load_raw(ticker, cfg)
        old = old[~old.get("synth", pd.Series(False, index=old.index)).fillna(False).astype(bool)]
        start = max(pd.Timestamp(cfg["start_date"]), old.index.max() - pd.Timedelta(days=REFETCH_DAYS))
    else:
        start = pd.Timestamp(cfg["start_date"])
    new = src.fetch(ticker, start=start)
    daily = new if old is None else pd.concat([old[old.index < new.index.min()], new])
    daily = daily[~daily.index.duplicated(keep="last")].sort_index()
    # 確定時刻は設定から毎回計算し直す（設定変更に追従させる）
    daily["available_at_utc"] = available_at_utc(daily.index, ticker, cfg)
    daily["synth"] = False
    # 確定時刻を過ぎていない（取引中の）行は保存しない
    daily = daily[daily["available_at_utc"] <= (now_utc or pd.Timestamp.now(tz="UTC"))]

    if hourly and isinstance(src, YFinanceSource):
        hp = raw_path(ticker, cfg, hourly=True)
        try:
            h_new = src.fetch_hourly(ticker, period="7d")
            if hp.exists():
                h_old = load_raw(ticker, cfg, hourly=True)
                h_new = pd.concat([h_old, h_new])
                h_new = h_new[~h_new.index.duplicated(keep="last")].sort_index()
            cut = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=HOURLY_KEEP_DAYS)
            h_new = h_new[h_new.index >= cut]
            _write_parquet(h_new, hp)
            daily = synth_from_hourly(daily, h_new, ticker, cfg, now_utc)
        except RuntimeError as e:
            print(f"  [warn] {ticker} 1時間足の取得に失敗: {e}")
    daily["synth"] = daily["synth"].fillna(False).astype(bool)
    _write_parquet(daily, p)
    return daily


def update_all(cfg: dict | None = None, tickers: list[str] | None = None, full: bool = False,
               hourly: bool = True, src: DataSource | None = None, verbose: bool = True) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    src = src or YFinanceSource(cfg)
    out = {}
    for t in tickers or all_tickers(cfg):
        df = update_ticker(t, src, cfg, full=full, hourly=hourly)
        out[t] = df
        if verbose:
            n_syn = int(df["synth"].sum())
            print(f"{t:>10}: {df.index.min().date()} 〜 {df.index.max().date()}  {len(df)}行"
                  + (f"（1時間足から補完 {n_syn}行）" if n_syn else ""))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="キャッシュを使わずに全期間を取得し直す")
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--no-hourly", action="store_true")
    args = ap.parse_args(argv)
    update_all(tickers=args.tickers, full=args.full, hourly=not args.no_hourly)


if __name__ == "__main__":
    main()
