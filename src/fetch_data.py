"""データ取得と parquet キャッシュ（差分更新）。

- 日足：data/raw/{name}.parquet（index=bar日付、列=open..volume, available_at_utc, synth）
- 1時間足：data/raw/{name}_1h.parquet（直近分のみ。日足の補完と暫定 B1 に使う）
- yfinance の日足は最新日が遅れて出ることがある（例：^N225）。1時間足に日足より新しい
  確定済みの bar 日付があれば、1時間足から日足を作って補う（synth=True）。東証の銘柄は、終値だけ
  気配情報の正式な終値に置き換える。
- CME 先物の直近の日足は、次の取引の値で上書きされるので、1時間足から作り直す（merge_hourly）。

usage: python -m src.fetch_data [--full] [--tickers ...] [--no-hourly]
"""
from __future__ import annotations

import argparse
import io

import pandas as pd

from .align import assign_trade_date, available_at_utc
from .check_bar_boundaries import all_tickers
from .config import JST, load_config, path, safe_name
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


def hourly_daily_bars(hourly: pd.DataFrame, ticker: str, cfg: dict, now_utc: pd.Timestamp) -> pd.DataFrame:
    """1時間足から作った日足。確定時刻を過ぎた bar 日付だけ。

    1時間足の最初の bar 日付は、取引の途中から始まっている可能性があるので作らない。
    """
    if hourly.empty:
        return pd.DataFrame()
    td = assign_trade_date(hourly.index, ticker, cfg=cfg)
    h = hourly.assign(trade_date=td).sort_index()
    h = h[h["trade_date"] > h["trade_date"].min()]
    if h.empty:
        return pd.DataFrame()
    g = h.groupby("trade_date")
    s = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "volume": g["volume"].sum()})
    s.index = pd.DatetimeIndex(s.index)
    s.index.name = "date"
    s["available_at_utc"] = available_at_utc(s.index, ticker, cfg)
    s = s[(s["available_at_utc"] <= now_utc) & (s.index.weekday < 5)]   # 土日の bar 日付は作らない
    return s


def session_close_from_quote(info: dict, tse_days: pd.DatetimeIndex, cfg: dict) -> tuple[pd.Timestamp, float] | None:
    """東証の銘柄の、直近に終わった取引日の正式な終値を気配情報から求める。

    1時間足・5分足には引けのオークション（15:25〜15:30）の値が入らないので、日足が遅れているときはこれを使う。
    - 最後の約定時刻が引け（15:25）以降で、取引中でない → その日の終値 = regularMarketPrice
    - 取引中、または引け前（前場・昼休み）→ 前の取引日の終値 = regularMarketPreviousClose
    """
    from .calendar_jp import TSECalendar

    try:
        rmt = pd.Timestamp(int(info["regularMarketTime"]), unit="s", tz="UTC").tz_convert(JST)
    except (KeyError, TypeError, ValueError):
        return None
    day = rmt.tz_localize(None).normalize()
    closed_after_bell = (rmt.hour, rmt.minute) >= (15, 25) and info.get("marketState") != "REGULAR"
    if closed_after_bell and info.get("regularMarketPrice"):
        return day, float(info["regularMarketPrice"])
    if info.get("regularMarketPreviousClose"):
        return TSECalendar(tse_days, cfg).prev_open(day), float(info["regularMarketPreviousClose"])
    return None


def merge_hourly(daily: pd.DataFrame, hourly: pd.DataFrame, ticker: str, cfg: dict,
                 now_utc: pd.Timestamp | None = None, quote: tuple[pd.Timestamp, float] | None = None) -> pd.DataFrame:
    """1時間足で日足を補う。synth_src 列に、どこから作った行かを残す。

    - 共通：日足より新しい確定済みの bar 日付を、1時間足から作る（yfinance の日足は最新日が遅れることがある）
    - CME の live.rebuild_from_hourly の銘柄（NIY・NKD）：直近 live.cme_rebuild_days 日分は、日足を捨てて
      1時間足から作り直す。yfinance は、次の取引（JST 7:00〜、冬は 8:00〜）の値で前日の日足を上書きするため。
    - 東証（tse_* グループ）：1時間足から作った行の終値は、気配情報の正式な終値（quote）で置き換える。
    """
    now_utc = now_utc or pd.Timestamp.now(tz="UTC")
    bars = hourly_daily_bars(hourly, ticker, cfg, now_utc)
    if bars.empty:
        return daily
    group = cfg["availability"]["ticker_group"][ticker]
    last_daily = daily.index.max() if len(daily) else pd.Timestamp("1900-01-01")
    if group == "cme" and ticker in cfg["live"].get("rebuild_from_hourly", []):
        repl = bars.tail(cfg["live"]["cme_rebuild_days"])
        repl = pd.concat([repl, bars[bars.index > last_daily]])
        repl = repl[~repl.index.duplicated(keep="last")]
        # 入れ替える日付と、それより新しい日付（次の取引の値で上書きされた行）の日足は使わない
        daily = daily[~daily.index.isin(repl.index) & (daily.index <= repl.index.max())]
        add = repl.assign(synth=True, synth_src="hourly")
    else:
        add = bars[bars.index > last_daily].assign(synth=True, synth_src="hourly")
        if group.startswith("tse") and quote is not None and quote[0] in add.index:
            d, c = quote
            add.loc[d, "close"] = c
            add.loc[d, "high"] = max(add.loc[d, "high"], c)
            add.loc[d, "low"] = min(add.loc[d, "low"], c)
            add.loc[d, "synth_src"] = "hourly+quote"
    if add.empty:
        return daily
    return pd.concat([daily, add]).sort_index()


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
    daily["synth_src"] = ""
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
            quote = None
            if cfg["availability"]["ticker_group"][ticker].startswith("tse"):
                info = src.fetch_quote(ticker)
                try:
                    tse_days = daily.index if ticker == cfg["tickers"]["n225"] \
                        else load_raw(cfg["tickers"]["n225"], cfg).index
                except FileNotFoundError:
                    tse_days = daily.index
                quote = session_close_from_quote(info, tse_days, cfg) if info else None
            daily = merge_hourly(daily, h_new, ticker, cfg, now_utc, quote)
        except RuntimeError as e:
            print(f"  [warn] {ticker} 1時間足の取得に失敗: {e}")
    daily["synth"] = daily["synth"].fillna(False).astype(bool)
    daily["synth_src"] = daily["synth_src"].fillna("").astype(str)
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
