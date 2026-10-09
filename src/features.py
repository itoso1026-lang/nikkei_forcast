"""特徴量生成。

手順：
1. 各データ源の「自分の取引日」上で、変化率などの native 特徴量を作る（過去方向の窓だけを使う）
2. align.asof_align で予測対象日 t に揃える（確定時刻 < JST t日 8:00 の最新行）
3. t 上で交差特徴量・カレンダー特徴量を作る
価格水準そのものは特徴量に入れない（*_close 列は基準価格・ギャップ計算用で、FEATURE から除外する）。

usage: python -m src.features
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .align import asof_align, assert_no_leak
from .calendar_jp import TSECalendar, nyse_holidays
from .config import load_config, path
from .data_quality import flag_rows
from .fetch_data import load_raw
from .target import make_target

PRICE_COLS_SUFFIX = ("_close",)
NON_FEATURE_SUFFIX = ("_src_date", "_src_avail", "_close")


def load_sources(cfg: dict) -> dict[str, pd.DataFrame]:
    src = {k: load_raw(t, cfg) for k, t in cfg["tickers"].items()}
    if cfg["adr"]["enabled"]:
        for p in cfg["adr"]["pairs"]:
            src[f"adr_{p['name']}"] = load_raw(p["adr"], cfg)
            src[f"tky_{p['name']}"] = load_raw(p["tokyo"], cfg)
    return src


def _logret(c: pd.Series, n: int = 1) -> pd.Series:
    c = c.where(c > 0)
    return np.log(c / c.shift(n))


def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


def _clean(df: pd.DataFrame, cfg: dict, protect: bool = False) -> pd.DataFrame:
    if protect or cfg["data_quality"]["outlier_action"] != "flag_ffill":
        return df
    f = flag_rows(df, cfg)
    df = df.copy()
    df.loc[f["outlier"], ["open", "high", "low", "close"]] = np.nan
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill()
    return df


def native_features(src: dict[str, pd.DataFrame], cfg: dict, tse_days: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """データ源ごとの native 特徴量。各 DataFrame は available_at_utc 列を持つ。"""
    out = {}

    def keep(df, cols):
        return df[cols].assign(available_at_utc=df["available_at_utc"])

    # 日経平均（N225 自身はクリーニングしない）
    n = src["n225"].copy()
    r = _logret(n["close"])
    n["r1"] = r
    n["oc"] = np.log(n["close"] / n["open"])
    n["rng"] = np.log(n["high"] / n["low"])
    n["rv5"] = r.rolling(5).std()
    n["rv20"] = r.rolling(20).std()
    n["ma20dev"] = np.log(n["close"] / n["close"].rolling(20).mean())
    n["ma60dev"] = np.log(n["close"] / n["close"].rolling(60).mean())
    n["rsi14"] = _rsi(n["close"])
    out["n225"] = keep(n, ["close", "r1", "oc", "rng", "rv5", "rv20", "ma20dev", "ma60dev", "rsi14"])

    # 先物：NIY は終値（B1・T2 の基準）と変化率。OHLC が平らな日が多いので、レンジは NKD で作る
    f = _clean(src["niy"], cfg)
    f["r1"] = _logret(f["close"])
    f["r5"] = _logret(f["close"], 5)
    # OHLC が平らで出来高0の日（終値が米国引けの値になっていないことがある）
    f["flat"] = ((f["open"] == f["close"]) & (f["high"] == f["low"]) & (f["volume"].fillna(0) == 0)).astype(int)
    out["niy"] = keep(f, ["close", "r1", "r5", "flat"])
    k = _clean(src["nkd"], cfg)
    k["r1"] = _logret(k["close"])
    k["rng"] = np.log(k["high"] / k["low"])
    k["pos"] = ((k["close"] - k["low"]) / (k["high"] - k["low"])).where(k["high"] > k["low"])
    out["nkd"] = keep(k, ["close", "r1", "rng", "pos"])

    # 為替：6J=F は 1円あたりのドルなので、USDJPY = 1/6J
    j = _clean(src["jpyfut"], cfg)
    usdjpy = 1.0 / j["close"]
    j["usdjpy_r1"] = _logret(usdjpy)
    j["usdjpy_r5"] = _logret(usdjpy, 5)
    j["usdjpy_level"] = usdjpy
    out["jpyfut"] = keep(j, ["usdjpy_r1", "usdjpy_r5", "usdjpy_level"])
    u = _clean(src["usdjpy"], cfg)          # JPY=X（確定時刻の設定により t-2 以前だけが揃う）
    u["r1"] = _logret(u["close"])
    u["r5"] = _logret(u["close"], 5)
    out["usdjpy"] = keep(u, ["r1", "r5"])

    for key in ("spx", "ixic", "dji", "sox"):
        s = _clean(src[key], cfg)
        for w in (1, 5, 20):
            s[f"r{w}"] = _logret(s["close"], w)
        out[key] = keep(s, ["r1", "r5", "r20"])

    v = src["vix"].copy()
    v["lvl"] = np.log(v["close"])
    v["chg1"] = _logret(v["close"])
    v["chg5"] = _logret(v["close"], 5)
    out["vix"] = keep(v, ["lvl", "chg1", "chg5"])
    tn = src["tnx"].copy()
    tn["d1"] = tn["close"].diff()
    tn["d5"] = tn["close"].diff(5)
    out["tnx"] = keep(tn, ["d1", "d5"])

    for key in ("dax", "sx5e", "wti", "gold"):
        s = _clean(src[key], cfg)
        s["r1"] = _logret(s["close"])
        s["r5"] = _logret(s["close"], 5)
        out[key] = keep(s, ["r1", "r5"])

    if cfg["adr"]["enabled"]:
        for p in cfg["adr"]["pairs"]:
            a = src[f"adr_{p['name']}"]
            out[f"adr_{p['name']}"] = keep(a, ["close"])
            t = src[f"tky_{p['name']}"]
            t = t[t.index.isin(tse_days)]   # 東京の個別株には祝日の行が混ざっていることがある
            out[f"tky_{p['name']}"] = keep(t, ["close"])
    return out


def calendar_features(t: pd.DatetimeIndex, cal: TSECalendar, cfg: dict, us_src_date: pd.Series) -> pd.DataFrame:
    c = pd.DataFrame(index=t)
    c["dow"] = t.weekday
    c["month"] = t.month
    prev = pd.DatetimeIndex([cal.prev_open(d) for d in t])
    nxt = pd.DatetimeIndex([cal.next_open(d) for d in t])
    c["gap_days"] = (t - prev).days
    c["month_end"] = (nxt.month != t.month).astype(int)
    c["month_start"] = (prev.month != t.month).astype(int)

    years = range(t.min().year - 1, t.max().year + 2)
    sq = pd.DatetimeIndex(sorted(cal.sq_date(y, m) for y in years for m in range(1, 13)))
    major = sq[sq.month.isin([3, 6, 9, 12])]
    c["sq_flag"] = t.isin(sq).astype(int)
    c["major_sq_flag"] = t.isin(major).astype(int)
    days = cal.days(t.min() - pd.Timedelta(days=10), max(major.max(), t.max()) + pd.Timedelta(days=10))
    pos = pd.Series(np.arange(len(days)), index=days)

    def bdays_to_next(targets: pd.DatetimeIndex) -> np.ndarray:
        idx = np.searchsorted(targets.values, t.values, side="left")
        nxt_ev = targets[np.minimum(idx, len(targets) - 1)]
        return pos.reindex(nxt_ev).values - pos.reindex(t).values

    c["days_to_sq"] = bdays_to_next(sq)
    c["days_to_major_sq"] = bdays_to_next(major)
    off = cfg["calendar"]["roll_offset_bdays"]
    roll_days = pd.DatetimeIndex([days[pos[m] + off] for m in major if m in pos.index and pos[m] + off < len(days)])
    c["roll_flag"] = t.isin(roll_days).astype(int)

    exdiv = pd.DatetimeIndex(sorted(cal.exdiv_date(y, m) for y in years for m in cfg["calendar"]["exdiv_months"]))
    c["exdiv_flag"] = t.isin(exdiv).astype(int)
    # 次のメジャーSQ までに配当落ちがある（先物が配当分だけ現物より安い期間）
    idx = np.searchsorted(exdiv.values, t.values, side="left")
    next_ex = exdiv[np.minimum(idx, len(exdiv) - 1)]
    idx2 = np.searchsorted(major.values, t.values, side="left")
    next_major = major[np.minimum(idx2, len(major) - 1)]
    c["exdiv_before_major_sq"] = ((next_ex >= t) & (next_ex <= next_major)).astype(int)

    # 前の平日が NYSE の休場だった
    prev_wd = t - pd.to_timedelta(np.where(t.weekday == 0, 3, 1), unit="D")
    c["us_holiday_prev"] = [int(d.date() in nyse_holidays(d.year)) for d in prev_wd]
    c["us_gap_days"] = (t - pd.DatetimeIndex(us_src_date)).days

    ev = cfg.get("events") or {}
    if ev.get("fomc"):
        f = pd.DatetimeIndex(pd.to_datetime(ev["fomc"]))
        # 前回の東証の引け以降（米国日付で prev〜t-1）に FOMC の発表があった
        c["fomc_prev"] = [int(((f >= p) & (f < d)).any()) for p, d in zip(prev, t)]
    if ev.get("boj"):
        b = pd.DatetimeIndex(pd.to_datetime(ev["boj"]))
        c["boj_today"] = t.isin(b).astype(int)
        c["boj_prev"] = prev.isin(b).astype(int)
    return c


def build(target_dates=None, cfg: dict | None = None, src: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """予測対象日 t ごとの特徴量・基準価格・目的変数のテーブル。

    列の種類：
    - 特徴量（feature_columns() で取得）
    - *_close：基準価格（n225_close = N225_{t-1}, niy_close = NIY_{t-1}）
    - *_src_date / *_src_avail：リーク検査用
    - actual_close, T1, T2：目的変数（t の実績。未確定なら NaN）
    """
    cfg = cfg or load_config()
    src = src or load_sources(cfg)
    cal = TSECalendar(src["n225"].index, cfg)
    t = pd.DatetimeIndex(target_dates if target_dates is not None else cal.history).normalize()
    nat = native_features(src, cfg, cal.history)

    parts = [asof_align(df, t, key, cfg) for key, df in nat.items()]
    X = pd.concat(parts, axis=1)
    assert_no_leak(X, cfg)

    # 先物の基準価格（B1・T2）
    X["niy_raw_close"] = X["niy_close"]
    X["niy_nkd_diff"] = np.log(X["nkd_close"] / X["niy_raw_close"])
    fut = cfg.get("futures", {"base": "niy"})
    if fut["base"] == "nkd":
        X["niy_close"] = X["nkd_close"]
    elif fut["base"] == "niy_clean":
        bad = (X["niy_flat"] == 1) & (X["niy_nkd_diff"].abs() * 1e4 > fut["clean_threshold_bp"])
        X["niy_close"] = X["niy_close"].where(~bad, X["nkd_close"])

    # 交差特徴量
    X["niy_gap"] = np.log(X["niy_close"] / X["n225_close"])
    X["nkd_gap"] = np.log(X["nkd_close"] / X["n225_close"])
    X["sox_minus_ixic_r1"] = X["sox_r1"] - X["ixic_r1"]
    X["sox_minus_ixic_r5"] = X["sox_r5"] - X["ixic_r5"]
    X["niy_stale_days"] = (t - pd.DatetimeIndex(X["niy_src_date"])).days

    actual = src["n225"]["close"].reindex(t)
    actual = actual.where(src["n225"]["available_at_utc"].reindex(t).notna())
    X["actual_close"] = actual
    X["T1"] = make_target(actual, X, "T1")
    X["T2"] = make_target(actual, X, "T2")

    fw = cfg["features"]
    w = fw["beta_window"]
    for key in ("spx", "sox"):
        # 日 d の N225 リターンと、d の前夜の米国リターンの関係。t の行には d ≤ t-1 だけを使う
        y, x = X["T1"], X[f"{key}_r1"]
        beta = (y.rolling(w, min_periods=w // 2).cov(x) / x.rolling(w, min_periods=w // 2).var()).shift(1)
        X[f"beta_{key}"] = beta
        X[f"beta_{key}_x_ret"] = beta * x
    bw = fw["basis_window"]
    X["basis_ma20"] = X["T2"].rolling(bw, min_periods=bw // 2).mean().shift(1)
    X["t2_lag1"] = X["T2"].shift(1)

    if cfg["adr"]["enabled"]:
        zs = []
        zw = fw["adr_z_window"]
        for p in cfg["adr"]["pairs"]:
            nm = p["name"]
            prem = np.log(X[f"adr_{nm}_close"] * X["jpyfut_usdjpy_level"] / p["ratio"] / X[f"tky_{nm}_close"])
            z = (prem - prem.rolling(zw, min_periods=zw // 2).mean()) / prem.rolling(zw, min_periods=zw // 2).std()
            X[f"adr_{nm}_z"] = z
            zs.append(z)
        X["adr_mean_z"] = pd.concat(zs, axis=1).mean(axis=1)

    cal_f = calendar_features(t, cal, cfg, X["spx_src_date"])
    X = pd.concat([X, cal_f], axis=1)
    X.index.name = "date"
    return X


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {"actual_close", "T1", "T2", "jpyfut_usdjpy_level"}
    return [c for c in df.columns if c not in drop and not c.endswith(NON_FEATURE_SUFFIX)]


def processed_path(cfg: dict | None = None):
    return path("processed", cfg) / "features.parquet"


def load_processed(cfg: dict | None = None) -> pd.DataFrame:
    return pd.read_parquet(processed_path(cfg))


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    cfg = load_config()
    df = build(cfg=cfg)
    df.to_parquet(processed_path(cfg))
    fc = feature_columns(df)
    print(f"保存しました: {processed_path(cfg)}  {len(df)}行 × 特徴量 {len(fc)}列")
    print("期間:", df.index.min().date(), "〜", df.index.max().date())
    na = df[fc].isna().mean().sort_values(ascending=False).head(8)
    print("欠損率の高い特徴量:\n", na.round(3).to_string())


if __name__ == "__main__":
    main()
