"""推論処理（CLI・API 共通）。

- load_model(path)        学習済みモデルとメタ情報を読む
- predict(target_date, …)  キャッシュ済みデータだけで予測する（取得・ログ書き込みはしない）
- run_pipeline(…)          差分取得 → predict → ログ追記 → latest.json 保存

使うデータは常に「確定時刻 < JST target_date 8:00」のものだけ（バックテストと同じ align.py）。
mode：live（当日分を 8:00 前に実行）/ late（当日分を 8:00 以降に実行）/ backfill（過去日付）
"""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .align import cutoff_utc
from .calendar_jp import TSECalendar, last_us_weekday_before
from .config import JST, ROOT, load_config, path
from .features import build, load_sources
from .locks import lock_for
from .target import BASE_COL, to_price


class MarketClosed(Exception):
    pass


class FutureDate(Exception):
    pass


class DataFetchError(Exception):
    pass


class Busy(Exception):
    pass


@dataclass
class PredictionResult:
    date: str
    pred_close: float
    pred_return_pct: float
    b1: float | None
    b0: float
    diff_vs_b1_yen: float | None
    diff_vs_b1_bp: float | None
    data_asof: str | None
    model_version: str
    is_fallback: bool
    warnings: list[str] = field(default_factory=list)
    created_at: str = ""
    mode: str = "live"

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, (np.floating, float)) and not np.isfinite(v):
                d[k] = None
            elif isinstance(v, np.floating):
                d[k] = float(v)
        return d


def load_model(model_dir: str | Path | None = None):
    d = Path(model_dir) if model_dir else path("models")
    with open(d / "production.pkl", "rb") as f:
        model = pickle.load(f)
    meta = json.loads((d / "production.json").read_text(encoding="utf-8"))
    return model, meta


def _now_jst(now=None) -> pd.Timestamp:
    """現在時刻（JST）。now に tz なしの時刻を渡したら JST とみなす（テスト用）。"""
    ts = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if ts.tzinfo is None:
        ts = ts.tz_localize(JST)
    return ts.tz_convert(JST)


def resolve_target(target_date, cal: TSECalendar, now_jst: pd.Timestamp) -> pd.Timestamp:
    today = now_jst.tz_localize(None).normalize()
    t = today if target_date is None else pd.Timestamp(target_date).normalize()
    if t > today:
        raise FutureDate(f"{t.date()} は未来の日付です")
    if not cal.is_open(t):
        raise MarketClosed(f"{t.date()} は東証の休場日です")
    return t


def _provisional_niy(cfg: dict, t: pd.Timestamp, now_utc: pd.Timestamp) -> tuple[float, str] | None:
    """日足が未確定のときの暫定 B1：確定時刻の制約内で最新の NIY 1時間足の終値。"""
    from .fetch_data import load_raw
    try:
        h = load_raw(cfg["tickers"]["niy"], cfg, hourly=True)
    except FileNotFoundError:
        return None
    end = h.index + pd.Timedelta(hours=1)
    lim = min(now_utc, cutoff_utc([t], cfg)[0])
    h = h[end <= lim]
    if h.empty:
        return None
    last_end = (h.index[-1] + pd.Timedelta(hours=1)).tz_convert(JST)
    return float(h["close"].iloc[-1]), last_end.isoformat()


def predict(target_date=None, model=None, meta: dict | None = None, cfg: dict | None = None,
            src: dict | None = None, now=None) -> PredictionResult:
    cfg = cfg or load_config()
    src = src or load_sources(cfg)
    now_jst = _now_jst(now)
    now_utc = now_jst.tz_convert("UTC")
    cal = TSECalendar(src["n225"].index, cfg)
    t = resolve_target(target_date, cal, now_jst)
    today = now_jst.tz_localize(None).normalize()
    cut = cutoff_utc([t], cfg)[0]
    mode = "backfill" if t < today else ("late" if now_utc >= cut else "live")

    dates = cal.history[cal.history < t].append(pd.DatetimeIndex([t]))
    X = build(dates, cfg, src)
    row = X.loc[t]
    warnings: list[str] = []
    margin = pd.Timedelta(minutes=cfg["availability"]["margin_min"])

    if mode == "late":
        warnings.append("8:00 以降の実行です。6J=F・CL=F・GC=F の直近の日足には、8:00 以降の取引が含まれている"
                        "可能性があります（yfinance が次の取引の値で上書きするため。NIY・NKD は 1時間足から作り直し済み）")
    if meta and t <= pd.Timestamp(meta["train_end"]):
        warnings.append(f"in-sample：{t.date()} はモデルの学習期間（〜{meta['train_end']}）に含まれます")

    # 入力データの鮮度
    exp_us = last_us_weekday_before(t)
    stale_days = cfg["live"]["stale_warning_days"]
    for key in ("spx", "sox", "jpyfut", "nkd"):
        sd = row.get(f"{key}_src_date")
        if pd.isna(sd):
            warnings.append(f"{key}：確定済みのデータがありません")
        elif pd.Timestamp(sd) < exp_us:
            warnings.append(f"{key}：最新の日付が {pd.Timestamp(sd).date()} です（想定 {exp_us.date()}。米国休場か未更新）")
        elif (t - pd.Timestamp(sd)).days > stale_days:
            warnings.append(f"{key}：データが {stale_days} 日以上古いです")

    n_date = row.get("n225_src_date")
    if not pd.isna(n_date) and pd.Timestamp(n_date) in src["n225"].index:
        if str(src["n225"].loc[pd.Timestamp(n_date)].get("synth_src", "")) == "hourly":
            warnings.append(f"前日（{pd.Timestamp(n_date).date()}）の日経平均の終値が正式な値ではありません"
                            "（1時間足から補完。引けのオークションの値を含まない）")

    niy_date = row.get("niy_src_date")
    niy_avail = row.get("niy_src_avail")
    fallback_kind = None
    b1 = row.get("niy_close")
    data_asof = None if pd.isna(niy_avail) else pd.Timestamp(niy_avail).tz_convert(JST).isoformat()
    if pd.isna(niy_date) or pd.Timestamp(niy_date) < exp_us:
        warnings.append(f"CME（NIY=F）の {exp_us.date()} の日足が未確定です")
        prov = _provisional_niy(cfg, t, now_utc)
        if prov:
            b1, data_asof = prov
            fallback_kind = "fallback-B1-provisional"
        else:
            b1 = None
            fallback_kind = "fallback-B0"
    elif now_utc < pd.Timestamp(niy_avail) - margin:
        warnings.append("CME（NIY=F）の日足が確定時刻前の値です")
        fallback_kind = "fallback-B1-provisional"
    missing = [c for c in cfg["features"]["critical"] if pd.isna(row.get(c))]
    if missing and fallback_kind is None:
        warnings.append(f"重要な特徴量が欠損しています：{missing}")
        fallback_kind = "fallback-B1"

    b0 = float(row["n225_close"])
    if fallback_kind:
        pred_close = float(b1) if b1 is not None and not pd.isna(b1) else b0
        version = fallback_kind
    else:
        feats = model.features
        bp = float(model.predict(X.loc[[t], feats])[0])
        base = float(row[BASE_COL[meta["target"]]])
        pred_close = float(to_price(bp / 1e4, base))
        version = meta["model_version"]

    b1f = None if b1 is None or pd.isna(b1) else float(b1)
    return PredictionResult(
        date=str(t.date()), pred_close=round(pred_close, 2),
        pred_return_pct=round((pred_close / b0 - 1) * 100, 4),
        b1=None if b1f is None else round(b1f, 2), b0=round(b0, 2),
        diff_vs_b1_yen=None if b1f is None else round(pred_close - b1f, 2),
        diff_vs_b1_bp=None if b1f is None else round(float(np.log(pred_close / b1f) * 1e4), 2),
        data_asof=data_asof, model_version=version, is_fallback=fallback_kind is not None,
        warnings=warnings, created_at=now_jst.isoformat(), mode=mode,
    )


def run_lock(cfg: dict | None = None, timeout: float = -1):
    """予測の実行ロック（プロセスをまたぐ）。timeout=0 なら待たずに Busy を投げる。"""
    return lock_for(path("predictions", cfg) / ".run", timeout=timeout)


def run_pipeline(target_date=None, cfg: dict | None = None, model=None, meta: dict | None = None,
                 fetch: bool = True, on_fetch_error: str = "warn", lock_timeout: float | None = None,
                 now=None) -> PredictionResult:
    from filelock import Timeout

    from .fetch_data import update_all
    from .storage import log_prediction, save_latest

    cfg = cfg or load_config()
    timeout = cfg["api"]["run_lock_timeout_sec"] if lock_timeout is None else lock_timeout
    try:
        lock = run_lock(cfg, timeout)
        lock.acquire()
    except Timeout:
        raise Busy("別の予測処理が実行中です")
    try:
        if model is None:
            model, meta = load_model()
        fetch_warn = None
        if fetch:
            try:
                update_all(cfg, verbose=False)
            except Exception as e:
                if on_fetch_error == "raise":
                    raise DataFetchError(str(e))
                fetch_warn = f"データ取得に失敗したため、キャッシュ済みのデータで予測しました：{e}"
        res = predict(target_date, model, meta, cfg, now=now)
        if fetch_warn:
            res.warnings.insert(0, fetch_warn)
        log_prediction(res, cfg)
        today = _now_jst(now).tz_localize(None).normalize()
        if pd.Timestamp(res.date) == today:
            save_latest(res, cfg)
        return res
    finally:
        lock.release()
