"""予測ログ（predictions/log.csv）と最新結果（predictions/latest.json）の読み書き。

書き込みは filelock で排他制御する（CLI と API が同時に動いても壊れないようにする）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import JST, load_config, path
from .locks import atomic_write_bytes, lock_for

LOG_COLUMNS = ["date", "pred_close", "b1", "b0", "data_asof", "model_version", "mode", "created_at"]


def log_path(cfg: dict | None = None) -> Path:
    return path("predictions", cfg) / "log.csv"


def latest_path(cfg: dict | None = None) -> Path:
    return path("predictions", cfg) / "latest.json"


def log_prediction(result, cfg: dict | None = None) -> None:
    p = log_path(cfg)
    d = result.to_dict()
    row = pd.DataFrame([{c: d.get(c) for c in LOG_COLUMNS}])
    with lock_for(p):
        new = not p.exists()
        row.to_csv(p, mode="a", header=new, index=False, encoding="utf-8")


def save_latest(result, cfg: dict | None = None) -> None:
    p = latest_path(cfg)
    data = json.dumps(result.to_dict(), ensure_ascii=False, indent=2).encode("utf-8")
    with lock_for(p):
        atomic_write_bytes(p, data)


def read_latest(cfg: dict | None = None) -> dict | None:
    p = latest_path(cfg)
    if not p.exists():
        return None
    with lock_for(p):
        return json.loads(p.read_text(encoding="utf-8"))


def read_log(cfg: dict | None = None) -> pd.DataFrame:
    p = log_path(cfg)
    if not p.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    with lock_for(p):
        df = pd.read_csv(p, dtype={"date": str, "model_version": str, "mode": str, "data_asof": str})
    df["created_at_ts"] = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert(JST)
    return df


def _before_cutoff(rows: pd.DataFrame, date: str, cfg: dict) -> pd.DataFrame:
    hh, mm = cfg["cutoff_jst"].split(":")
    cut = pd.Timestamp(date).tz_localize(JST) + pd.Timedelta(hours=int(hh), minutes=int(mm))
    return rows[rows["created_at_ts"] < cut]


def select_live(log: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """ライブ評価用：mode=live で、created_at が JST 8:00 より前の最後の行（日付ごと）。"""
    cfg = cfg or load_config()
    out = []
    for d, g in log[log["mode"] == "live"].groupby("date"):
        g = _before_cutoff(g, d, cfg)
        if len(g):
            out.append(g.sort_values("created_at_ts").iloc[-1])
    return pd.DataFrame(out, columns=log.columns)


def select_for_date(log: pd.DataFrame, date: str, cfg: dict | None = None) -> dict | None:
    """GET /predict/{date}：live（8:00 より前の最後）→ late → backfill の順に、最後の行を返す。"""
    cfg = cfg or load_config()
    g = log[log["date"] == date]
    if g.empty:
        return None
    live = _before_cutoff(g[g["mode"] == "live"], date, cfg)
    for cand in (live, g[g["mode"] == "late"], g[g["mode"] == "backfill"], g):
        if len(cand):
            r = cand.sort_values("created_at_ts").iloc[-1].drop(labels=["created_at_ts"])
            return {k: (None if pd.isna(v) else v) for k, v in r.to_dict().items()}
    return None
