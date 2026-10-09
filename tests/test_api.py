"""API のテスト。合成データ（一時ディレクトリ）と小さなモデルを作り、データ取得はモックにする。"""
import copy
import os

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from src import config as config_mod
from src.align import available_at_utc
from src.calendar_jp import nyse_holidays, rule_is_open
from src.check_bar_boundaries import all_tickers
from src.config import JST, ROOT, safe_name


def _synthetic_sources(cfg, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(0)
    start = end - pd.Timedelta(days=4 * 365)
    cal_days = pd.date_range(start, end + pd.Timedelta(days=2))
    latent = pd.Series(np.exp(np.log(30000) + np.cumsum(rng.normal(0, 0.008, len(cal_days)))), index=cal_days)
    out = {}
    groups = cfg["availability"]["ticker_group"]
    for t in all_tickers(cfg):
        g = groups[t]
        days = pd.bdate_range(start, end)
        if g.startswith("tse"):
            days = pd.DatetimeIndex([d for d in days if rule_is_open(d, cfg)])
        elif g in ("us_stock", "cme"):
            days = days[[d.date() not in nyse_holidays(d.year) for d in days]]
        if t == "^N225":
            close = latent.reindex(days).values
        elif t in ("NIY=F", "NKD=F"):   # 米国日付 D の先物 ≒ 翌日朝の水準
            close = latent.reindex(days + pd.Timedelta(days=1)).values * np.exp(rng.normal(0, 0.002, len(days)))
            close = np.round(close / 5) * 5
        elif t == "6J=F":
            close = 0.0067 * np.exp(np.cumsum(rng.normal(0, 0.004, len(days))))
        elif t == "^VIX":
            close = np.clip(20 * np.exp(np.cumsum(rng.normal(0, 0.03, len(days)))), 9, 80)
        elif t == "^TNX":
            close = 4 + np.cumsum(rng.normal(0, 0.03, len(days)))
        else:
            close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
        df = pd.DataFrame({"open": close * 0.999, "high": close * 1.004, "low": close * 0.996, "close": close,
                           "volume": 1000.0}, index=days)
        df.index.name = "date"
        df["available_at_utc"] = available_at_utc(df.index, t, cfg)
        df["synth"] = False
        out[t] = df
    return out


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """一時ディレクトリに config・合成データ・モデルを用意し、NIKKEI_CONFIG で切り替える。"""
    tmp = tmp_path_factory.mktemp("api")
    base = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg = copy.deepcopy(base)
    for k in cfg["paths"]:
        cfg["paths"][k] = str(tmp / k)
    cfg["tune"]["params_file"] = str(tmp / "no_params.json")
    cfg["walk_forward"]["max_rounds"] = 200
    cfg["api"]["min_rerun_interval_min"] = 10
    p = tmp / "config.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    old = os.environ.get("NIKKEI_CONFIG")
    os.environ["NIKKEI_CONFIG"] = str(p)
    cfg = config_mod.load_config()

    now = pd.Timestamp.now(tz=JST)
    raw = config_mod.path("raw", cfg)
    srcs = _synthetic_sources(cfg, now.tz_localize(None).normalize())
    for t, df in srcs.items():
        df = df[df["available_at_utc"] <= now.tz_convert("UTC")]
        df.to_parquet(raw / f"{safe_name(t)}.parquet")

    from src.features import build
    from src.train import train_final
    train_final(cfg, build(cfg=cfg))
    yield cfg
    if old is None:
        os.environ.pop("NIKKEI_CONFIG", None)
    else:
        os.environ["NIKKEI_CONFIG"] = old


@pytest.fixture
def client(env, monkeypatch):
    import src.fetch_data as fd
    monkeypatch.setattr(fd, "update_all", lambda *a, **k: {})   # データ取得はモック
    from src.api.main import app, state
    state.recent.clear()
    with TestClient(app) as c:
        yield c


def _past_open_day(cfg, back=10):
    from src.fetch_data import load_raw
    return load_raw("^N225", cfg).index[-back]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_version"]
    assert any(d["ticker"] == "^N225" and d["last_date"] for d in body["data"])


def test_run_backfill_and_get(client, env):
    d = _past_open_day(env)
    r = client.post("/predict/run", json={"target_date": str(d.date())})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "backfill" and body["date"] == str(d.date())
    assert any("in-sample" in w for w in body["warnings"])
    g = client.get(f"/predict/{d.date()}")
    assert g.status_code == 200 and g.json()["pred_close"] == body["pred_close"]
    # 間隔内の再実行は前回の結果を返す
    r2 = client.post("/predict/run", json={"target_date": str(d.date())})
    assert r2.json()["cached"] is True
    lst = client.get("/predictions", params={"from": str(d.date()), "to": str(d.date())})
    assert lst.status_code == 200 and lst.json()[0]["actual_close"] is not None
    assert lst.json()[0]["err_bp"] is not None


def test_not_found_and_errors(client, env):
    assert client.get("/predict/2001-01-05").status_code == 404
    r = client.post("/predict/run", json={"target_date": "2100-01-04"})
    assert r.status_code == 422 and r.json()["error"] == "future_date"
    # 休場日（土曜）
    sat = _past_open_day(env, 30)
    sat = sat + pd.Timedelta(days=(5 - sat.weekday()) % 7)
    r = client.post("/predict/run", json={"target_date": str(sat.date())})
    assert r.status_code == 404 and r.json()["error"] == "market_closed"
    assert client.post("/predict/run", json={"target_date": "2026-13-01"}).status_code == 422


def test_busy_returns_409(client, env):
    from src.inference import run_lock
    lock = run_lock(env, timeout=0)
    lock.acquire()
    try:
        r = client.post("/predict/run", json={"target_date": str(_past_open_day(env, 12).date())})
        assert r.status_code == 409 and r.json()["error"] == "busy"
    finally:
        lock.release()


def test_predict_today(client, env):
    from src.calendar_jp import TSECalendar
    from src.fetch_data import load_raw
    today = pd.Timestamp.now(tz=JST).tz_localize(None).normalize()
    if not TSECalendar(load_raw("^N225", env).index, env).is_open(today):
        assert client.get("/predict/today").status_code == 404
        return
    r = client.post("/predict/run", json=None)
    assert r.status_code == 200, r.text
    t = client.get("/predict/today")
    assert t.status_code == 200 and t.json()["stale"] is False


def test_backfill_ignores_data_after_cutoff(client, env):
    """target_date の JST 8:00 以降に確定したデータを書き換えても、予測は変わらない。"""
    from src.config import path
    from src.fetch_data import raw_path
    d = _past_open_day(env, 15)
    r1 = client.post("/predict/run", json={"target_date": str(d.date())}).json()
    p = raw_path("NIY=F", env)
    orig = pd.read_parquet(p)
    try:
        mod = orig.copy()
        after = mod.index >= d     # 米国日付 d 以降の足は、d の 8:00 JST より後に確定する
        mod.loc[after, ["open", "high", "low", "close"]] *= 3.0
        mod.to_parquet(p)
        from src.api.main import state
        state.recent.clear()
        r2 = client.post("/predict/run", json={"target_date": str(d.date())}).json()
        assert r2["pred_close"] == r1["pred_close"]
        assert r2["b1"] == r1["b1"]
    finally:
        orig.to_parquet(p)


def test_cli_and_api_give_same_prediction(client, env):
    from src.predict_today import main as cli
    from src.storage import read_log
    d = _past_open_day(env, 20)
    api = client.post("/predict/run", json={"target_date": str(d.date())}).json()
    assert cli(["--date", str(d.date()), "--no-fetch"]) == 0
    log = read_log(env)
    last = log[log["date"] == str(d.date())].iloc[-1]
    assert float(last["pred_close"]) == api["pred_close"]


def test_api_key(client, env, monkeypatch):
    from src.api import main as m
    cfg = copy.deepcopy(env)
    cfg["api"]["api_key_required"] = True
    monkeypatch.setattr(m, "load_config", lambda: cfg)
    monkeypatch.setenv("NIKKEI_API_KEY", "secret")
    assert client.get("/predict/2001-01-05").status_code == 401
    assert client.get("/predict/2001-01-05", headers={"X-API-Key": "secret"}).status_code == 404
    assert client.get("/health").status_code == 200
