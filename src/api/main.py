"""予測 API（FastAPI）。

起動：uvicorn src.api.main:app --host 127.0.0.1 --port 8000（ワーカーは1つ。モデルをメモリに持つため）
yfinance は非公式のデータ源なので、外部に公開しないこと。
"""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from datetime import date as Date

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..calendar_jp import TSECalendar
from ..check_bar_boundaries import all_tickers
from ..config import JST, load_config
from ..fetch_data import load_raw
from ..inference import (Busy, DataFetchError, FutureDate, MarketClosed, PredictionResult, load_model,
                         run_pipeline)
from ..metrics import bp_error
from ..storage import read_latest, read_log, select_for_date


# ---------- レスポンスモデル ----------
class Prediction(BaseModel):
    date: str
    pred_close: float
    pred_return_pct: float | None = None
    b1: float | None = None
    b0: float | None = None
    diff_vs_b1_yen: float | None = None
    diff_vs_b1_bp: float | None = None
    data_asof: str | None = None
    model_version: str
    is_fallback: bool | None = None
    warnings: list[str] = Field(default_factory=list)
    created_at: str | None = None
    mode: str | None = None
    stale: bool | None = Field(None, description="GET /predict/today：latest.json の日付が今日でない")
    cached: bool | None = Field(None, description="POST /predict/run：再実行の間隔内のため前回の結果を返した")


class RunRequest(BaseModel):
    target_date: Date | None = Field(None, description="YYYY-MM-DD。null なら今日")


class SourceStatus(BaseModel):
    ticker: str
    last_date: str | None
    available_at_jst: str | None


class Health(BaseModel):
    status: str
    model_version: str | None
    train_end: str | None
    server_time_jst: str
    data: list[SourceStatus]


class PredictionWithActual(Prediction):
    actual_close: float | None = None
    err_bp: float | None = None
    b1_err_bp: float | None = None


class ErrorBody(BaseModel):
    error: str
    detail: str


class ModelInfo(BaseModel):
    model_version: str
    train_end: str
    target: str
    model: str


# ---------- 状態 ----------
class State:
    def __init__(self):
        self.model = None
        self.meta: dict | None = None
        self.model_error: str | None = None
        self.run_guard = threading.Lock()
        self.recent: dict[str, tuple[pd.Timestamp, PredictionResult]] = {}

    def load(self):
        try:
            self.model, self.meta = load_model()
            self.model_error = None
        except FileNotFoundError as e:
            self.model, self.meta, self.model_error = None, None, str(e)


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.load()
    yield


app = FastAPI(title="日経平均 当日終値予測 API", version="1.0", lifespan=lifespan,
              responses={404: {"model": ErrorBody}, 409: {"model": ErrorBody}, 422: {"model": ErrorBody},
                         503: {"model": ErrorBody}})


class ApiError(Exception):
    def __init__(self, status: int, error: str, detail: str):
        self.status, self.error, self.detail = status, error, detail


@app.exception_handler(ApiError)
async def _api_error(_: Request, e: ApiError):
    return JSONResponse(status_code=e.status, content={"error": e.error, "detail": e.detail})


@app.exception_handler(HTTPException)
async def _http_error(_: Request, e: HTTPException):
    return JSONResponse(status_code=e.status_code, content={"error": "http_error", "detail": str(e.detail)})


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, e: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": "invalid_request", "detail": str(e.errors())})


def require_api_key(x_api_key: str | None = Header(None)):
    cfg = load_config()
    if not cfg["api"].get("api_key_required"):
        return
    expected = os.environ.get("NIKKEI_API_KEY")
    if not expected or x_api_key != expected:
        raise ApiError(401, "unauthorized", "X-API-Key が正しくありません")


def _today() -> pd.Timestamp:
    return pd.Timestamp.now(tz=JST).tz_localize(None).normalize()


def _calendar(cfg) -> TSECalendar:
    try:
        return TSECalendar(load_raw(cfg["tickers"]["n225"], cfg).index, cfg)
    except FileNotFoundError as e:
        raise ApiError(503, "no_data", str(e))


def _clean(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()}


def _need_model():
    if state.model is None:
        raise ApiError(503, "no_model", f"モデルが読み込まれていません：{state.model_error}")


# ---------- エンドポイント ----------
@app.get("/health", response_model=Health)
def health():
    cfg = load_config()
    data = []
    for t in all_tickers(cfg):
        try:
            df = load_raw(t, cfg)
            data.append(SourceStatus(ticker=t, last_date=str(df.index.max().date()),
                                     available_at_jst=pd.Timestamp(df["available_at_utc"].iloc[-1])
                                     .tz_convert(JST).isoformat()))
        except FileNotFoundError:
            data.append(SourceStatus(ticker=t, last_date=None, available_at_jst=None))
    m = state.meta or {}
    return Health(status="ok" if state.model is not None else "no_model", model_version=m.get("model_version"),
                  train_end=m.get("train_end"), server_time_jst=pd.Timestamp.now(tz=JST).isoformat(), data=data)


@app.get("/predict/today", response_model=Prediction, dependencies=[Depends(require_api_key)])
def predict_today():
    cfg = load_config()
    today = _today()
    if not _calendar(cfg).is_open(today):
        raise ApiError(404, "market_closed", f"{today.date()} は東証の休場日です")
    latest = read_latest(cfg)
    if latest is None:
        raise ApiError(404, "not_found", "まだ予測がありません（predict_today.py か POST /predict/run を実行してください）")
    return Prediction(**_clean(latest), stale=latest["date"] != str(today.date()))


@app.get("/predict/{date}", response_model=Prediction, dependencies=[Depends(require_api_key)])
def predict_date(date: Date):
    cfg = load_config()
    row = select_for_date(read_log(cfg), str(date), cfg)
    if row is None:
        raise ApiError(404, "not_found", f"{date} の予測はありません")
    row["is_fallback"] = str(row.get("model_version", "")).startswith("fallback")
    return Prediction(**_clean(row))


@app.post("/predict/run", response_model=Prediction, dependencies=[Depends(require_api_key)])
def predict_run(req: RunRequest | None = None):
    cfg = load_config()
    _need_model()
    target = pd.Timestamp(req.target_date) if req and req.target_date else _today()
    key = str(target.date())
    interval = pd.Timedelta(minutes=cfg["api"]["min_rerun_interval_min"])
    prev = state.recent.get(key)
    if prev and pd.Timestamp.now(tz=JST) - prev[0] < interval:
        return Prediction(**_clean(prev[1].to_dict()), cached=True)
    if not state.run_guard.acquire(blocking=False):
        raise ApiError(409, "busy", "予測処理が実行中です")
    try:
        res = run_pipeline(target, cfg, state.model, state.meta, fetch=True, on_fetch_error="raise",
                           lock_timeout=0)
    except Busy as e:
        raise ApiError(409, "busy", str(e))
    except MarketClosed as e:
        raise ApiError(404, "market_closed", str(e))
    except FutureDate as e:
        raise ApiError(422, "future_date", str(e))
    except DataFetchError as e:
        raise ApiError(503, "data_fetch_failed", str(e))
    finally:
        state.run_guard.release()
    state.recent[key] = (pd.Timestamp.now(tz=JST), res)
    return Prediction(**_clean(res.to_dict()), cached=False)


@app.get("/predictions", response_model=list[PredictionWithActual], dependencies=[Depends(require_api_key)])
def predictions(from_: Date | None = Query(None, alias="from"), to: Date | None = Query(None)):
    cfg = load_config()
    log = read_log(cfg)
    if log.empty:
        return []
    dates = sorted(set(log["date"]))
    if from_:
        dates = [d for d in dates if d >= str(from_)]
    if to:
        dates = [d for d in dates if d <= str(to)]
    try:
        n225 = load_raw(cfg["tickers"]["n225"], cfg)
    except FileNotFoundError:
        n225 = None
    now = pd.Timestamp.now(tz="UTC")
    out = []
    for d in dates:
        row = select_for_date(log, d, cfg)
        row["is_fallback"] = str(row.get("model_version", "")).startswith("fallback")
        actual = None
        if n225 is not None and pd.Timestamp(d) in n225.index:
            r = n225.loc[pd.Timestamp(d)]
            if pd.Timestamp(r["available_at_utc"]) <= now:   # 実績が確定している日のみ
                actual = float(r["close"])
        item = PredictionWithActual(**_clean(row), actual_close=actual)
        if actual is not None:
            item.err_bp = round(float(bp_error(item.pred_close, actual)), 2)
            if item.b1 is not None:
                item.b1_err_bp = round(float(bp_error(item.b1, actual)), 2)
        out.append(item)
    return out


@app.post("/model/reload", response_model=ModelInfo, dependencies=[Depends(require_api_key)])
def model_reload():
    state.load()
    _need_model()
    m = state.meta
    state.recent.clear()
    return ModelInfo(model_version=m["model_version"], train_end=m["train_end"], target=m["target"], model=m["model"])
