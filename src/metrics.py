"""評価指標。主指標は bp-MAE = mean(|log(pred) - log(actual)|) × 10000。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def bp_error(pred_close, actual_close):
    return (np.log(pred_close) - np.log(actual_close)) * 1e4


def bp_mae(pred_close, actual_close) -> float:
    return float(np.nanmean(np.abs(bp_error(pred_close, actual_close))))


def _dir_acc(pred, actual, ref) -> float:
    p, a = np.sign(np.asarray(pred) - np.asarray(ref)), np.sign(np.asarray(actual) - np.asarray(ref))
    m = (p != 0) & (a != 0) & ~np.isnan(p) & ~np.isnan(a)
    return float((p[m] == a[m]).mean()) if m.any() else float("nan")


def summary(pred, actual, b1=None, b0=None) -> dict:
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    m = ~np.isnan(pred) & ~np.isnan(actual)
    pred, actual = pred[m], actual[m]
    e_bp = bp_error(pred, actual)
    e_yen = pred - actual
    out = {
        "n": int(m.sum()),
        "bp_mae": float(np.mean(np.abs(e_bp))) if len(e_bp) else np.nan,
        "bp_rmse": float(np.sqrt(np.mean(e_bp ** 2))) if len(e_bp) else np.nan,
        "yen_mae": float(np.mean(np.abs(e_yen))) if len(e_yen) else np.nan,
        "yen_rmse": float(np.sqrt(np.mean(e_yen ** 2))) if len(e_yen) else np.nan,
        "mape_pct": float(np.mean(np.abs(e_yen / actual)) * 100) if len(e_yen) else np.nan,
    }
    if b1 is not None:
        out["dir_acc_vs_b1"] = _dir_acc(pred, actual, np.asarray(b1, float)[m])
    if b0 is not None:
        out["dir_acc_vs_prev"] = _dir_acc(pred, actual, np.asarray(b0, float)[m])
    return out
