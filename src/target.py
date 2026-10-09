"""目的変数の生成と価格への逆変換。

T1: log(N225_t / N225_{t-1})            基準価格 = N225_{t-1}
T2: log(N225_t / NIY_{t-1(米国日付)})    基準価格 = NIY_close（予測時点で確定済みの最新値）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_COL = {"T1": "n225_close", "T2": "niy_close"}


def base_price(aligned: pd.DataFrame, target: str) -> pd.Series:
    return aligned[BASE_COL[target]]


def make_target(actual_close: pd.Series, aligned: pd.DataFrame, target: str) -> pd.Series:
    """actual_close は予測対象日 t の N225 終値（特徴量には絶対に入れない）。"""
    return np.log(actual_close / base_price(aligned, target)).rename(target)


def to_price(pred_logret, base) -> np.ndarray | pd.Series:
    return base * np.exp(pred_logret)


def to_logret(price, base):
    return np.log(price / base)
