"""ベースライン（価格）。B2（Ridge）は walk-forward で学習するので train.py の model='ridge'。

B0     : N225_{t-1}
B1     : NIY_close_{t-1}（config futures.base に従う）
B1p    : B1 × exp(mean(T2 の実績, 直近20営業日))  … B1'
B1_nkd : NKD_close_{t-1}（参考）
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simple_baselines(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["B0"] = df["n225_close"]
    out["B1"] = df["niy_close"]
    out["B1p"] = df["niy_close"] * np.exp(df["basis_ma20"])
    out["B1_nkd"] = df["nkd_close"]
    return out
