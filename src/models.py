"""モデルの学習・予測（LightGBM / XGBoost / CatBoost / Ridge）。目的変数は bp 単位。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, load_config


def lgb_params(cfg: dict) -> dict:
    p = dict(cfg["params"]["lgb"])
    f = ROOT / cfg["tune"]["params_file"]
    if f.exists():
        p.update(json.loads(f.read_text(encoding="utf-8")))
    return p


class Fitted:
    """学習済みモデルの薄いラッパー（推論・保存用）。"""

    def __init__(self, name: str, model, features: list[str], best_iter: int | None):
        self.name, self.model, self.features, self.best_iter = name, model, features, best_iter

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.features]
        if self.name == "lgb":
            return self.model.predict(X, num_iteration=self.best_iter)
        if self.name == "xgb":
            import xgboost as xgb
            return self.model.predict(xgb.DMatrix(X), iteration_range=(0, self.best_iter))
        return np.asarray(self.model.predict(X))


def fit(name: str, objective: str, Xtr: pd.DataFrame, ytr: np.ndarray,
        Xva: pd.DataFrame | None = None, yva: np.ndarray | None = None,
        n_rounds: int | None = None, cfg: dict | None = None, params: dict | None = None,
        threads: int = 1) -> Fitted:
    """Xva があれば early stopping（方式A）、なければ n_rounds で固定（方式B）。"""
    cfg = cfg or load_config()
    wf = cfg["walk_forward"]
    seed = cfg["seed"]
    delta = cfg["huber_delta_bp"]
    feats = list(Xtr.columns)
    es = Xva is not None
    max_rounds = wf["max_rounds"] if es or n_rounds is None else int(n_rounds)

    if name == "lgb":
        import lightgbm as lgb
        p = dict(params or lgb_params(cfg))
        p.update(objective="regression_l1" if objective == "l1" else "huber", metric="l1",
                 seed=seed, num_threads=threads, verbose=-1, deterministic=True, force_row_wise=True)
        if objective == "huber":
            p["alpha"] = delta
        dtr = lgb.Dataset(Xtr, ytr, free_raw_data=False)
        if es:
            dva = lgb.Dataset(Xva, yva, reference=dtr)
            m = lgb.train(p, dtr, num_boost_round=max_rounds, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(wf["early_stopping_rounds"], verbose=False)])
            return Fitted(name, m, feats, int(m.best_iteration) or max_rounds)
        m = lgb.train(p, dtr, num_boost_round=max_rounds)
        return Fitted(name, m, feats, max_rounds)

    if name == "xgb":
        import xgboost as xgb
        p = dict(params or cfg["params"]["xgb"])
        p.update(objective="reg:absoluteerror" if objective == "l1" else "reg:pseudohubererror",
                 eval_metric="mae", seed=seed, nthread=threads, tree_method="hist")
        if objective == "huber":
            p["huber_slope"] = delta
        dtr = xgb.DMatrix(Xtr, ytr)
        if es:
            dva = xgb.DMatrix(Xva, yva)
            m = xgb.train(p, dtr, num_boost_round=max_rounds, evals=[(dva, "va")],
                          early_stopping_rounds=wf["early_stopping_rounds"], verbose_eval=False)
            return Fitted(name, m, feats, int(m.best_iteration) + 1)
        m = xgb.train(p, dtr, num_boost_round=max_rounds)
        return Fitted(name, m, feats, max_rounds)

    if name == "cat":
        from catboost import CatBoostRegressor
        p = dict(params or cfg["params"]["cat"])
        loss = "MAE" if objective == "l1" else f"Huber:delta={delta}"
        m = CatBoostRegressor(loss_function=loss, eval_metric="MAE", iterations=max_rounds,
                              random_seed=seed, thread_count=threads, verbose=False,
                              bootstrap_type="Bernoulli", allow_writing_files=False, **p)
        if es:
            m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=wf["early_stopping_rounds"],
                  use_best_model=True)
            return Fitted(name, m, feats, int(m.get_best_iteration()) + 1)
        m.fit(Xtr, ytr)
        return Fitted(name, m, feats, max_rounds)

    if name == "ridge":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          RidgeCV(alphas=np.logspace(-2, 4, 25)))
        X = Xtr if Xva is None else pd.concat([Xtr, Xva])
        y = ytr if yva is None else np.concatenate([ytr, yva])
        m.fit(X, y)
        return Fitted(name, m, feats, None)

    raise ValueError(name)
