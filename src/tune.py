"""LightGBM のハイパーパラメータを Optuna でチューニングする。

- 目的：walk-forward（四半期ごとに再学習、early stopping は方式A）の bp-MAE
- 直近 holdout_months（初期値12か月）はチューニングに使わない
- 結果：reports/lgb_best_params.json（train.py が自動で読み込む）、reports/tune_trials.csv

usage: python -m src.tune [--n-trials 50] [--n-jobs 8]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import optuna
import pandas as pd

from .config import ROOT, load_config, path
from .features import load_processed
from .models import fit
from .train import RunSpec, features_for, fold_months, train_rows, window_slice


def holdout_start(df: pd.DataFrame, cfg: dict) -> pd.Timestamp:
    last = df.dropna(subset=["T1"]).index.max()
    return (last - pd.DateOffset(months=cfg["walk_forward"]["holdout_months"])).to_period("M").to_timestamp() \
        + pd.DateOffset(months=1)


def _fold_abs_err(data, feats, spec, m, refit, end, cfg, params) -> np.ndarray:
    """1 fold（方式A）の |誤差|（bp）。"""
    te = data[(data.index >= m) & (data.index < min(m + pd.DateOffset(months=refit), end))]
    if te.empty:
        return np.array([])
    tr = window_slice(data, m, spec.window, cfg)
    cut = m - pd.DateOffset(months=cfg["walk_forward"]["valid_months"])
    a, v = tr[tr.index < cut], tr[tr.index >= cut]
    y = spec.target
    f = fit("lgb", spec.objective, a[feats], a[y].values * 1e4, v[feats], v[y].values * 1e4, cfg=cfg, params=params)
    return np.abs(f.predict(te) - te[y].values * 1e4)


def main(argv=None):
    from joblib import Parallel, delayed

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-trials", type=int)
    ap.add_argument("--n-jobs", type=int, default=12, help="fold を並列に学習するプロセス数")
    ap.add_argument("--fresh", action="store_true", help="保存済みの試行を捨てて最初からやり直す")
    args = ap.parse_args(argv)
    cfg = load_config()
    tc = cfg["tune"]
    df = load_processed(cfg)
    end = holdout_start(df, cfg)
    spec = RunSpec(tc["target"], "lgb", tc["objective"], tc["window"])
    data = train_rows(df, spec.target)
    feats = features_for("lgb", data, cfg)
    months = [m for m in fold_months(data, spec.window, cfg, tc["refit_months"], end) if m < end]
    print(f"チューニング期間：〜{end.date()}（以降は holdout）  {spec.key}  四半期ごとに再学習（{len(months)} fold）")
    pool = Parallel(n_jobs=args.n_jobs, backend="loky")

    def objective(trial: optuna.Trial) -> float:
        p = {
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
        }
        errs = pool(delayed(_fold_abs_err)(data, feats, spec, m, tc["refit_months"], end, cfg, p) for m in months)
        return float(np.mean(np.concatenate(errs)))

    # 試行は SQLite に保存し、中断しても続きから再開できるようにする
    db = path("processed", cfg) / "optuna.db"
    if args.fresh and db.exists():
        db.unlink()
    sampler = optuna.samplers.TPESampler(seed=cfg["seed"])
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name="lgb",
                                storage=f"sqlite:///{db.as_posix()}", load_if_exists=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, (args.n_trials or tc["n_trials"]) - done)
    print(f"完了済み {done} 試行、残り {remaining} 試行")
    study.optimize(objective, n_trials=remaining, n_jobs=1,
                   callbacks=[lambda s, t: print(f"  trial {t.number:>3}: {t.value:.3f}  best {s.best_value:.3f}",
                                                 flush=True)])
    best = dict(study.best_params, bagging_freq=1)
    dest = ROOT / tc["params_file"]
    dest.write_text(json.dumps(best, indent=2), encoding="utf-8")
    study.trials_dataframe().to_csv(path("reports", cfg) / "tune_trials.csv", index=False, encoding="utf-8-sig")
    print(f"best bp-MAE {study.best_value:.3f}  保存しました: {dest}")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
