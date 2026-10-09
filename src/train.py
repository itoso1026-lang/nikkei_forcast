"""Walk-forward 学習と、本番用モデルの学習。

- テスト：1か月ずつ前進し、毎月再学習する（--refit-months で変更可。tune.py は 3）
- 学習ウィンドウ：expanding（最初は3年）/ rolling（5年）
- early stopping
    A：学習期間の末尾6か月を検証データにし（時系列順）、残りで学習
    B：それまでの fold の A の best_iteration の中央値で固定し、学習期間全体で学習
- 最終予測：GBDT 3種＋Ridge の単純平均（avg）と、OOF 予測による Ridge スタッキング（stack）

出力：data/processed/oof.parquet

usage:
  python -m src.train                       # config の grid をすべて実行
  python -m src.train --models lgb --targets T2
  python -m src.train --final               # production 設定のモデルを全データで学習して models/ に保存
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import config_hash, git_hash, load_config, path
from .features import feature_columns, load_processed
from .models import fit
from .target import BASE_COL, to_price


@dataclass(frozen=True)
class RunSpec:
    target: str
    model: str
    objective: str
    window: str

    @property
    def key(self) -> str:
        return f"{self.target}|{self.model}|{self.objective}|{self.window}"


def train_rows(df: pd.DataFrame, target: str) -> pd.DataFrame:
    return df[df[target].notna() & df[BASE_COL[target]].notna()]


def fold_months(df: pd.DataFrame, window: str, cfg: dict, refit_months: int = 1,
                end: pd.Timestamp | None = None) -> list[pd.Timestamp]:
    wf = cfg["walk_forward"]
    start = df.index.min()
    years = wf["expanding_min_years"] if window == "expanding" else wf["rolling_years"]
    first = (start + pd.DateOffset(years=years)).to_period("M").to_timestamp()
    if first < start + pd.DateOffset(years=years):
        first = first + pd.DateOffset(months=1)
    last = (end or df.index.max()).to_period("M").to_timestamp()
    return list(pd.date_range(first, last, freq=f"{refit_months}MS"))


def window_slice(df: pd.DataFrame, m: pd.Timestamp, window: str, cfg: dict) -> pd.DataFrame:
    tr = df[df.index < m]
    if window == "rolling":
        tr = tr[tr.index >= m - pd.DateOffset(years=cfg["walk_forward"]["rolling_years"])]
    return tr


def features_for(model: str, df: pd.DataFrame, cfg: dict) -> list[str]:
    return list(cfg["features"]["ridge_features"]) if model == "ridge" else feature_columns(df)


def run_walk_forward(spec: RunSpec, df: pd.DataFrame, cfg: dict, refit_months: int = 1,
                     end: pd.Timestamp | None = None, params: dict | None = None,
                     es_modes: tuple[str, ...] = ("A", "B")) -> pd.DataFrame:
    """1つの (target, model, objective, window) について walk-forward を回し、OOF 予測を返す。"""
    data = train_rows(df, spec.target)
    feats = features_for(spec.model, data, cfg)
    y_all = data[spec.target] * 1e4
    vm = cfg["walk_forward"]["valid_months"]
    best_iters: list[int] = []
    out = []
    for m in fold_months(data, spec.window, cfg, refit_months, end):
        m_end = m + pd.DateOffset(months=refit_months)
        if end is not None:
            m_end = min(m_end, end)
        te = data[(data.index >= m) & (data.index < m_end)]
        if te.empty:
            continue
        tr = window_slice(data, m, spec.window, cfg)
        if spec.model == "ridge":
            f = fit("ridge", spec.objective, tr[feats], y_all.loc[tr.index].values, cfg=cfg)
            pred = f.predict(te)
            for es in es_modes:
                out.append(pd.DataFrame({"pred_bp": pred, "es": es, "best_iter": np.nan, "fold": m}, index=te.index))
            continue
        cut = m - pd.DateOffset(months=vm)
        fit_part, va = tr[tr.index < cut], tr[tr.index >= cut]
        fa = fit(spec.model, spec.objective, fit_part[feats], y_all.loc[fit_part.index].values,
                 va[feats], y_all.loc[va.index].values, cfg=cfg, params=params)
        best_iters.append(fa.best_iter)
        if "A" in es_modes:
            out.append(pd.DataFrame({"pred_bp": fa.predict(te), "es": "A", "best_iter": fa.best_iter, "fold": m},
                                    index=te.index))
        if "B" in es_modes:
            prev = best_iters[:-1] or best_iters
            n = int(np.median(prev))
            fb = fit(spec.model, spec.objective, tr[feats], y_all.loc[tr.index].values, n_rounds=n, cfg=cfg,
                     params=params)
            out.append(pd.DataFrame({"pred_bp": fb.predict(te), "es": "B", "best_iter": n, "fold": m},
                                    index=te.index))
    res = pd.concat(out)
    res.index.name = "date"
    res = res.reset_index()
    for k, v in (("target", spec.target), ("model", spec.model), ("objective", spec.objective),
                 ("window", spec.window)):
        res[k] = v
    return res


def _run(spec: RunSpec, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    t0 = time.time()
    r = run_walk_forward(spec, df, cfg)
    print(f"  done {spec.key}  {len(r)}行  {time.time() - t0:.0f}s", flush=True)
    return r


def add_ensembles(oof: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """avg（GBDT 3種＋Ridge の単純平均）と stack（OOF による Ridge スタッキング）を追加する。"""
    from sklearn.linear_model import Ridge

    gbdt = [m for m in ("lgb", "xgb", "cat") if m in set(oof["model"])]
    if not gbdt or "ridge" not in set(oof["model"]):
        return oof
    extra = []
    ridge = oof[oof.model == "ridge"]
    for (tg, obj, win, es), g in oof[oof.model.isin(gbdt)].groupby(["target", "objective", "window", "es"]):
        wide = g.pivot_table(index="date", columns="model", values="pred_bp")
        r = ridge[(ridge.target == tg) & (ridge.window == win) & (ridge.es == es)].drop_duplicates("date")
        wide["ridge"] = r.set_index("date")["pred_bp"]
        wide = wide.dropna()
        base = {"target": tg, "objective": obj, "window": win, "es": es}
        avg = wide.mean(axis=1)
        extra.append(pd.DataFrame({"date": avg.index, "pred_bp": avg.values, "model": "avg", **base}))
        # スタッキング：月 m の予測には、m より前の OOF 予測だけで学習した Ridge を使う
        y = oof_target(cfg).reindex(wide.index)[tg] * 1e4
        months = wide.index.to_period("M").unique()
        min_m = cfg["walk_forward"]["stacking_min_months"]
        preds = []
        for i, mp in enumerate(months):
            if i < min_m:
                continue
            ms = mp.to_timestamp()
            hist = wide.index < ms
            cur = wide.index.to_period("M") == mp
            mdl = Ridge(alpha=1.0).fit(wide[hist].values, y[hist].values)
            preds.append(pd.Series(mdl.predict(wide[cur].values), index=wide.index[cur]))
        if preds:
            st = pd.concat(preds)
            extra.append(pd.DataFrame({"date": st.index, "pred_bp": st.values, "model": "stack", **base}))
    return pd.concat([oof] + extra, ignore_index=True)


_TARGETS_CACHE: dict = {}


def oof_target(cfg: dict) -> pd.DataFrame:
    if "df" not in _TARGETS_CACHE:
        _TARGETS_CACHE["df"] = load_processed(cfg)[["T1", "T2"]]
    return _TARGETS_CACHE["df"]


def attach_prices(oof: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    oof = oof.copy()
    base = np.where(oof["target"] == "T1", df["n225_close"].reindex(oof["date"]).values,
                    df["niy_close"].reindex(oof["date"]).values)
    oof["pred_close"] = to_price(oof["pred_bp"].values / 1e4, base)
    oof["actual_close"] = df["actual_close"].reindex(oof["date"]).values
    return oof


def oof_path(cfg: dict):
    return path("processed", cfg) / "oof.parquet"


def _previous_meta(cfg: dict) -> dict:
    p = path("models", cfg) / "production.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _oof(cfg: dict) -> pd.DataFrame | None:
    return pd.read_parquet(oof_path(cfg)) if oof_path(cfg).exists() else None


def _rounds_for(model: str, pr: dict, cfg: dict, data: pd.DataFrame, feats: list[str],
                oof: pd.DataFrame | None, prev: dict) -> tuple[int, str]:
    """方式B のラウンド数。
    ① バックテストの OOF の中央値 → ② config の初期設定 → ③ 前回のモデルの値 → ④ 直近6か月で early stopping。
    """
    if oof is not None:
        o = oof[(oof.target == pr["target"]) & (oof.model == model) & (oof.objective == pr["objective"])
                & (oof.window == "expanding") & (oof.es == "A")]
        if len(o):
            return int(o.groupby("fold")["best_iter"].first().median()), "backtest"
    if (pr.get("rounds") or {}).get(model):
        return int(pr["rounds"][model]), "config"
    prev_rounds = (prev.get("members") or {}).get(model, {}).get("n_rounds")
    if prev_rounds is None and prev.get("model") == model:
        prev_rounds = prev.get("n_rounds")
    if prev_rounds:
        return int(prev_rounds), "previous_model"
    cut = data.index.max() - pd.DateOffset(months=cfg["walk_forward"]["valid_months"])
    a, v = data[data.index <= cut], data[data.index > cut]
    y = pr["target"]
    fa = fit(model, pr["objective"], a[feats], a[y].values * 1e4, v[feats], v[y].values * 1e4, cfg=cfg)
    return int(fa.best_iter), "early_stopping_6m"


def _stack_weights(members: list[str], pr: dict, cfg: dict, df: pd.DataFrame,
                   oof: pd.DataFrame | None, prev: dict) -> tuple[dict, float, str]:
    """スタッキングの係数。
    ① バックテストの OOF 全体で Ridge → ② config の初期設定 → ③ 前回のモデルの値 → ④ 単純平均。
    """
    from sklearn.linear_model import Ridge

    if oof is not None:
        o = oof[(oof.target == pr["target"]) & (oof.model.isin(members)) & (oof.window == "expanding")
                & (oof.es == pr["es"]) & ((oof.objective == pr["objective"]) | (oof.model == "ridge"))]
        wide = o.pivot_table(index="date", columns="model", values="pred_bp").dropna()
        if set(members) <= set(wide.columns) and len(wide):
            wide = wide[members]
            y = df[pr["target"]].reindex(wide.index) * 1e4
            m = Ridge(alpha=1.0).fit(wide.values, y.values)
            return {k: float(c) for k, c in zip(members, m.coef_)}, float(m.intercept_), "backtest"
    sw = pr.get("stack_weights")
    if sw and set(sw["coef"]) == set(members):
        return {k: float(sw["coef"][k]) for k in members}, float(sw["intercept"]), "config"
    st = prev.get("stack")
    if st and set(st["coef"]) == set(members):
        return st["coef"], st["intercept"], "previous_model"
    return {k: 1.0 / len(members) for k in members}, 0.0, "equal_weights"


def train_final(cfg: dict, df: pd.DataFrame) -> dict:
    """production 設定のモデルを、確定済みの全データで学習して models/ に保存する。

    model: stack なら members（lgb, xgb, cat, ridge）をそれぞれ学習し、係数で組み合わせる。
    ラウンド数と係数は、バックテストの結果（data/processed/oof.parquet）があればそこから、
    なければ config.yaml の production.rounds / stack_weights（初期設定）を使う
    （バックテストをやり直さなくても再学習できる）。
    """
    from .models import StackModel

    pr = cfg["production"]
    data = train_rows(df, pr["target"])
    y = data[pr["target"]].values * 1e4
    oof, prev = _oof(cfg), _previous_meta(cfg)
    members = pr.get("members", ["lgb", "xgb", "cat", "ridge"]) if pr["model"] == "stack" else [pr["model"]]

    fitted, info = {}, {}
    for m in members:
        feats = features_for(m, data, cfg)
        if m == "ridge":
            fitted[m] = fit("ridge", "l2", data[feats], y, cfg=cfg)
            info[m] = {"n_rounds": None, "rounds_source": None}
            continue
        n, src = _rounds_for(m, pr, cfg, data, feats, oof, prev)
        fitted[m] = fit(m, pr["objective"], data[feats], y, n_rounds=n, cfg=cfg)
        info[m] = {"n_rounds": n, "rounds_source": src}
        print(f"  {m}: {n} rounds（{src}）", flush=True)

    meta_extra = {}
    if pr["model"] == "stack":
        coef, intercept, wsrc = _stack_weights(members, pr, cfg, df, oof, prev)
        model = StackModel(fitted, coef, intercept)
        meta_extra["stack"] = {"coef": coef, "intercept": intercept, "source": wsrc}
        print(f"  stack: {', '.join(f'{k} {v:.3f}' for k, v in coef.items())}, 切片 {intercept:.2f}bp（{wsrc}）")
    else:
        model = fitted[members[0]]

    train_end = data.index.max()
    gh = git_hash()
    meta = {
        "target": pr["target"], "model": pr["model"], "objective": pr["objective"], "es": pr["es"],
        "n_rounds": info[members[0]]["n_rounds"] if len(members) == 1 else None,
        "members": info, **meta_extra,
        "features": model.features, "train_start": str(data.index.min().date()),
        "train_end": str(train_end.date()), "trained_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
        "config_hash": config_hash(cfg), "git_hash": gh,
        "futures_base": cfg.get("futures", {}).get("base", "niy"),
        "model_version": f"{pr['model']}-{pr['target']}-{train_end:%Y%m%d}-{gh}",
    }
    d = path("models", cfg)
    with open(d / "production.pkl", "wb") as f:
        pickle.dump(model, f)
    (d / "production.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"保存しました: {d / 'production.pkl'}  version={meta['model_version']}  学習期間 〜{meta['train_end']}")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--models", nargs="*", help="lgb xgb cat ridge")
    ap.add_argument("--objectives", nargs="*")
    ap.add_argument("--windows", nargs="*")
    ap.add_argument("--n-jobs", type=int)
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config()
    np.random.seed(cfg["seed"])
    df = load_processed(cfg)

    if args.final:
        train_final(cfg, df)
        return

    g = cfg["grid"]
    targets = args.targets or g["targets"]
    models = args.models or (g["models"] + ["ridge"])
    objectives = args.objectives or g["objectives"]
    windows = args.windows or g["windows"]
    specs = []
    for tg, m, w in itertools.product(targets, models, windows):
        for o in (["l2"] if m == "ridge" else objectives):
            specs.append(RunSpec(tg, m, o, w))
    # 重い CatBoost から先に投入する
    specs.sort(key=lambda s: {"cat": 0, "xgb": 1, "lgb": 2, "ridge": 3}[s.model])
    n_jobs = args.n_jobs or cfg["walk_forward"]["n_jobs"]
    print(f"{len(specs)} 設定を {n_jobs} 並列で実行します")
    from joblib import Parallel, delayed
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_run)(s, df, cfg) for s in specs)
    new = pd.concat(results, ignore_index=True)

    p = oof_path(cfg)
    if p.exists():  # 今回実行しなかった設定の結果は残す
        old = pd.read_parquet(p)
        old = old[~old["model"].isin(["avg", "stack"])]
        keys = set(new[["target", "model", "objective", "window"]].drop_duplicates().itertuples(index=False))
        mask = [tuple(r) not in keys for r in old[["target", "model", "objective", "window"]].itertuples(index=False)]
        new = pd.concat([old[mask], new.drop(columns=[c for c in ("pred_close", "actual_close") if c in new])],
                        ignore_index=True)
    new = new[[c for c in new.columns if c not in ("pred_close", "actual_close")]]
    oof = add_ensembles(new, cfg)
    oof = attach_prices(oof, df)
    oof.to_parquet(p)
    print(f"保存しました: {p}  {len(oof)}行  合計 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
