"""バックテスト評価・可視化・SHAP。

主指標は bp-MAE。全ての予測（各モデル・アンサンブル・ベースライン）を、全てが揃う共通期間で比べる。
出力（reports/）：
- eval_overall.csv     共通期間の指標と、B0・B1・B1'・B2 に対する bp-MAE の差・改善率
- eval_by_year.csv     年別 bp-MAE
- eval_holdout.csv     直近1年（チューニングに使っていない期間）
- eval_high_vol.csv    高ボラ期間
- shap_importance.csv
- fig_*.png

usage: python -m src.evaluate
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .baselines import simple_baselines
from .config import load_config, path
from .features import load_processed
from .metrics import bp_error, summary
from .train import RunSpec, features_for, fit, oof_path, train_rows
from .tune import holdout_start

# 参照パレット（dataviz skill references/palette.md、ライトモード）
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

BASELINES = ["B0", "B1", "B1p", "B1_nkd"]


def _style():
    plt.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "sans-serif"],
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "lines.linewidth": 1.5,
    })


def predictions_wide(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """列 = 予測ID（target|model|objective|window|es、ベースライン名）、値 = 予測終値。"""
    df = load_processed(cfg)
    oof = pd.read_parquet(oof_path(cfg))
    oof["run"] = oof["target"] + "|" + oof["model"] + "|" + oof["objective"] + "|" + oof["window"] + "|" + oof["es"]
    wide = oof.pivot_table(index="date", columns="run", values="pred_close")
    b = simple_baselines(df).reindex(wide.index)
    wide = pd.concat([b, wide], axis=1)
    # B2 = Ridge（production の目的変数、expanding）
    b2 = f"{cfg['production']['target']}|ridge|l2|expanding|A"
    if b2 in wide:
        wide["B2"] = wide[b2]
    return wide, df


def table(wide: pd.DataFrame, df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    a = df["actual_close"].reindex(idx)
    rows = []
    for c in wide.columns:
        s = summary(wide.loc[idx, c], a, b1=wide.loc[idx, "B1"], b0=wide.loc[idx, "B0"])
        s["run"] = c
        rows.append(s)
    t = pd.DataFrame(rows).set_index("run")
    for b in ["B0", "B1", "B1p", "B2", "B1_nkd"]:
        if b in t.index:
            t[f"diff_vs_{b}_bp"] = t["bp_mae"] - t.loc[b, "bp_mae"]
            t[f"impr_vs_{b}"] = 1 - t["bp_mae"] / t.loc[b, "bp_mae"]
    t.loc["B1", "dir_acc_vs_b1"] = np.nan
    return t.sort_values("bp_mae")


def plot_timeseries(wide, df, idx, run, dest):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(idx, df["actual_close"].reindex(idx), color=INK, lw=1.6, label="実績")
    ax.plot(idx, wide.loc[idx, run], color=S1, lw=1.3, label=f"モデル（{run}）")
    ax.plot(idx, wide.loc[idx, "B1"], color=S2, lw=1.1, label="B1（NIY 前日終値）", alpha=0.9)
    ax.set_title("日経平均終値：予測と実績（holdout 期間）")
    ax.set_ylabel("円")
    ax.legend(loc="upper left", ncols=3)
    fig.tight_layout()
    fig.savefig(dest, dpi=130)
    plt.close(fig)


def plot_residuals(wide, df, idx, run, dest):
    a = df["actual_close"].reindex(idx)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(-400, 400, 81)
    for c, col, lab in ((run, S1, "モデル"), ("B1", S2, "B1"), ("B2", S3, "B2（Ridge）")):
        if c in wide:
            e = bp_error(wide.loc[idx, c], a)
            ax.hist(e.clip(-400, 400), bins=bins, histtype="step", lw=1.8, color=col,
                    label=f"{lab}  MAE {np.nanmean(np.abs(e)):.1f}bp")
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_title("残差の分布（共通期間）")
    ax.set_xlabel("予測 − 実績（bp、±400 で打ち切り）")
    ax.set_ylabel("日数")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(dest, dpi=130)
    plt.close(fig)


def plot_by_year(by_year: pd.DataFrame, run, dest):
    cols = [(run, S1, "モデル"), ("B1", S2, "B1"), ("B2", S3, "B2（Ridge）")]
    cols = [c for c in cols if c[0] in by_year.columns]
    years = by_year.index.astype(str)
    x = np.arange(len(years))
    w = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, (c, col, lab) in enumerate(cols):
        ax.bar(x + (i - (len(cols) - 1) / 2) * w, by_year[c], width=w - 0.03, color=col, label=lab,
               edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, years)
    ax.set_ylabel("bp-MAE")
    ax.set_title("年別 bp-MAE（低いほど良い）")
    ax.legend(loc="upper left", ncols=3)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(dest, dpi=130)
    plt.close(fig)


def shap_report(cfg, df, ho_start, rep):
    """production 設定の LightGBM を holdout より前のデータで学習し、holdout 上の SHAP を出す。"""
    import shap

    pr = cfg["production"]
    model = pr["model"] if pr["model"] in ("lgb", "xgb", "cat") else "lgb"
    data = train_rows(df, pr["target"])
    feats = features_for(model, data, cfg)

    def fit_before(m):
        tr = data[data.index < m]
        cut = m - pd.DateOffset(months=cfg["walk_forward"]["valid_months"])
        a, v = tr[tr.index < cut], tr[tr.index >= cut]
        return fit(model, pr["objective"], a[feats], a[pr["target"]].values * 1e4,
                   v[feats], v[pr["target"]].values * 1e4, cfg=cfg)

    f = fit_before(ho_start)
    ho = data[data.index >= ho_start]
    ex = shap.TreeExplainer(f.model)
    sv = ex.shap_values(ho[feats])
    imp = pd.Series(np.abs(sv).mean(axis=0), index=feats).sort_values(ascending=False)
    imp.rename("mean_abs_shap_bp").to_csv(rep / "shap_importance.csv", encoding="utf-8-sig")
    top = imp.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top.values, color=S1, height=0.7)
    ax.set_title(f"SHAP 重要度（holdout、{model} {pr['target']}）")
    ax.set_xlabel("mean |SHAP|（bp）")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(rep / "fig_shap_importance.png", dpi=130)
    plt.close(fig)
    # 特定日の waterfall：その日の月の初めより前のデータだけで学習したモデルで説明する
    for d in cfg["evaluation"]["waterfall_dates"]:
        d = pd.Timestamp(d)
        if d not in data.index:
            continue
        fm = fit_before(d.to_period("M").to_timestamp())
        e = shap.TreeExplainer(fm.model)
        exp = e(data.loc[[d], feats])
        plt.figure()
        shap.plots.waterfall(exp[0], max_display=15, show=False)
        plt.title(f"{d.date()} の予測の内訳（{pr['target']}, bp）", loc="left")
        plt.tight_layout()
        plt.savefig(rep / f"fig_waterfall_{d:%Y%m%d}.png", dpi=130)
        plt.close("all")
    return imp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-shap", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config()
    _style()
    rep = path("reports", cfg)
    wide, df = predictions_wide(cfg)
    a = df["actual_close"].reindex(wide.index)
    wide = wide[a.notna()]
    ok = wide.notna().all(axis=1)
    common = wide.index[ok & (wide.index >= pd.Timestamp(cfg["walk_forward"]["compare_from"]))]
    print(f"共通期間：{common.min().date()} 〜 {common.max().date()}（{len(common)}日）")

    overall = table(wide, df, common)
    overall.to_csv(rep / "eval_overall.csv", encoding="utf-8-sig")

    a = df["actual_close"]
    by_year = pd.DataFrame({c: np.abs(bp_error(wide.loc[common, c], a.reindex(common))).groupby(common.year).mean()
                            for c in wide.columns})
    by_year.index.name = "year"
    by_year.round(2).to_csv(rep / "eval_by_year.csv", encoding="utf-8-sig")

    ho_start = holdout_start(df, cfg)
    ho_idx = common[common >= ho_start]
    holdout = table(wide, df, ho_idx)
    holdout.to_csv(rep / "eval_holdout.csv", encoding="utf-8-sig")

    hv = []
    for p in cfg["evaluation"]["high_vol_periods"]:
        idx = common[(common >= p["start"]) & (common <= p["end"])]
        if len(idx):
            t = table(wide, df, idx)[["n", "bp_mae", "diff_vs_B1_bp", "impr_vs_B1"]]
            t.insert(0, "period", p["name"])
            hv.append(t)
    if hv:
        pd.concat(hv).to_csv(rep / "eval_high_vol.csv", encoding="utf-8-sig")

    pr = cfg["production"]
    run = f"{pr['target']}|{pr['model']}|{pr['objective']}|expanding|{pr['es']}"
    if run not in wide:
        run = overall.index[0]
    plot_timeseries(wide, df, ho_idx, run, rep / "fig_timeseries_holdout.png")
    plot_residuals(wide, df, common, run, rep / "fig_residuals.png")
    plot_by_year(by_year, run, rep / "fig_by_year.png")

    pd.set_option("display.width", 200)
    cols = ["n", "bp_mae", "yen_mae", "bp_rmse", "dir_acc_vs_b1", "diff_vs_B1_bp", "impr_vs_B1", "impr_vs_B2"]
    print("\n=== 共通期間（bp-MAE の小さい順、上位15＋ベースライン）===")
    show = overall[cols]
    print(pd.concat([show.head(15), show.loc[[b for b in BASELINES + ["B2"] if b in show.index]]])
          .drop_duplicates().round(3).to_string())
    print(f"\n=== holdout（{ho_start.date()}〜）===")
    hs = holdout[cols]
    print(pd.concat([hs.head(10), hs.loc[[b for b in BASELINES + ["B2"] if b in hs.index]]])
          .drop_duplicates().round(3).to_string())

    thr = cfg["evaluation"]["leak_warning_improvement"]
    sus = overall[overall["impr_vs_B1"] >= thr]
    if len(sus):
        print(f"\n[警告] B1 比の改善率が {thr:.0%} 以上の予測があります。リークを疑ってください：{list(sus.index)}")

    if not args.no_shap:
        imp = shap_report(cfg, df, ho_start, rep)
        print("\nSHAP 重要度 上位10:\n", imp.head(10).round(2).to_string())
    print(f"\n保存しました: {rep}")


if __name__ == "__main__":
    main()
