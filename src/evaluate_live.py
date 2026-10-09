"""ライブ予測ログと実績の照合。

対象は mode=live の行だけ。同じ日付が複数あれば、created_at が JST 8:00 より前の最後の行を使う。
出力：reports/live_eval.csv（日別）と、モデル・B1・B0 の指標の要約。

usage: python -m src.evaluate_live [--fetch]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import load_config, path
from .fetch_data import load_raw, update_all
from .metrics import bp_error, summary
from .storage import read_log, select_live


def live_table(cfg: dict) -> pd.DataFrame:
    log = read_log(cfg)
    live = select_live(log, cfg)
    if live.empty:
        return live
    n225 = load_raw(cfg["tickers"]["n225"], cfg)
    live = live.copy()
    live["date_ts"] = pd.to_datetime(live["date"])
    live["actual_close"] = n225["close"].reindex(live["date_ts"]).values
    live["is_fallback"] = live["model_version"].str.startswith("fallback")
    live["err_bp"] = bp_error(live["pred_close"].astype(float), live["actual_close"])
    live["b1_err_bp"] = bp_error(live["b1"].astype(float), live["actual_close"])
    live["b0_err_bp"] = bp_error(live["b0"].astype(float), live["actual_close"])
    return live.drop(columns=["created_at_ts", "date_ts"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true", help="先に ^N225 の実績を取得する")
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.fetch:
        update_all(cfg, tickers=[cfg["tickers"]["n225"]], verbose=False)
    t = live_table(cfg)
    dest = path("reports", cfg) / "live_eval.csv"
    if t.empty:
        print("ライブ予測（mode=live）のログがまだありません。")
        return
    t.to_csv(dest, index=False, encoding="utf-8-sig")
    done = t[t["actual_close"].notna()]
    print(f"ライブ予測 {len(t)}日（実績確定 {len(done)}日、うちフォールバック {int(done['is_fallback'].sum())}日）")
    if len(done):
        rows = {}
        for name, col in (("model", "pred_close"), ("B1", "b1"), ("B0", "b0")):
            rows[name] = summary(done[col].astype(float), done["actual_close"], b1=done["b1"].astype(float),
                                 b0=done["b0"].astype(float))
        s = pd.DataFrame(rows).T
        s.loc["B1", "dir_acc_vs_b1"] = np.nan
        print(s[["n", "bp_mae", "yen_mae", "bp_rmse", "dir_acc_vs_b1"]].round(3).to_string())
    print(f"保存しました: {dest}")


if __name__ == "__main__":
    main()
