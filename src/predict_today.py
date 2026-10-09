"""当日朝の推論（CLI）。差分取得 → 予測 → predictions/log.csv 追記 → latest.json 保存。

東証の休場日なら何もせずに終了する。JST 7:30 の自動実行は scripts/run_daily.ps1 から。

usage: python -m src.predict_today [--date YYYY-MM-DD] [--no-fetch]
"""
from __future__ import annotations

import argparse
import sys

from .inference import Busy, FutureDate, MarketClosed, run_pipeline


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="予測対象日（省略時は今日）。過去日付は backfill として記録する")
    ap.add_argument("--no-fetch", action="store_true", help="データを取得せず、キャッシュだけで予測する")
    args = ap.parse_args(argv)
    try:
        r = run_pipeline(args.date, fetch=not args.no_fetch)
    except MarketClosed as e:
        print(f"{e}。何もせずに終了します。")
        return 0
    except FutureDate as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2
    except Busy as e:
        print(f"[error] {e}", file=sys.stderr)
        return 3

    print(f"予測対象日      : {r.date}（mode={r.mode}）")
    print(f"予測終値        : {r.pred_close:,.2f} 円（前日比 {r.pred_return_pct:+.2f}%）")
    print(f"前日終値（B0）  : {r.b0:,.2f} 円")
    if r.b1 is not None:
        print(f"先物（B1）      : {r.b1:,.2f} 円  予測との差 {r.diff_vs_b1_yen:+,.2f} 円（{r.diff_vs_b1_bp:+.1f}bp）")
    print(f"先物データ確定  : {r.data_asof}")
    print(f"モデル          : {r.model_version}{'（フォールバック）' if r.is_fallback else ''}")
    for w in r.warnings:
        print(f"[warn] {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
