"""本番モデルの再学習（データが増えたときに実行する）。

最新のデータを取得 → 特徴量を作り直す → 本番モデル（config.yaml の production）を全データで学習して models/ に保存。
バックテスト（全パターンの walk-forward）はやり直さない。学習回数とスタッキングの係数は、
バックテストの結果があればそこから、なければ前回のモデルに保存した値を引き継ぐ。

usage: python -m src.retrain [--no-fetch]
"""
from __future__ import annotations

import argparse
import sys

from filelock import Timeout

from .config import load_config
from .features import build, processed_path
from .fetch_data import update_all
from .inference import run_lock
from .train import _previous_meta, train_final


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-fetch", action="store_true", help="データを取得せず、手元のデータだけで学習する")
    args = ap.parse_args(argv)
    cfg = load_config()
    lock = run_lock(cfg, cfg["api"]["run_lock_timeout_sec"])
    try:
        lock.acquire()
    except Timeout:
        print("[error] 予測処理が実行中です。終わってから実行してください。", file=sys.stderr)
        return 3
    try:
        before = _previous_meta(cfg)
        if not args.no_fetch:
            print("1/3 データを取得しています…")
            update_all(cfg, verbose=False)
        print("2/3 特徴量を作っています…")
        df = build(cfg=cfg)
        df.to_parquet(processed_path(cfg))
        print("3/3 本番モデルを学習しています…")
        meta = train_final(cfg, df)
    finally:
        lock.release()
    print()
    if before:
        print(f"前のモデル: {before.get('model_version')}（学習期間 〜{before.get('train_end')}）")
    print(f"新しいモデル: {meta['model_version']}（学習期間 〜{meta['train_end']}）")
    print("API を起動しているなら、POST /model/reload で読み込み直してください。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
