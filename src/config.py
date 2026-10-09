"""設定ファイルの読み込みとパス解決。"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[1]
JST = ZoneInfo("Asia/Tokyo")
UTC = ZoneInfo("UTC")


def load_config(path: str | None = None) -> dict:
    """config.yaml を読む。環境変数 NIKKEI_CONFIG があればそのファイルを使う（テスト用）。"""
    return _load(str(path or os.environ.get("NIKKEI_CONFIG") or ROOT / "config.yaml"))


@lru_cache(maxsize=8)
def _load(p: str) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str, cfg: dict | None = None) -> Path:
    """paths.* は ROOT からの相対パス（絶対パスも可）。"""
    cfg = cfg or load_config()
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def config_hash(cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]


def git_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def safe_name(ticker: str) -> str:
    return ticker.replace("^", "IDX_").replace("=", "_").replace(".", "_")
