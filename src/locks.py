"""ファイル書き込みの排他制御（CLI と API の同時実行対策）。"""
from __future__ import annotations

from pathlib import Path

from filelock import FileLock


def lock_for(target: Path, timeout: float = 60) -> FileLock:
    target = Path(target)
    return FileLock(str(target) + ".lock", timeout=timeout)


def atomic_write_bytes(target: Path, data: bytes) -> None:
    """一時ファイルに書いてから置き換える（書き込み途中のファイルを読ませない）。"""
    target = Path(target)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(target)
