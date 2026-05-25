"""Shared helpers for the per-dataset population scripts.

Concrete, dependency-free utilities only. Each dataset lives in its own script
(pickapic.py, danbooru.py, video_world.py, bootstrap.py); this module just holds
the path/IO helpers they all need so no logic is duplicated.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_data_root() -> Path:
    value = os.environ.get("VRL_DATA_ROOT", "").strip()
    if value:
        return Path(value).expanduser().resolve()
    return (repo_root() / "data" / "external").resolve()


def default_cache_dir() -> Path:
    return (repo_root() / "data" / "cache" / "hf").resolve()


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "default_cache_dir",
    "default_data_root",
    "emit",
    "repo_root",
    "write_jsonl",
    "write_report",
]
