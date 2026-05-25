"""Shared helpers for the per-dataset population scripts.

Concrete, dependency-free utilities only. Each dataset lives in its own script
(pickapic.py, danbooru.py, video_world.py, bootstrap.py); this module just holds
the path/IO helpers they all need so no logic is duplicated.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
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


def write_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    sort_keys: bool = True,
) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=sort_keys) + "\n")
            count += 1
    return count


def dedupe_text(parts: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        text = re.sub(r"\s+", " ", str(part).strip())
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "dedupe_text",
    "default_cache_dir",
    "default_data_root",
    "emit",
    "repo_root",
    "write_jsonl",
    "write_report",
]
