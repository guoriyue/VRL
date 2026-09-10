"""Shared helpers for the per-dataset population scripts.

Concrete, dependency-free utilities only. Each dataset lives in its own module
(pickapic.py, danbooru/, video_world/, bootstrap.py); this module just holds
the path helpers they all need. File writers live in :mod:`vrl.utils.json_files`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Single source of truth for repo/data-root resolution is vrl.utils.artifacts
# (it owns DATA_ROOT_ENV and is also used off the script path);
# re-export here so the dataset scripts keep importing from common.
from vrl.utils.artifacts import default_data_root, repo_root


def default_cache_dir() -> Path:
    return (repo_root() / "data" / "cache" / "hf").resolve()


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def dedupe_text(parts: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        text = re.sub(r"\s+", " ", str(part).strip())
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


__all__ = [
    "dedupe_text",
    "default_cache_dir",
    "default_data_root",
    "emit",
    "repo_root",
]
