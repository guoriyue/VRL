"""Shared helpers for the per-dataset population scripts.

Concrete, dependency-free utilities only. Each dataset lives in its own module
(video_world/, bootstrap.py); this module just holds
the path helpers they all need. File writers live in :mod:`vrl.utils.json_files`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vrl.utils.paths import repo_root


def default_cache_dir() -> Path:
    return (repo_root() / "data" / "cache" / "hf").resolve()


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


__all__ = [
    "default_cache_dir",
    "emit",
]
