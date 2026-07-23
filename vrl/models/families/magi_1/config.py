"""Lightweight public config schema for the MAGI-1 model family."""

from __future__ import annotations

from typing import Any

from vrl.config.model_schema import ModelSection


class Magi1ModelSection(ModelSection):
    """MAGI-1 isolated runtime and checkpoint component paths."""

    checkpoint_path: Any = None
    config_path: Any = None
    python_executable: Any = None
    source_path: Any = None
    source_revision: Any = None
    t5_pretrained_path: Any = None
    timeout_seconds: Any = None
    vae_pretrained_path: Any = None


__all__ = ["Magi1ModelSection"]
