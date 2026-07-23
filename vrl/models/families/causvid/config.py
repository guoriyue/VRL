"""Lightweight public config schema for the CausVid model family."""

from __future__ import annotations

from typing import Any

from vrl.config.model_schema import ModelSection


class CausVidModelSection(ModelSection):
    """CausVid pinned source, Wan base, and released checkpoint keys."""

    accept_noncommercial_license: bool = False
    base_model_path: Any = None
    base_model_revision: Any = None
    causvid_source_path: Any = None
    causvid_source_revision: Any = None
    checkpoint_file: Any = None
    checkpoint_sha256: Any = None


__all__ = ["CausVidModelSection"]
