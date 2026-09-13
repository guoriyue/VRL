"""Locate the ruff executable shipped in the locked wheel.

rules_python extracts a wheel's scripts to ``<repo>/bin`` beside
``<repo>/site-packages``, one level above where ruff's own finder looks.
"""

from pathlib import Path

import ruff
from ruff.__main__ import find_ruff_bin


def ruff_bin() -> str:
    candidate = Path(ruff.__file__).resolve().parents[2] / "bin" / "ruff"
    if candidate.is_file():
        return str(candidate)
    return find_ruff_bin()
