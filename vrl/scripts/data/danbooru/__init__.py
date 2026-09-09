"""CLI composition for Danbooru-derived dataset builders.

Implementation ownership lives in the submodules (``metadata`` parsing,
``anatomy``/``safety`` prompts, image ``assets``, ``config`` constants,
``cli`` composition). This package exports only the setup-CLI surface;
everything else is imported from its owning submodule — the former
flat-module re-export facade had no external callers left.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vrl.scripts.data.danbooru.cli import main as _main
from vrl.scripts.data.danbooru.cli import manifest_setup_hints
from vrl.scripts.data.danbooru.cli import register as _register


def register(subparsers: Any) -> None:
    """Register the Danbooru commands on a setup CLI parser."""

    _register(subparsers)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the Danbooru CLI through the package composition."""

    _main(argv)


__all__ = [
    "main",
    "manifest_setup_hints",
    "register",
]
