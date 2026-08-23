"""Pick-a-Pic v2 preference dataset population.

Default downloads the metadata-only revision (no image pairs). Pass
``--with-images`` for real DPO image training; the images stay in the
HuggingFace cache and are never committed to the repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from vrl.scripts.data.common import default_cache_dir, emit

COMMAND_NAME = "pickapic"


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser(COMMAND_NAME)
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--with-images", action="store_true")
    parser.set_defaults(func=_cmd_pickapic)


def _cmd_pickapic(args: argparse.Namespace) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("datasets is required to populate Pick-a-Pic") from exc

    # The original yuvalkirstain/pickapic_v2 (and *_no_images) repos were
    # delisted from the Hub (404 as of 2026-06; recorded in
    # presets/dataset/pickapic_v1.yaml). The config's data.dataset_name is the
    # source of truth; this default only backstops direct CLI invocations.
    dataset_name = args.dataset_name or "pickapic-anonymous/pickapic_v1"
    dataset = load_dataset(
        dataset_name,
        split=args.split,
        cache_dir=str(args.cache_dir.expanduser()),
    )
    emit(
        {
            "dataset": "pickapic",
            "dataset_name": dataset_name,
            "split": args.split,
            "cache_dir": str(args.cache_dir.expanduser().resolve()),
            "rows": len(dataset),
            "with_images": bool(args.with_images),
        },
    )


def image_setup_argv(dataset_name: str | None = None) -> tuple[str, ...]:
    """Setup command for the bootstrap report; dataset identity comes from the
    experiment config so the emitted command survives Hub delistings."""

    argv: tuple[str, ...] = (COMMAND_NAME, "--with-images")
    if dataset_name:
        argv += ("--dataset-name", str(dataset_name))
    return argv


__all__ = ["image_setup_argv", "register"]
