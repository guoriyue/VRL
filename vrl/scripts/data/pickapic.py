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

    dataset_name = args.dataset_name or (
        "yuvalkirstain/pickapic_v2" if args.with_images else "yuvalkirstain/pickapic_v2_no_images"
    )
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


def image_setup_argv() -> tuple[str, ...]:
    return (COMMAND_NAME, "--with-images")


__all__ = ["image_setup_argv", "register"]
