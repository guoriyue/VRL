"""CLI composition for Danbooru dataset builders."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from vrl.scripts.data.common import default_data_root, emit
from vrl.scripts.data.danbooru.anatomy import build_anatomy_prompts
from vrl.scripts.data.danbooru.assets import (
    build_positive_images,
    download_danbooru_images,
    select_positive_targets,
)
from vrl.scripts.data.danbooru.config import (
    ANIME_FETCH_IMAGES_COMMAND,
    ANIME_POSITIVES_COMMAND,
    ANIME_PROMPTS_COMMAND,
    ANIME_SAFETY_PROMPTS_COMMAND,
    DANBOORU_METADATA_FILE,
    DANBOORU_REPO_ID,
    POSITIVE_IMAGE_LIMIT,
    POSITIVE_IMAGE_MIN_SCORE,
    POSITIVE_IMAGE_SOURCE,
    POSITIVE_IMAGES_OUTPUT,
)
from vrl.scripts.data.danbooru.metadata import resolve_metadata_path
from vrl.scripts.data.danbooru.safety import build_safety_prompts


def build_default_manifests(
    *,
    metadata: str | Path | None = None,
    download_metadata: bool = True,
    hf_cache_dir: str | Path | None = None,
) -> None:
    resolved = resolve_metadata_path(
        metadata=metadata,
        download_metadata=download_metadata,
        hf_repo=DANBOORU_REPO_ID,
        hf_file=DANBOORU_METADATA_FILE,
        hf_cache_dir=hf_cache_dir,
    )
    build_anatomy_prompts(metadata=resolved)
    build_safety_prompts(metadata=resolved)


def manifest_setup_hints() -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (("datasets/danbooru/anatomy/", (ANIME_PROMPTS_COMMAND,)),)


def main(
    argv: Sequence[str] | None,
    *,
    fetch: Callable[[str, Path], None],
) -> None:
    if argv is None:
        import sys

        argv = sys.argv[1:]
    if not argv:
        build_default_manifests()
        return
    parser = argparse.ArgumentParser(
        description="Build Danbooru-derived anime datasets in place.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    register(subparsers, fetch=fetch)
    args = parser.parse_args(argv)
    args.func(args)


def register(
    subparsers: Any,
    *,
    fetch: Callable[[str, Path], None],
) -> None:
    prompts = subparsers.add_parser(ANIME_PROMPTS_COMMAND)
    prompts.add_argument("--metadata", type=Path, default=None)
    prompts.set_defaults(func=_cmd_anime_prompts)

    safety = subparsers.add_parser(ANIME_SAFETY_PROMPTS_COMMAND)
    safety.add_argument("--metadata", type=Path, default=None)
    safety.set_defaults(func=_cmd_anime_safety_prompts)

    positives = subparsers.add_parser(ANIME_POSITIVES_COMMAND)
    positives.add_argument("--metadata", type=Path, required=True)
    positives.add_argument("--image-root", type=Path, default=None)
    positives.add_argument("--output", type=Path, default=POSITIVE_IMAGES_OUTPUT)
    positives.add_argument("--hand-crops-output", type=Path, default=None)
    positives.add_argument("--fetch-images", action="store_true")
    positives.add_argument("--overwrite", action="store_true")
    positives.set_defaults(func=partial(_cmd_anime_positives, fetch=fetch))

    fetch_images = subparsers.add_parser(ANIME_FETCH_IMAGES_COMMAND)
    fetch_images.add_argument("--metadata", type=Path, required=True)
    fetch_images.add_argument("--image-root", type=Path, default=None)
    fetch_images.add_argument("--overwrite", action="store_true")
    fetch_images.set_defaults(func=partial(_cmd_anime_fetch_images, fetch=fetch))


def _cmd_anime_prompts(args: argparse.Namespace) -> None:
    build_anatomy_prompts(
        metadata=args.metadata,
        download_metadata=args.metadata is None,
    )


def _cmd_anime_safety_prompts(args: argparse.Namespace) -> None:
    build_safety_prompts(
        metadata=args.metadata,
        download_metadata=args.metadata is None,
    )


def _cmd_anime_positives(
    args: argparse.Namespace,
    *,
    fetch: Callable[[str, Path], None],
) -> None:
    report = build_positive_images(
        metadata=args.metadata,
        output=args.output,
        image_root=args.image_root,
        hand_crops_output=args.hand_crops_output,
        source=POSITIVE_IMAGE_SOURCE,
        min_score=POSITIVE_IMAGE_MIN_SCORE,
        limit=POSITIVE_IMAGE_LIMIT,
        fetch_images=args.fetch_images,
        overwrite=args.overwrite,
        fetch=fetch,
    )
    emit(report)


def _cmd_anime_fetch_images(
    args: argparse.Namespace,
    *,
    fetch: Callable[[str, Path], None],
) -> None:
    image_root = Path(
        args.image_root or (default_data_root() / "danbooru" / "images"),
    ).expanduser()
    image_root.mkdir(parents=True, exist_ok=True)
    targets = select_positive_targets(
        args.metadata,
        image_root,
        min_score=int(POSITIVE_IMAGE_MIN_SCORE),
        limit=POSITIVE_IMAGE_LIMIT,
        source=POSITIVE_IMAGE_SOURCE,
    )
    downloaded, skipped, failed = download_danbooru_images(
        args.metadata,
        targets,
        fetch=fetch,
        overwrite=args.overwrite,
    )
    emit(
        {
            "dataset": "danbooru_images",
            "image_root": str(image_root.resolve()),
            "selected": len(targets),
            "downloaded": downloaded,
            "skipped_existing": skipped,
            "failed": failed,
        },
    )
