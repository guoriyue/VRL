"""CLI composition for Danbooru dataset builders."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vrl.scripts.data.danbooru.config import (
    ANIME_SAFETY_PROMPTS_COMMAND,
    DANBOORU_METADATA_FILE,
    DANBOORU_REPO_ID,
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
    build_safety_prompts(metadata=resolved)


def manifest_setup_hints() -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (("manifests/danbooru/safety/", (ANIME_SAFETY_PROMPTS_COMMAND,)),)


def main(argv: Sequence[str] | None) -> None:
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
    register(subparsers)
    args = parser.parse_args(argv)
    args.func(args)


def register(subparsers: Any) -> None:
    safety = subparsers.add_parser(ANIME_SAFETY_PROMPTS_COMMAND)
    safety.add_argument("--metadata", type=Path, default=None)
    safety.set_defaults(func=_cmd_anime_safety_prompts)


def _cmd_anime_safety_prompts(args: argparse.Namespace) -> None:
    build_safety_prompts(
        metadata=args.metadata,
        download_metadata=args.metadata is None,
    )
