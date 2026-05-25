"""User-facing dataset setup commands for artifact-backed manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _cmd_anime(args: argparse.Namespace) -> None:
    if not args.metadata_only:
        raise ValueError("anime setup currently supports only --metadata-only")

    from vrl.scripts.data.danbooru import build_anatomy_prompts

    build_anatomy_prompts(
        download_metadata=True,
        train_output=args.train_output,
        eval_output=args.eval_output,
        report_output=args.report_output,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        min_score=args.min_score,
        preferred_min_score=args.preferred_min_score,
        bucket_balance="quota",
        prompt_style="mixed",
    )


def _cmd_artifact_setup(args: argparse.Namespace) -> None:
    from vrl.trainers.data.artifacts import default_data_root

    data_root = args.data_root.expanduser().resolve() if args.data_root else default_data_root()
    created = _setup_artifact_dirs(args.command, args.profile, data_root=data_root)
    print(
        json.dumps(
            {
                "dataset": args.command,
                "profile": args.profile,
                "data_root": data_root.as_posix(),
                "created": [path.as_posix() for path in created],
            },
            indent=2,
            sort_keys=True,
        ),
    )


def _setup_artifact_dirs(command: str, profile: str, *, data_root: Path) -> list[Path]:
    from vrl.trainers.data.artifacts import repo_root

    paths: list[Path] = []
    if command == "pickapic":
        paths.extend([data_root / "pickapic", repo_root() / "data" / "cache" / "hf"])
    elif command == "anime-artifacts":
        paths.extend(
            [
                data_root / "danbooru" / "images",
                data_root / "danbooru" / "hand_crops",
            ],
        )
    elif command == "video-world-artifacts":
        paths.extend(
            [
                data_root / "video_world" / "references",
                data_root / "video_world" / "source_videos",
            ],
        )
    else:  # pragma: no cover - argparse prevents this
        raise ValueError(f"unknown artifact setup command: {command}")

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

    return paths


def _add_artifact_setup_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
) -> None:
    parser = subparsers.add_parser(name)
    parser.add_argument(
        "--profile",
        choices=("tiny", "metadata", "dev", "prod"),
        default="tiny",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Artifact root. Defaults to VRL_DATA_ROOT or ./data/external.",
    )
    parser.set_defaults(func=_cmd_artifact_setup)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    anime = subparsers.add_parser("anime")
    anime.add_argument("--metadata-only", action="store_true", required=True)
    anime.add_argument(
        "--train-output", type=Path, default=Path("datasets/danbooru/anatomy/train_prompts.jsonl")
    )
    anime.add_argument(
        "--eval-output", type=Path, default=Path("datasets/danbooru/anatomy/eval_prompts.jsonl")
    )
    anime.add_argument(
        "--report-output", type=Path, default=Path("datasets/danbooru/anatomy/prompt_report.json")
    )
    anime.add_argument("--train-limit", type=int, default=20_000)
    anime.add_argument("--eval-limit", type=int, default=1_000)
    anime.add_argument("--min-score", type=int, default=5)
    anime.add_argument("--preferred-min-score", type=int, default=20)
    anime.set_defaults(func=_cmd_anime)

    for name in ("pickapic", "anime-artifacts", "video-world-artifacts"):
        _add_artifact_setup_parser(subparsers, name)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


__all__ = ["main"]


if __name__ == "__main__":
    main()
