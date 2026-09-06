"""CLI wiring for importing public LeRobot datasets into Video2World manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from vrl.scripts.data.common import (
    default_cache_dir,
    default_data_root,
    emit,
    write_jsonl,
    write_report,
)
from vrl.scripts.data.video_world.lerobot import (
    iter_lerobot_first_frames,
    iter_lerobot_target_clips,
)
from vrl.scripts.data.video_world.manifests import (
    build_target_video_world_rows,
    build_video_world_rows,
)
from vrl.trainers.data.artifacts import ArtifactManifestReport

COMMAND_NAME = "video-world-bridge"
TARGET_COMMAND_NAME = "video-world-targets"
DATASET_NAME = "video_world"


def register(subparsers: Any) -> None:
    bridge = subparsers.add_parser(COMMAND_NAME)
    bridge.add_argument("--repo-id", default="lerobot/droid_100")
    bridge.add_argument("--limit", type=int, default=50)
    bridge.add_argument("--eval-limit", type=int, default=8)
    bridge.add_argument("--data-root", type=Path, default=None)
    bridge.add_argument("--manifest-dir", type=Path, default=None)
    bridge.add_argument("--name", default="robot")
    bridge.add_argument("--source", default="droid")
    bridge.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    bridge.add_argument(
        "--camera",
        default="",
        help="observation.images.<name> video key; default = first camera",
    )
    bridge.set_defaults(func=_cmd_video_world_bridge)

    targets = subparsers.add_parser(TARGET_COMMAND_NAME)
    targets.add_argument("--repo-id", default="lerobot/droid_100")
    targets.add_argument("--limit", type=int, default=100)
    targets.add_argument("--eval-limit", type=int, default=8)
    targets.add_argument("--data-root", type=Path, default=None)
    targets.add_argument("--manifest-dir", type=Path, default=None)
    targets.add_argument("--name", default="droid_targets")
    targets.add_argument("--source", default="droid")
    targets.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    targets.add_argument(
        "--camera",
        default="",
        help="observation.images.<name> video key; default = first camera",
    )
    targets.add_argument(
        "--max-target-frames",
        type=int,
        default=33,
        help="Maximum frames copied from each public demonstration clip.",
    )
    targets.set_defaults(func=_cmd_video_world_targets)


def manifest_setup_hints() -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (
        (
            f"data/external/{DATASET_NAME}/manifests/droid_full_targets_",
            (
                TARGET_COMMAND_NAME,
                "--repo-id",
                "lerobot/droid_1.0.1",
                "--name",
                "droid_full_targets",
                "--limit",
                "1000",
                "--eval-limit",
                "64",
                "--camera",
                "observation.images.exterior_1_left",
            ),
        ),
        (
            f"data/external/{DATASET_NAME}/droid_full_targets_report.json",
            (
                TARGET_COMMAND_NAME,
                "--repo-id",
                "lerobot/droid_1.0.1",
                "--name",
                "droid_full_targets",
                "--limit",
                "1000",
                "--eval-limit",
                "64",
                "--camera",
                "observation.images.exterior_1_left",
            ),
        ),
        (
            f"data/external/{DATASET_NAME}/manifests/droid_targets_",
            (TARGET_COMMAND_NAME, "--repo-id", "lerobot/droid_100", "--name", "droid_targets"),
        ),
        (
            f"data/external/{DATASET_NAME}/droid_targets_report.json",
            (TARGET_COMMAND_NAME, "--repo-id", "lerobot/droid_100", "--name", "droid_targets"),
        ),
        (f"data/external/{DATASET_NAME}/", (COMMAND_NAME,)),
    )


def _cmd_video_world_bridge(args: argparse.Namespace) -> None:
    data_root = args.data_root.expanduser().resolve() if args.data_root else default_data_root()
    video_root = data_root / "video_world"
    manifest_dir = args.manifest_dir or (video_root / "manifests")
    reference_dir = video_root / "references"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    try:
        episodes = iter_lerobot_first_frames(
            args.repo_id,
            limit=args.limit,
            cache_dir=args.cache_dir.expanduser(),
            camera=args.camera,
        )
        rows = build_video_world_rows(
            episodes,
            reference_dir=reference_dir,
            data_root=data_root,
            source=args.source,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "video-world-bridge needs `pyarrow`, `av`, `pillow`, and `huggingface_hub`",
        ) from exc

    if not rows:
        raise RuntimeError(
            f"No usable episodes from {args.repo_id!r}. Check it is a LeRobot v2.1 "
            "dataset (meta/tasks.parquet + data/*.parquet + videos/*.mp4), or set "
            "--camera to a valid observation.images.<name> key.",
        )

    eval_count = min(args.eval_limit, len(rows) - 1) if len(rows) > 1 else 0
    split_at = len(rows) - eval_count
    train_rows, eval_rows = rows[:split_at], rows[split_at:]

    train_manifest = manifest_dir / f"{args.name}_train.jsonl"
    eval_manifest = manifest_dir / f"{args.name}_eval.jsonl"
    write_jsonl(train_manifest, train_rows)
    if eval_rows:
        write_jsonl(eval_manifest, eval_rows)

    validation_summary: dict[str, Any] = {}
    if eval_rows:
        validation = ArtifactManifestReport.from_video_world_pair(
            train_manifest,
            eval_manifest,
            data_root=data_root,
        )
        validation_summary = validation.to_dict()

    report = {
        "dataset": "video_world_bridge",
        "source": args.source,
        "repo_id": args.repo_id,
        "camera": args.camera,
        "source_split": "main",
        "decode_method": "pyav_http_first_frame",
        "episode_count": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "data_root": data_root.as_posix(),
        "train_manifest": train_manifest.as_posix(),
        "eval_manifest": eval_manifest.as_posix() if eval_rows else "",
        "reference_dir": reference_dir.as_posix(),
        "validation_summary": validation_summary,
        "license_note": (
            "Respect the source dataset license (BridgeData V2 / DROID). "
            "Reference frames live under data/external and are not committed to git."
        ),
    }
    write_report(video_root / f"{args.name}_report.json", report)
    emit(report)


def _cmd_video_world_targets(args: argparse.Namespace) -> None:
    data_root = args.data_root.expanduser().resolve() if args.data_root else default_data_root()
    video_root = data_root / "video_world"
    manifest_dir = args.manifest_dir or (video_root / "manifests")
    reference_dir = video_root / "references"
    target_dir = video_root / "targets"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    try:
        episodes = iter_lerobot_target_clips(
            args.repo_id,
            limit=args.limit,
            cache_dir=args.cache_dir.expanduser(),
            camera=args.camera,
            max_target_frames=args.max_target_frames,
        )
        rows = build_target_video_world_rows(
            episodes,
            reference_dir=reference_dir,
            target_dir=target_dir,
            data_root=data_root,
            source=args.source,
            fps=15.0,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "video-world-targets needs `pyarrow`, `av`, `pillow`, `imageio`, and `huggingface_hub`",
        ) from exc

    if not rows:
        raise RuntimeError(
            f"No usable target clips from {args.repo_id!r}. Check it is a public LeRobot "
            "dataset with meta/tasks.parquet + data/*.parquet + videos/*.mp4, or set "
            "--camera to a valid observation.images.<name> key.",
        )

    eval_count = min(args.eval_limit, len(rows) - 1) if len(rows) > 1 else 0
    split_at = len(rows) - eval_count
    train_rows, eval_rows = rows[:split_at], rows[split_at:]

    train_manifest = manifest_dir / f"{args.name}_train.jsonl"
    eval_manifest = manifest_dir / f"{args.name}_eval.jsonl"
    write_jsonl(train_manifest, train_rows)
    if eval_rows:
        write_jsonl(eval_manifest, eval_rows)

    validation_summary: dict[str, Any] = {}
    if eval_rows:
        validation = ArtifactManifestReport.from_video_world_pair(
            train_manifest,
            eval_manifest,
            data_root=data_root,
            require_target_video=True,
        )
        validation_summary = validation.to_dict()

    report = {
        "dataset": "video_world_targets",
        "source": args.source,
        "repo_id": args.repo_id,
        "camera": args.camera,
        "source_split": "main",
        "decode_method": "pyav_http_target_clip",
        "episode_count": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "data_root": data_root.as_posix(),
        "train_manifest": train_manifest.as_posix(),
        "eval_manifest": eval_manifest.as_posix() if eval_rows else "",
        "reference_dir": reference_dir.as_posix(),
        "target_dir": target_dir.as_posix(),
        "max_target_frames": int(args.max_target_frames),
        "validation_summary": validation_summary,
        "license_note": (
            "Respect the source dataset license. Target clips are decoded from public "
            "LeRobot videos and stored under data/external; do not commit them to git."
        ),
    }
    write_report(video_root / f"{args.name}_report.json", report)
    emit(report)


__all__ = [
    "COMMAND_NAME",
    "DATASET_NAME",
    "TARGET_COMMAND_NAME",
    "manifest_setup_hints",
    "register",
]
