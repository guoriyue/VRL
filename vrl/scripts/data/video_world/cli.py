"""CLI wiring for importing public LeRobot datasets into Video2World manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from vrl.scripts.data.common import (
    default_cache_dir,
    emit,
)
from vrl.scripts.data.manifest_artifacts import DatasetFileReport
from vrl.scripts.data.video_world.dataset_index import (
    build_target_video_world_rows,
    build_video_world_rows,
)
from vrl.scripts.data.video_world.lerobot import (
    iter_lerobot_first_frames,
    iter_lerobot_target_clips,
)
from vrl.utils.json_files import write_json, write_jsonl
from vrl.utils.paths import RootedPaths, user_path

COMMAND_NAME = "video-world-bridge"
TARGET_COMMAND_NAME = "video-world-targets"
DATASET_NAME = "video_world"


def register(subparsers: Any) -> None:
    bridge = subparsers.add_parser(COMMAND_NAME)
    bridge.set_defaults(func=_cmd_video_world_bridge)
    targets = subparsers.add_parser(TARGET_COMMAND_NAME)
    targets.set_defaults(func=_cmd_video_world_targets)
    for parser, limit, name in ((bridge, 50, "robot"), (targets, 100, "droid_targets")):
        parser.add_argument("--repo-id", default="lerobot/droid_100")
        parser.add_argument("--limit", type=int, default=limit)
        parser.add_argument("--eval-limit", type=int, default=8)
        parser.add_argument("--data-root", type=Path, default=None)
        parser.add_argument("--manifest-dir", type=Path, default=None)
        parser.add_argument("--name", default=name)
        parser.add_argument("--source", default="droid")
        parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
        parser.add_argument(
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


def manifest_setup_hints() -> tuple[tuple[str, tuple[str, ...]], ...]:
    full_targets = (
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
    )
    targets = (TARGET_COMMAND_NAME, "--repo-id", "lerobot/droid_100", "--name", "droid_targets")
    return (
        (f"data/external/{DATASET_NAME}/manifests/droid_full_targets_", full_targets),
        (f"data/external/{DATASET_NAME}/droid_full_targets_report.json", full_targets),
        (f"data/external/{DATASET_NAME}/manifests/droid_targets_", targets),
        (f"data/external/{DATASET_NAME}/droid_targets_report.json", targets),
        (f"data/external/{DATASET_NAME}/", (COMMAND_NAME,)),
    )


def _cmd_video_world_bridge(args: argparse.Namespace) -> None:
    data_root = RootedPaths.data_root(args.data_root).root
    try:
        episodes = iter_lerobot_first_frames(
            args.repo_id,
            limit=args.limit,
            cache_dir=user_path(args.cache_dir),
            camera=args.camera,
        )
        rows = build_video_world_rows(
            episodes,
            reference_dir=data_root / "video_world" / "references",
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
    _write_manifests_and_report(
        args,
        rows,
        data_root=data_root,
        require_target_video=False,
        report={
            "dataset": "video_world_bridge",
            "decode_method": "pyav_http_first_frame",
            "license_note": (
                "Respect the source dataset license (BridgeData V2 / DROID). "
                "Reference frames live under data/external and are not committed to git."
            ),
        },
    )


def _cmd_video_world_targets(args: argparse.Namespace) -> None:
    data_root = RootedPaths.data_root(args.data_root).root
    target_dir = data_root / "video_world" / "targets"
    try:
        episodes = iter_lerobot_target_clips(
            args.repo_id,
            limit=args.limit,
            cache_dir=user_path(args.cache_dir),
            camera=args.camera,
            max_target_frames=args.max_target_frames,
        )
        rows = build_target_video_world_rows(
            episodes,
            reference_dir=data_root / "video_world" / "references",
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
            "dataset with meta/episodes/*.parquet + videos/*.mp4 (v3.0) or "
            "meta/episodes.jsonl (v2.0), or set "
            "--camera to a valid observation.images.<name> key.",
        )
    _write_manifests_and_report(
        args,
        rows,
        data_root=data_root,
        require_target_video=True,
        report={
            "dataset": "video_world_targets",
            "decode_method": "pyav_http_target_clip",
            "target_dir": target_dir.as_posix(),
            "max_target_frames": int(args.max_target_frames),
            "license_note": (
                "Respect the source dataset license. Target clips are decoded from public "
                "LeRobot videos and stored under data/external; do not commit them to git."
            ),
        },
    )


def _write_manifests_and_report(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    *,
    data_root: Path,
    require_target_video: bool,
    report: dict[str, Any],
) -> None:
    """Split ``rows`` into train/eval manifests, validate them, and write the run report."""

    video_root = data_root / "video_world"
    manifest_dir = args.manifest_dir or (video_root / "manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    eval_count = min(args.eval_limit, len(rows) - 1) if len(rows) > 1 else 0
    split_at = len(rows) - eval_count
    train_rows, eval_rows = rows[:split_at], rows[split_at:]

    train_manifest = manifest_dir / f"{args.name}_train.jsonl"
    eval_manifest = manifest_dir / f"{args.name}_eval.jsonl"
    write_jsonl(train_manifest, train_rows)
    validation_summary: dict[str, Any] = {}
    if eval_rows:
        write_jsonl(eval_manifest, eval_rows)
        validation_summary = DatasetFileReport.from_video_world_manifest(
            train_manifest,
            eval_manifest=eval_manifest,
            data_root=data_root,
            require_target_video=require_target_video,
        ).to_dict()

    report = {
        **report,
        "source": args.source,
        "repo_id": args.repo_id,
        "camera": args.camera,
        "source_split": "main",
        "episode_count": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "data_root": data_root.as_posix(),
        "train_manifest": train_manifest.as_posix(),
        "eval_manifest": eval_manifest.as_posix() if eval_rows else "",
        "reference_dir": (video_root / "references").as_posix(),
        "validation_summary": validation_summary,
    }
    write_json(video_root / f"{args.name}_report.json", report)
    emit(report)


__all__ = [
    "COMMAND_NAME",
    "DATASET_NAME",
    "TARGET_COMMAND_NAME",
    "manifest_setup_hints",
    "register",
]
