"""CLI for importing public LeRobot datasets into Video2World manifests.

python -m vrl.scripts.data.video_world.cli {video-world-bridge,video-world-targets}
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
from vrl.utils.paths import RootedPaths

COMMAND_NAME = "video-world-bridge"
TARGET_COMMAND_NAME = "video-world-targets"
DATASET_NAME = "video_world"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    bridge = subparsers.add_parser(COMMAND_NAME)
    bridge.set_defaults(func=_cmd_video_world_bridge)
    targets = subparsers.add_parser(TARGET_COMMAND_NAME)
    targets.set_defaults(func=_cmd_video_world_targets)
    for command, limit, name in ((bridge, 50, "robot"), (targets, 100, "droid_targets")):
        command.add_argument("--repo-id", default="lerobot/droid_100")
        command.add_argument("--limit", type=int, default=limit)
        command.add_argument("--eval-limit", type=int, default=8)
        command.add_argument("--data-root", type=Path, default=None)
        command.add_argument("--manifest-dir", type=Path, default=None)
        command.add_argument("--name", default=name)
        command.add_argument("--source", default="droid")
        command.add_argument(
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
    args = parser.parse_args(argv)
    args.func(args)


def _cmd_video_world_bridge(args: argparse.Namespace) -> None:
    data_root = RootedPaths.data_root(args.data_root).root
    try:
        episodes = iter_lerobot_first_frames(
            args.repo_id,
            limit=args.limit,
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
    print(json.dumps(report, indent=2, sort_keys=True))


__all__ = ["COMMAND_NAME", "DATASET_NAME", "TARGET_COMMAND_NAME", "main"]


if __name__ == "__main__":
    main()
