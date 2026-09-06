"""Derive honest text-to-video manifests from target-backed Video2World rows.

Wan 2.1 T2V cannot consume a first-frame reference. This tool preserves each
robot instruction and target video for supervision/evaluation while removing
the unusable conditioning artifact and recording ``conditioning=text_only``.
The source Video2World manifests remain unchanged.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from vrl.scripts.data.common import emit, write_jsonl, write_report
from vrl.trainers.data.artifacts import ArtifactManifestReport
from vrl.utils.artifacts import SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS, sha256_file

COMMAND_NAME = "derive-text-video-targets"


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="droid_full_targets_t2v")


def register(subparsers: Any) -> None:
    """Register the derivation command with the canonical dataset setup CLI."""

    parser = subparsers.add_parser(
        COMMAND_NAME,
        help="Derive text-only T2V manifests from target-backed Video2World rows.",
    )
    _add_arguments(parser)
    parser.set_defaults(func=_cmd_derive_text_video_targets)


def manifest_setup_hints() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Map the canonical derived DROID manifests to their reproducible build."""

    return (
        (
            "data/external/video_world/manifests/droid_full_targets_t2v_",
            (
                COMMAND_NAME,
                "--train-manifest",
                "data/external/video_world/manifests/droid_full_targets_train.jsonl",
                "--eval-manifest",
                "data/external/video_world/manifests/droid_full_targets_eval.jsonl",
                "--data-root",
                "data/external",
                "--output-dir",
                "data/external/video_world/manifests",
                "--name",
                "droid_full_targets_t2v",
            ),
        ),
    )


def derive_text_video_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Remove first-frame inputs while preserving target and source provenance."""

    derived: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        prompt = str(raw.get("prompt", "")).strip()
        target_video = str(raw.get("target_video", "")).strip()
        if not prompt:
            raise ValueError(f"source row {index} has no prompt")
        if not target_video:
            raise ValueError(f"source row {index} has no target_video")
        metadata = dict(raw.get("metadata") or {})
        source_conditioning = str(metadata.get("conditioning", "") or "").strip()
        if source_conditioning:
            metadata["source_conditioning"] = source_conditioning
        metadata["conditioning"] = "text_only"
        metadata["derived_from_task_type"] = str(
            raw.get("task_type") or "video2world",
        )
        derived.append(
            {
                "prompt": prompt,
                "target_video": target_video,
                "task_type": "text_to_video",
                "metadata": metadata,
            },
        )
    return derived


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}: row {index} must be a JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: manifest is empty")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)
    return parser


def _cmd_derive_text_video_targets(args: argparse.Namespace) -> None:
    train_source = args.train_manifest.expanduser().resolve()
    eval_source = args.eval_manifest.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = derive_text_video_rows(_read_jsonl(train_source))
    eval_rows = derive_text_video_rows(_read_jsonl(eval_source))
    train_output = output_dir / f"{args.name}_train.jsonl"
    eval_output = output_dir / f"{args.name}_eval.jsonl"
    write_jsonl(train_output, train_rows)
    write_jsonl(eval_output, eval_rows)

    validation = ArtifactManifestReport.from_pair(
        train_output,
        eval_output,
        data_root=data_root,
        artifact_fields=("target_video",),
        required_artifact_fields=("target_video",),
        required_metadata_fields=SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS,
    )
    train_prompts = {row["prompt"] for row in train_rows}
    eval_prompts = {row["prompt"] for row in eval_rows}
    report = {
        "dataset": "text_video_targets",
        "source_train_manifest": train_source.as_posix(),
        "source_eval_manifest": eval_source.as_posix(),
        "data_root": data_root.as_posix(),
        "train_manifest": train_output.as_posix(),
        "eval_manifest": eval_output.as_posix(),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_unique_prompts": len(train_prompts),
        "eval_unique_prompts": len(eval_prompts),
        "prompt_overlap": sorted(train_prompts & eval_prompts),
        "train_sha256": sha256_file(train_output),
        "eval_sha256": sha256_file(eval_output),
        "validation_summary": validation.to_dict(),
    }
    report_path = output_dir / f"{args.name}_report.json"
    write_report(report_path, report)
    emit({**report, "report": report_path.as_posix()})


def main(argv: list[str] | None = None) -> None:
    _cmd_derive_text_video_targets(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
