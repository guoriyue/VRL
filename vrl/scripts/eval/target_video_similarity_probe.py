"""Probe target_video_similarity before using it as an online RL reward.

Examples:
  python -m vrl.scripts.eval.target_video_similarity_probe \
    --manifest data/external/video_world/manifests/droid_targets_eval.jsonl \
    --generated-dir outputs/droid_generations \
    --out outputs/droid_target_similarity_probe.jsonl

If ``--generated-dir`` is omitted, the probe scores each row's target media
against itself. That should be near 1.0 and acts as a loader/schema sanity check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.target_video_similarity import TargetVideoSimilarityModel
from vrl.trainers.data.artifacts import default_data_root, resolve_artifact_path
from vrl.trainers.data.prompts import load_prompt_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--feature-size", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_root = args.data_root.expanduser().resolve() if args.data_root else default_data_root()
    examples = load_prompt_manifest(args.manifest)
    model = TargetVideoSimilarityModel(
        {
            "data_root": str(data_root),
            "num_frames": args.num_frames,
            "feature_size": args.feature_size,
        },
    )
    generated = _generated_paths(args.generated_dir) if args.generated_dir else []
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        metadata = dict(example.metadata)
        if example.target_video:
            metadata["target_video"] = example.target_video
        if example.target_image:
            metadata["target_image"] = example.target_image
        generated_path = (
            generated[index]
            if index < len(generated)
            else _self_probe_path(example, data_root=data_root)
        )
        artifact = RewardInferenceArtifact(
            artifact_id=f"probe-{index}",
            path=str(generated_path),
            media_type="video" if generated_path.suffix.lower() not in _IMAGE_SUFFIXES else "image",
            prompt=example.prompt,
            metadata=metadata,
        )
        scores = dict(model(artifact=artifact, request=None))
        rows.append(
            {
                "row_index": index,
                "prompt": example.prompt,
                "generated_path": generated_path.as_posix(),
                "target_video": example.target_video,
                "target_image": example.target_image,
                "scores": scores,
            },
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "rows": len(rows),
        "out": args.out.as_posix(),
        "mean_target_similarity": (
            sum(row["scores"]["target_similarity"] for row in rows) / len(rows)
            if rows
            else 0.0
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _generated_paths(path: Path | None) -> list[Path]:
    if path is None:
        return []
    root = path.expanduser().resolve()
    if root.is_file():
        return [root]
    return sorted(
        item
        for item in root.iterdir()
        if item.suffix.lower() in {*_IMAGE_SUFFIXES, ".mp4", ".mov", ".avi", ".webm"}
    )


def _self_probe_path(example: Any, *, data_root: Path) -> Path:
    raw = example.target_video or example.target_image
    if not raw:
        raise ValueError(f"manifest row for prompt {example.prompt!r} has no target media")
    return resolve_artifact_path(raw, data_root=data_root, allow_absolute=True)


_IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".ppm", ".webp"}


if __name__ == "__main__":
    main()
