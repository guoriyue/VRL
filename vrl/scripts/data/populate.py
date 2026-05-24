"""Dataset-specific population commands.

This module intentionally keeps setup logic concrete. It downloads or writes the
actual files each dataset needs; it does not introduce a generic artifact
manifest framework.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_data_root() -> Path:
    return (_repo_root() / "data" / "external").resolve()


def _default_cache_dir() -> Path:
    return (_repo_root() / "data" / "cache" / "hf").resolve()


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


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
    _emit(
        {
            "dataset": "pickapic",
            "dataset_name": dataset_name,
            "split": args.split,
            "cache_dir": str(args.cache_dir.expanduser().resolve()),
            "rows": len(dataset),
            "with_images": bool(args.with_images),
        },
    )


def _cmd_anime_prompts(args: argparse.Namespace) -> None:
    from vrl.scripts.data.danbooru import main as danbooru_main

    command = [
        "build-prompts",
        "--train-output",
        str(args.train_output),
        "--eval-output",
        str(args.eval_output),
        "--report-output",
        str(args.report_output),
        "--train-limit",
        str(args.train_limit),
        "--eval-limit",
        str(args.eval_limit),
        "--min-score",
        str(args.min_score),
        "--preferred-min-score",
        str(args.preferred_min_score),
        "--bucket-balance",
        "quota",
        "--prompt-style",
        args.prompt_style,
    ]
    if args.metadata:
        command.extend(["--metadata", str(args.metadata)])
    else:
        command.append("--download-danbooru-metadata")
    danbooru_main(command)


def _cmd_anime_positives(args: argparse.Namespace) -> None:
    from vrl.scripts.data.danbooru import main as danbooru_main

    image_root = args.image_root or (_default_data_root() / "danbooru" / "images")
    danbooru_main(
        [
            "build-positives",
            "--metadata",
            str(args.metadata),
            "--image-root",
            str(image_root),
            "--output",
            str(args.output),
            "--source",
            args.source,
            "--min-score",
            str(args.min_score),
            "--limit",
            str(args.limit),
        ],
    )
    if args.hand_crops_output:
        danbooru_main(
            [
                "build-hand-crops",
                "--positive-manifest",
                str(args.output),
                "--output",
                str(args.hand_crops_output),
                "--fallback-whole-image",
            ],
        )


def _cmd_video_world_tiny(args: argparse.Namespace) -> None:
    data_root = args.data_root.expanduser().resolve() if args.data_root else _default_data_root()
    video_root = data_root / "video_world"
    manifest_dir = args.manifest_dir or (video_root / "manifests")
    reference_dir = video_root / "references"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    train_ref = reference_dir / f"{args.name}_train_ref.ppm"
    eval_ref = reference_dir / f"{args.name}_eval_ref.ppm"
    _write_ppm(train_ref, rgb=(16, 32, 48))
    _write_ppm(eval_ref, rgb=(48, 32, 16))

    train_manifest = manifest_dir / f"{args.name}_train.jsonl"
    eval_manifest = manifest_dir / f"{args.name}_eval.jsonl"
    _write_jsonl(
        train_manifest,
        [
            _video_world_row(
                prompt="The robot arm moves toward the cup and closes its gripper.",
                reference=train_ref,
                manifest_dir=manifest_dir,
                source_episode=f"{args.name}_train_episode",
            ),
        ],
    )
    _write_jsonl(
        eval_manifest,
        [
            _video_world_row(
                prompt="The camera follows a tabletop object as it slides forward.",
                reference=eval_ref,
                manifest_dir=manifest_dir,
                source_episode=f"{args.name}_eval_episode",
            ),
        ],
    )
    _emit(
        {
            "dataset": "video_world_tiny",
            "data_root": data_root.as_posix(),
            "train_manifest": train_manifest.as_posix(),
            "eval_manifest": eval_manifest.as_posix(),
            "references": [train_ref.as_posix(), eval_ref.as_posix()],
        },
    )


def _video_world_row(
    *,
    prompt: str,
    reference: Path,
    manifest_dir: Path,
    source_episode: str,
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "reference_image": os.path.relpath(reference, manifest_dir),
        "task_type": "video2world",
        "metadata": {
            "source": "tiny_local",
            "source_episode": source_episode,
            "conditioning": "first_frame",
        },
    }


def _write_ppm(path: Path, *, rgb: tuple[int, int, int]) -> None:
    path.write_text(
        f"P3\n1 1\n255\n{rgb[0]} {rgb[1]} {rgb[2]}\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pickapic = subparsers.add_parser("pickapic")
    pickapic.add_argument("--dataset-name", default="")
    pickapic.add_argument("--split", default="train")
    pickapic.add_argument("--cache-dir", type=Path, default=_default_cache_dir())
    pickapic.add_argument("--with-images", action="store_true")
    pickapic.set_defaults(func=_cmd_pickapic)

    anime_prompts = subparsers.add_parser("anime-prompts")
    anime_prompts.add_argument("--metadata", type=Path, default=None)
    anime_prompts.add_argument(
        "--train-output",
        type=Path,
        default=Path("datasets/danbooru/anatomy/train_prompts.jsonl"),
    )
    anime_prompts.add_argument(
        "--eval-output",
        type=Path,
        default=Path("datasets/danbooru/anatomy/eval_prompts.jsonl"),
    )
    anime_prompts.add_argument(
        "--report-output",
        type=Path,
        default=Path("datasets/danbooru/anatomy/prompt_report.json"),
    )
    anime_prompts.add_argument("--train-limit", type=int, default=20_000)
    anime_prompts.add_argument("--eval-limit", type=int, default=1_000)
    anime_prompts.add_argument("--min-score", type=int, default=5)
    anime_prompts.add_argument("--preferred-min-score", type=int, default=20)
    anime_prompts.add_argument("--prompt-style", default="mixed")
    anime_prompts.set_defaults(func=_cmd_anime_prompts)

    anime_positives = subparsers.add_parser("anime-positives")
    anime_positives.add_argument("--metadata", type=Path, required=True)
    anime_positives.add_argument("--image-root", type=Path, default=None)
    anime_positives.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/danbooru/anatomy/positive_images.jsonl"),
    )
    anime_positives.add_argument("--hand-crops-output", type=Path, default=None)
    anime_positives.add_argument("--source", default="danbooru2023")
    anime_positives.add_argument("--min-score", type=int, default=20)
    anime_positives.add_argument("--limit", type=int, default=10_000)
    anime_positives.set_defaults(func=_cmd_anime_positives)

    video_world = subparsers.add_parser("video-world-tiny")
    video_world.add_argument("--data-root", type=Path, default=None)
    video_world.add_argument("--manifest-dir", type=Path, default=None)
    video_world.add_argument("--name", default="tiny")
    video_world.set_defaults(func=_cmd_video_world_tiny)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


__all__ = ["main"]


if __name__ == "__main__":
    main()
