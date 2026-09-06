"""JRDB (JackRabbot) sidewalk-delivery domain import into the video_world tree.

``jrdb-targets`` cuts first-frame conditioning images and short target clips out
of the official JRDB camera archives (https://jrdb.erc.monash.edu/), the
sidewalk-delivery source named by SPRINT_cosmos_robotic_data_factory_domain_rl
§4. JRDB downloads sit behind a registration wall (per-user license agreement),
so unlike the HF LeRobot importers there is no anonymous downloader here:
``--jrdb-root`` must point at the extracted official archive and the command
fails loudly with download instructions when it does not. The importer parses
the official archive layout directly (``images/<camera>/<sequence>/<frame>.jpg``)
— it is a sequence parser over the public dataset layout, not a hand-authored
local-manifest bridge.

JRDB sequences carry no language annotations; prompts are derived from the
sequence name's location tokens through a fixed template and that derivation is
recorded in row metadata (``prompt_source``) so consumers can tell templated
prompts from demonstration captions.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from vrl.scripts.data.common import default_data_root, emit, write_jsonl, write_report
from vrl.scripts.data.video_world import build_target_video_world_rows
from vrl.trainers.data.artifacts import require_source_backed_video_world_manifest_pair

COMMAND_NAME = "jrdb-targets"
SOURCE_REPO = "jrdb.erc.monash.edu"
_FRAME_SUFFIXES = (".jpg", ".jpeg", ".png")
# Sequence names look like "bytes-cafe-2019-02-07_0": location tokens, then the
# capture date, then a take index.
_SEQUENCE_LOCATION = re.compile(r"^(?P<location>.+?)-\d{4}-\d{2}-\d{2}")


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser(COMMAND_NAME)
    parser.add_argument(
        "--jrdb-root",
        type=Path,
        required=True,
        help="Extracted official JRDB archive (the directory holding images/).",
    )
    parser.add_argument(
        "--camera",
        default="image_stitched",
        help="Camera directory under images/ (image_stitched or image_0..image_8).",
    )
    parser.add_argument("--split", default="train", help="Recorded as source_split.")
    parser.add_argument(
        "--clip-frames",
        type=int,
        default=33,
        help="Frames per exported target clip (first frame doubles as reference).",
    )
    parser.add_argument(
        "--clip-stride",
        type=int,
        default=132,
        help="Frame distance between consecutive clip starts within a sequence.",
    )
    parser.add_argument(
        "--clips-per-sequence",
        type=int,
        default=4,
        help="Maximum clips cut from one sequence before moving to the next.",
    )
    parser.add_argument("--limit", type=int, default=100, help="Maximum total clips.")
    parser.add_argument("--eval-limit", type=int, default=8)
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="JRDB camera frame rate; recorded as source_fps and used for the mp4.",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--name", default="jrdb_targets")
    parser.add_argument("--source", default="jrdb")
    parser.set_defaults(func=_cmd_jrdb_targets)


def sequence_prompt(sequence: str) -> str:
    match = _SEQUENCE_LOCATION.match(sequence)
    location = (match.group("location") if match else sequence).replace("-", " ").strip()
    return f"First-person view of a mobile robot driving through {location} among pedestrians"


def iter_jrdb_clips(
    jrdb_root: Path,
    *,
    camera: str,
    split: str,
    clip_frames: int,
    clip_stride: int,
    clips_per_sequence: int,
    limit: int,
    fps: float,
) -> Iterator[Mapping[str, Any]]:
    """Yield normalized episode dicts from the official extracted JRDB layout.

    Frames load lazily per clip so large imports stream instead of holding the
    whole camera archive in memory.
    """

    from PIL import Image

    if clip_frames < 2:
        raise ValueError("--clip-frames must be >= 2 (need a first frame plus a continuation)")
    if clip_stride < clip_frames:
        raise ValueError("--clip-stride must be >= --clip-frames (clips must not overlap)")
    images_dir = jrdb_root / "images" if (jrdb_root / "images").is_dir() else jrdb_root
    camera_dir = images_dir / camera
    if not camera_dir.is_dir():
        raise FileNotFoundError(
            f"JRDB camera directory not found: {camera_dir}. Download the official "
            "archives from https://jrdb.erc.monash.edu/ (registration required), "
            "extract them, and pass the directory holding images/ as --jrdb-root.",
        )
    emitted = 0
    for sequence_dir in sorted(p for p in camera_dir.iterdir() if p.is_dir()):
        frame_paths = sorted(
            p for p in sequence_dir.iterdir() if p.suffix.lower() in _FRAME_SUFFIXES
        )
        prompt = sequence_prompt(sequence_dir.name)
        for clip_index in range(clips_per_sequence):
            start = clip_index * clip_stride
            if start + clip_frames > len(frame_paths) or emitted >= limit:
                break
            clip_paths = frame_paths[start : start + clip_frames]
            frames = [Image.open(path).convert("RGB") for path in clip_paths]
            yield {
                "prompt": prompt,
                "episode_id": f"{sequence_dir.name}_{start:06d}",
                "frames": frames,
                "metadata": {
                    "source_repo": SOURCE_REPO,
                    "source_split": split,
                    "source_video": str(sequence_dir.relative_to(jrdb_root)),
                    "source_frame_index": start,
                    "source_target_frame_count": len(clip_paths),
                    "source_fps": fps,
                    "source_camera": camera,
                    "decode_method": "local_jpg_sequence",
                    "prompt_source": "jrdb_sequence_name_template",
                },
            }
            emitted += 1
        if emitted >= limit:
            return


def _cmd_jrdb_targets(args: Any) -> None:
    data_root = args.data_root.expanduser().resolve() if args.data_root else default_data_root()
    video_root = data_root / "video_world"
    manifest_dir = args.manifest_dir or (video_root / "manifests")
    reference_dir = video_root / "references"
    target_dir = video_root / "targets"

    jrdb_root = args.jrdb_root.expanduser().resolve()
    episodes = iter_jrdb_clips(
        jrdb_root,
        camera=args.camera,
        split=args.split,
        clip_frames=args.clip_frames,
        clip_stride=args.clip_stride,
        clips_per_sequence=args.clips_per_sequence,
        limit=args.limit,
        fps=args.fps,
    )
    rows = build_target_video_world_rows(
        episodes,
        reference_dir=reference_dir,
        target_dir=target_dir,
        data_root=data_root,
        source=args.source,
        fps=args.fps,
    )
    if not rows:
        raise RuntimeError(
            f"No JRDB clips imported from {jrdb_root} (camera={args.camera}); "
            "check --jrdb-root points at the extracted archive and that the "
            "sequences hold at least --clip-frames frames.",
        )

    eval_count = min(args.eval_limit, len(rows) - 1) if len(rows) > 1 else 0
    split_at = len(rows) - eval_count
    train_rows, eval_rows = rows[:split_at], rows[split_at:]
    train_manifest = manifest_dir / f"{args.name}_train.jsonl"
    eval_manifest = manifest_dir / f"{args.name}_eval.jsonl"
    write_jsonl(train_manifest, train_rows)
    validation_summary: dict[str, Any] | None = None
    if eval_rows:
        write_jsonl(eval_manifest, eval_rows)
        validation = require_source_backed_video_world_manifest_pair(
            train_manifest,
            eval_manifest,
            data_root=data_root,
            require_target_video=True,
        )
        validation_summary = validation.to_dict()

    report = {
        "dataset": "video_world",
        "source": args.source,
        "repo_id": SOURCE_REPO,
        "jrdb_root": str(jrdb_root),
        "camera": args.camera,
        "source_split": args.split,
        "decode_method": "local_jpg_sequence",
        "clip_frames": args.clip_frames,
        "clip_stride": args.clip_stride,
        "clips_per_sequence": args.clips_per_sequence,
        "fps": args.fps,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "data_root": str(data_root),
        "train_manifest": str(train_manifest),
        "eval_manifest": str(eval_manifest) if eval_rows else None,
        "reference_dir": str(reference_dir),
        "target_dir": str(target_dir),
        "validation_summary": validation_summary,
        "license_note": (
            "JRDB is distributed for research under a per-user agreement; the "
            "archives are downloaded manually after registration and exported "
            "clips must stay within that license."
        ),
    }
    write_report(video_root / f"{args.name}_report.json", report)
    emit(report)
