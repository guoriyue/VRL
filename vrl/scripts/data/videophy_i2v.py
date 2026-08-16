"""VideoPhy I2V dataset population from official public videos.

This importer turns the public VideoPhy benchmark table into an image-caption
manifest for Wan I2V training:

- captions come from the existing repo split files under ``datasets/videophy``;
- source videos come from ``videophysics/videophy_test_public``;
- reference images are frame 0 decoded from the selected official video URL.

The generated PNGs and manifests live under ``data/external`` (or
``VRL_DATA_ROOT``) so binary artifacts never get committed to git.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vrl.scripts.data.common import (
    default_cache_dir,
    default_data_root,
    emit,
    repo_root,
    write_jsonl,
    write_report,
)

DEFAULT_REPO_ID = "videophysics/videophy_test_public"
DEFAULT_CSV_FILE = "videophy_test_public.csv"
COMMAND_NAME = "videophy-i2v"
DATASET_NAME = "videophy_i2v"
DECODE_METHOD = "imageio_ffmpeg_first_frame"


@dataclass(frozen=True, slots=True)
class VideoPhyVideoRow:
    """One candidate source video row from the VideoPhy public CSV."""

    caption: str
    video_url: str
    row_index: int
    source: str = ""
    states_of_matter: str = ""
    complexity: int = 0
    majority_sa: int = 0
    majority_pc: int = 0


@dataclass(frozen=True, slots=True)
class SelectedVideo:
    """One source-backed image-caption target for a split manifest row."""

    split: str
    split_index: int
    prompt: str
    source_row: VideoPhyVideoRow


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser(COMMAND_NAME)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--csv-file", default=DEFAULT_CSV_FILE)
    parser.add_argument(
        "--train-prompts", type=Path, default=repo_root() / "datasets/videophy/train.txt"
    )
    parser.add_argument(
        "--eval-prompts", type=Path, default=repo_root() / "datasets/videophy/eval.txt"
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, cap each split to the first N prompts for local smoke preparation.",
    )
    parser.add_argument(
        "--keep-videos",
        action="store_true",
        help="Keep downloaded mp4 files under data/external instead of deleting them after frame extraction.",
    )
    parser.set_defaults(func=_cmd_videophy_i2v)


def manifest_setup_hints() -> tuple[tuple[str, tuple[str, ...]], ...]:
    return ((f"data/external/{DATASET_NAME}/", (COMMAND_NAME,)),)


def expected_manifest_sources() -> dict[str, str]:
    return {
        f"data/external/{DATASET_NAME}/manifests/train.jsonl": "datasets/videophy/train.txt",
        f"data/external/{DATASET_NAME}/manifests/eval.jsonl": "datasets/videophy/eval.txt",
    }


def prepare_videophy_i2v_dataset(
    *,
    csv_path: Path,
    train_prompts: Path,
    eval_prompts: Path,
    data_root: Path,
    repo_id: str,
    csv_file: str,
    limit: int = 0,
    keep_videos: bool = False,
    width: int = 832,
    height: int = 480,
    fetch_video: Callable[[str, Path], None] | None = None,
    extract_first_frame: Callable[[Path, Path], None] | None = None,
) -> dict[str, Any]:
    """Build source-backed Wan I2V manifests from a local VideoPhy CSV."""

    dataset_root = data_root / DATASET_NAME
    manifest_dir = dataset_root / "manifests"
    image_root = dataset_root / "images"
    video_root = dataset_root / "videos"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    train_text = _read_prompts(train_prompts)
    eval_text = _read_prompts(eval_prompts)
    if limit > 0:
        train_text = train_text[:limit]
        eval_text = eval_text[:limit]

    candidates = load_videophy_video_rows(csv_path)
    selected = [
        *select_videos_for_prompts(train_text, candidates, split="train"),
        *select_videos_for_prompts(eval_text, candidates, split="eval"),
    ]

    fetch = fetch_video or _download_url
    decode = extract_first_frame or _extract_first_frame
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for item in selected:
        row = _materialize_selected_video(
            item,
            dataset_root=dataset_root,
            repo_id=repo_id,
            image_root=image_root,
            video_root=video_root,
            keep_videos=keep_videos,
            width=width,
            height=height,
            fetch_video=fetch,
            extract_first_frame=decode,
        )
        if item.split == "train":
            train_rows.append(row)
        else:
            eval_rows.append(row)

    train_manifest = manifest_dir / "train.jsonl"
    eval_manifest = manifest_dir / "eval.jsonl"
    write_jsonl(train_manifest, train_rows)
    write_jsonl(eval_manifest, eval_rows)

    report = {
        "dataset": DATASET_NAME,
        "source_repo": repo_id,
        "source_csv": csv_file,
        "source_csv_path": csv_path.as_posix(),
        "source_split": "test",
        "decode_method": DECODE_METHOD,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "data_root": data_root.as_posix(),
        "dataset_root": dataset_root.as_posix(),
        "train_manifest": train_manifest.as_posix(),
        "eval_manifest": eval_manifest.as_posix(),
        "reference_dir": image_root.as_posix(),
        "kept_videos": keep_videos,
        "image_size": {"width": width, "height": height},
        "selection_policy": (
            "Per caption, prefer rows with majority_sa=1 and majority_pc=1; "
            "tie-break by lower complexity, then CSV order."
        ),
        "license_note": (
            "Source metadata comes from the MIT-licensed Hugging Face dataset "
            "videophysics/videophy_test_public. Downloaded videos and decoded "
            "frames stay under data/external or VRL_DATA_ROOT and are not "
            "committed to git."
        ),
    }
    report_path = dataset_root / "report.json"
    write_report(report_path, report)
    report["source_report"] = report_path.as_posix()
    return report


def load_videophy_video_rows(csv_path: Path) -> list[VideoPhyVideoRow]:
    """Parse the official VideoPhy public CSV into normalized video rows."""

    rows: list[VideoPhyVideoRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, raw in enumerate(reader):
            caption = _normalize_caption(raw.get("caption", ""))
            video_url = str(raw.get("video_url", "")).strip()
            if not caption or not video_url:
                continue
            rows.append(
                VideoPhyVideoRow(
                    caption=caption,
                    video_url=video_url,
                    row_index=row_index,
                    source=str(raw.get("source", "")).strip(),
                    states_of_matter=str(raw.get("states_of_matter", "")).strip(),
                    complexity=_as_int(raw.get("complexity")),
                    majority_sa=_as_int(raw.get("majority_sa")),
                    majority_pc=_as_int(raw.get("majority_pc")),
                ),
            )
    return rows


def select_videos_for_prompts(
    prompts: Iterable[str],
    candidates: Iterable[VideoPhyVideoRow],
    *,
    split: str,
) -> list[SelectedVideo]:
    """Select one official VideoPhy video per prompt, preserving prompt order."""

    by_caption: dict[str, list[VideoPhyVideoRow]] = {}
    for row in candidates:
        by_caption.setdefault(_normalize_caption(row.caption), []).append(row)

    selected: list[SelectedVideo] = []
    missing: list[str] = []
    for index, prompt in enumerate(prompts):
        caption = _normalize_caption(prompt)
        rows = by_caption.get(caption, [])
        if not rows:
            missing.append(prompt)
            continue
        best = sorted(rows, key=_candidate_sort_key)[0]
        selected.append(
            SelectedVideo(
                split=split,
                split_index=index,
                prompt=prompt.strip(),
                source_row=best,
            ),
        )
    if missing:
        preview = "; ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f"; ... ({len(missing)} total)"
        raise ValueError(f"VideoPhy CSV is missing prompts for {split}: {preview}{suffix}")
    return selected


def _candidate_sort_key(row: VideoPhyVideoRow) -> tuple[int, int, int, int]:
    return (-row.majority_sa, -row.majority_pc, row.complexity, row.row_index)


def _materialize_selected_video(
    item: SelectedVideo,
    *,
    dataset_root: Path,
    repo_id: str,
    image_root: Path,
    video_root: Path,
    keep_videos: bool,
    width: int,
    height: int,
    fetch_video: Callable[[str, Path], None],
    extract_first_frame: Callable[[Path, Path], None],
) -> dict[str, Any]:
    image_path = image_root / item.split / f"{item.split_index:03d}.png"
    video_path = video_root / item.split / f"{item.split_index:03d}.mp4"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        fetch_video(item.source_row.video_url, video_path)
        extract_first_frame(video_path, image_path)
    source_size = _resize_image(image_path, width=width, height=height)
    if not keep_videos and video_path.exists():
        video_path.unlink()

    metadata = {
        "source": "videophy",
        "source_repo": repo_id,
        "source_split": "test",
        "source_csv_row": item.source_row.row_index,
        "source_video_url": item.source_row.video_url,
        "source_video": _relative_or_absolute(video_path, dataset_root) if keep_videos else "",
        "source_frame_index": 0,
        "source_frame_size": {"width": source_size[0], "height": source_size[1]},
        "image_size": {"width": width, "height": height},
        "decode_method": DECODE_METHOD,
        "conditioning": "first_frame",
        "video_source": item.source_row.source,
        "states_of_matter": item.source_row.states_of_matter,
        "complexity": item.source_row.complexity,
        "majority_sa": item.source_row.majority_sa,
        "majority_pc": item.source_row.majority_pc,
    }
    return {
        "image": _relative_or_absolute(image_path, dataset_root),
        "caption": item.prompt,
        "task_type": "image_to_video",
        "metadata": metadata,
    }


def _download_url(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return
    tmp = target.with_suffix(target.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(target)


def _extract_first_frame(video_path: Path, image_path: Path) -> None:
    import imageio.v2 as imageio
    from PIL import Image

    reader = imageio.get_reader(str(video_path), "ffmpeg")
    try:
        frame = reader.get_data(0)
    finally:
        reader.close()
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).convert("RGB").save(image_path, format="PNG")


def _resize_image(image_path: Path, *, width: int, height: int) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        original_size = rgb.size
        if width > 0 and height > 0 and rgb.size != (width, height):
            rgb = rgb.resize((width, height), Image.Resampling.LANCZOS)
        rgb.save(image_path, format="PNG")
    return original_size


def _cmd_videophy_i2v(args: argparse.Namespace) -> None:
    from huggingface_hub import hf_hub_download

    data_root = args.data_root.expanduser().resolve() if args.data_root else default_data_root()
    csv_path = Path(
        hf_hub_download(
            args.repo_id,
            args.csv_file,
            repo_type="dataset",
            cache_dir=str(args.cache_dir.expanduser()),
        ),
    )
    report = prepare_videophy_i2v_dataset(
        csv_path=csv_path,
        train_prompts=args.train_prompts.expanduser(),
        eval_prompts=args.eval_prompts.expanduser(),
        data_root=data_root,
        repo_id=args.repo_id,
        csv_file=args.csv_file,
        limit=args.limit,
        keep_videos=args.keep_videos,
        width=args.width,
        height=args.height,
    )
    emit(report)


def _read_prompts(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalize_caption(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return path.as_posix()


__all__ = [
    "SelectedVideo",
    "VideoPhyVideoRow",
    "expected_manifest_sources",
    "load_videophy_video_rows",
    "manifest_setup_hints",
    "prepare_videophy_i2v_dataset",
    "register",
    "select_videos_for_prompts",
]
