"""VideoPhy I2V dataset population from official public videos.

This importer turns the public VideoPhy2 benchmark tables into an image-caption
manifest for Wan I2V training:

- captions come from the existing repo split files under ``manifests/videophy``;
- source videos come from ``videophysics/videophy2_train`` (training CSV) and
  ``videophysics/videophy2_test`` (test CSV), matched to the captions;
- reference images are frame 0 decoded from the selected official video URL.

The generated PNGs and manifests live under ``data/external`` (or
``VRL_DATA_ROOT``) so binary artifacts never get committed to git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

from vrl.scripts.data.common import (
    default_cache_dir,
    default_data_root,
    emit,
    repo_root,
)
from vrl.utils.json_files import write_json, write_jsonl

# (repo_id, csv_file, official split). Train captions live only in the training
# CSV and held-out captions only in the test CSV, so both are always read.
DEFAULT_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("videophysics/videophy2_train", "videophy2_training.csv", "train"),
    ("videophysics/videophy2_test", "videophy2_test.csv", "test"),
)
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
    # Which official CSV the row came from (provenance recorded per manifest row).
    repo_id: str = ""
    csv_file: str = ""
    source_split: str = ""


@dataclass(frozen=True, slots=True)
class VideoPhySource:
    """One official VideoPhy CSV: where it came from and its local copy."""

    repo_id: str
    csv_file: str
    csv_path: Path
    split: str


@dataclass(frozen=True, slots=True)
class SelectedVideo:
    """One source-backed image-caption target for a split manifest row."""

    split: str
    split_index: int
    prompt: str
    source_row: VideoPhyVideoRow


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser(COMMAND_NAME)
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        metavar="REPO_ID:CSV_FILE:SPLIT",
        help="Official VideoPhy CSV to draw videos from; repeatable. "
        "Defaults to the VideoPhy2 training and test CSVs.",
    )
    parser.add_argument(
        "--train-prompts", type=Path, default=repo_root() / "manifests/videophy/train.txt"
    )
    parser.add_argument(
        "--eval-prompts", type=Path, default=repo_root() / "manifests/videophy/eval.txt"
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
        f"data/external/{DATASET_NAME}/manifests/train.jsonl": "manifests/videophy/train.txt",
        f"data/external/{DATASET_NAME}/manifests/eval.jsonl": "manifests/videophy/eval.txt",
    }


def prepare_videophy_i2v_dataset(
    *,
    sources: Sequence[VideoPhySource],
    train_prompts: Path,
    eval_prompts: Path,
    data_root: Path,
    limit: int = 0,
    keep_videos: bool = False,
    width: int = 832,
    height: int = 480,
    fetch_video: Callable[[str, Path], None] | None = None,
    extract_first_frame: Callable[[Path, Path], None] | None = None,
) -> dict[str, Any]:
    """Build source-backed Wan I2V manifests from local VideoPhy CSVs."""

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

    candidates = [
        row
        for source in sources
        for row in load_videophy_video_rows(
            source.csv_path,
            repo_id=source.repo_id,
            csv_file=source.csv_file,
            source_split=source.split,
        )
    ]
    selected = [
        *select_videos_for_prompts(train_text, candidates, split="train"),
        *select_videos_for_prompts(eval_text, candidates, split="eval"),
    ]

    fetch = fetch_video or _download_url
    decode = extract_first_frame or _extract_first_frame
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for item in selected:
        alternatives = sorted(
            (
                r
                for r in candidates
                if _normalize_caption(r.caption) == _normalize_caption(item.prompt)
            ),
            key=_candidate_sort_key,
        )
        unavailable = []
        for candidate in alternatives:
            try:
                row = _materialize_selected_video(
                    replace(item, source_row=candidate),
                    dataset_root=dataset_root,
                    image_root=image_root,
                    video_root=video_root,
                    keep_videos=keep_videos,
                    width=width,
                    height=height,
                    fetch_video=fetch,
                    extract_first_frame=decode,
                )
                break
            except HTTPError as error:
                if error.code not in (403, 404, 410):
                    raise
                unavailable.append({"url": candidate.video_url, "status": error.code})
        else:
            raise RuntimeError(f"No accessible VideoPhy video for {item.prompt!r}: {unavailable}")
        if unavailable:
            row["metadata"]["unavailable_candidates"] = unavailable
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
        "source_repo": ",".join(source.repo_id for source in sources),
        "source_csv": ",".join(source.csv_file for source in sources),
        "source_csv_path": ",".join(source.csv_path.as_posix() for source in sources),
        "source_split": ",".join(source.split for source in sources),
        "sources": [
            {
                "repo_id": source.repo_id,
                "csv_file": source.csv_file,
                "csv_path": source.csv_path.as_posix(),
                "split": source.split,
            }
            for source in sources
        ],
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
            "Per caption, prefer rows with the highest semantic-adherence (sa) then "
            "physical-commonsense (pc) labels; "
            "tie-break by lower complexity, then CSV order; try the next candidate "
            "only when a source returns HTTP 403, 404 or 410."
        ),
        "license_note": (
            "Source metadata comes from the Hugging Face datasets "
            "videophysics/videophy2_train and videophysics/videophy2_test. "
            "Downloaded videos and decoded "
            "frames stay under data/external or VRL_DATA_ROOT and are not "
            "committed to git."
        ),
    }
    report_path = dataset_root / "report.json"
    write_json(report_path, report)
    report["source_report"] = report_path.as_posix()
    return report


def load_videophy_video_rows(
    csv_path: Path,
    *,
    repo_id: str = "",
    csv_file: str = "",
    source_split: str = "",
) -> list[VideoPhyVideoRow]:
    """Parse one official VideoPhy CSV into normalized video rows.

    VideoPhy v1 labelled rows ``majority_sa`` / ``majority_pc``; VideoPhy2 uses
    ``sa`` / ``pc`` (0-5 scales). Both spellings feed the same preference order.
    """

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
                    majority_sa=_as_int(raw.get("majority_sa", raw.get("sa"))),
                    majority_pc=_as_int(raw.get("majority_pc", raw.get("pc"))),
                    repo_id=repo_id,
                    csv_file=csv_file,
                    source_split=source_split,
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
    image_root: Path,
    video_root: Path,
    keep_videos: bool,
    width: int,
    height: int,
    fetch_video: Callable[[str, Path], None],
    extract_first_frame: Callable[[Path, Path], None],
) -> dict[str, Any]:
    source_key = hashlib.sha256(item.source_row.video_url.encode()).hexdigest()[:16]
    stem = f"{item.split_index:03d}-{source_key}-{width}x{height}"
    image_path = image_root / item.split / f"{stem}.png"
    video_path = video_root / item.split / f"{stem}.mp4"
    source_info_path = image_path.with_suffix(".source.json")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    # A different candidate must never reuse the prior candidate's frame. Keep
    # the original dimensions separately so retries do not report resized sizes.
    if not image_path.exists() or not source_info_path.exists():
        fetch_video(item.source_row.video_url, video_path)
        extract_first_frame(video_path, image_path)
        source_size = _resize_image(image_path, width=width, height=height)
        write_json(source_info_path, {"source_size": list(source_size)})
    else:
        source_size = json.loads(source_info_path.read_text())["source_size"]
        _resize_image(image_path, width=width, height=height)
    if not keep_videos and video_path.exists():
        video_path.unlink()

    metadata = {
        "source": "videophy",
        "source_repo": item.source_row.repo_id,
        "source_csv": item.source_row.csv_file,
        "source_split": item.source_row.source_split,
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
    specs = (
        DEFAULT_SOURCES if not args.source else tuple(_parse_source(item) for item in args.source)
    )
    sources = [
        VideoPhySource(
            repo_id=repo_id,
            csv_file=csv_file,
            split=split,
            csv_path=Path(
                hf_hub_download(
                    repo_id,
                    csv_file,
                    repo_type="dataset",
                    cache_dir=str(args.cache_dir.expanduser()),
                ),
            ),
        )
        for repo_id, csv_file, split in specs
    ]
    report = prepare_videophy_i2v_dataset(
        sources=sources,
        train_prompts=args.train_prompts.expanduser(),
        eval_prompts=args.eval_prompts.expanduser(),
        data_root=data_root,
        limit=args.limit,
        keep_videos=args.keep_videos,
        width=args.width,
        height=args.height,
    )
    emit(report)


def _parse_source(spec: str) -> tuple[str, str, str]:
    parts = spec.split(":")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"--source must be REPO_ID:CSV_FILE:SPLIT, got {spec!r}")
    return parts[0], parts[1], parts[2]


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
    "VideoPhySource",
    "VideoPhyVideoRow",
    "expected_manifest_sources",
    "load_videophy_video_rows",
    "manifest_setup_hints",
    "prepare_videophy_i2v_dataset",
    "register",
    "select_videos_for_prompts",
]
