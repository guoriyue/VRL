"""Video2World manifests from public LeRobot v3.0 robot datasets.

    python -m vrl.scripts.data.video_world {robot,droid_targets,droid_full_targets}
        [--data-root DIR] [--limit N]

- ``robot``: first frames only (``dataset/video_world_v2w``).
- ``droid_targets`` / ``droid_full_targets``: first frame plus a target clip
  (``dataset/droid_target_v2w`` / ``dataset/droid_full_target_v2w``).
  ``droid_full_targets`` also writes the text-only ``droid_full_targets_t2v``
  view (``dataset/droid_full_target_t2v``): Wan 2.1 T2V cannot consume a
  first-frame reference, so the view keeps each instruction and target video,
  drops the reference image and records ``conditioning=text_only``.

Source parsing reads the LeRobot v3.0 layout:

- captions  <- ``meta/tasks.parquet`` (first frames) or ``meta/episodes/*.parquet``
  (target clips);
- episodes  <- ``data/*.parquet`` (per-episode ``frame_index==0`` row + global index);
- pixels    <- per-camera ``videos/.../*.mp4`` decoded with PyAV (streamed over HTTP).

Reference frames and target clips are written under ``<data root>/video_world``
and never committed to git. Optional dataset and video dependencies are
imported inside the paths that use them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vrl.utils.artifacts import sha256_file
from vrl.utils.hf_hub import HuggingFaceRepoRevision
from vrl.utils.json_files import write_json, write_jsonl
from vrl.utils.media import write_png
from vrl.utils.paths import RootedPaths


@dataclass(frozen=True, slots=True)
class VideoWorldDataset:
    """One buildable manifest set: its LeRobot source and how much to take."""

    repo_id: str
    camera: str
    limit: int
    eval_limit: int
    target_clips: bool
    # Also write the text-only ``<name>_t2v`` view of the target rows.
    text_video_view: bool = False


# Manifest name -> source. The values reproduce the manifests the dataset
# presets read; ``--limit`` is the only per-run override.
DATASETS = {
    "robot": VideoWorldDataset(
        repo_id="lerobot/droid_100",
        camera="observation.images.exterior_image_1_left",
        limit=50,
        eval_limit=8,
        target_clips=False,
    ),
    "droid_targets": VideoWorldDataset(
        repo_id="lerobot/droid_100",
        camera="observation.images.exterior_image_1_left",
        limit=100,
        eval_limit=8,
        target_clips=True,
    ),
    "droid_full_targets": VideoWorldDataset(
        repo_id="lerobot/droid_1.0.1",
        camera="observation.images.exterior_1_left",
        limit=1000,
        eval_limit=64,
        target_clips=True,
        text_video_view=True,
    ),
}
SOURCE = "droid"
MAX_TARGET_FRAMES = 33
# Clip FPS when neither info.json nor the camera feature declares one.
FALLBACK_FPS = 15.0


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("name", choices=tuple(DATASETS))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--limit", type=int, default=0, help="Episodes to take; 0 uses the dataset's own count."
    )
    args = parser.parse_args(argv)

    dataset = DATASETS[args.name]
    limit = args.limit or dataset.limit
    data_root = RootedPaths.data_root(args.data_root).root
    video_root = data_root / "video_world"
    reference_dir = video_root / "references"
    target_dir = video_root / "targets"
    if dataset.target_clips:
        rows = build_target_video_world_rows(
            iter_lerobot_target_clips(dataset.repo_id, camera=dataset.camera, limit=limit),
            reference_dir=reference_dir,
            target_dir=target_dir,
            data_root=data_root,
            source=SOURCE,
            fps=FALLBACK_FPS,
        )
        report: dict[str, Any] = {
            "dataset": "video_world_targets",
            "decode_method": "pyav_http_target_clip",
            "target_dir": target_dir.as_posix(),
            "max_target_frames": MAX_TARGET_FRAMES,
            "license_note": (
                "Respect the source dataset license. Target clips are decoded from public "
                "LeRobot videos and stored under data/external; do not commit them to git."
            ),
        }
    else:
        rows = build_video_world_rows(
            iter_lerobot_first_frames(dataset.repo_id, camera=dataset.camera, limit=limit),
            reference_dir=reference_dir,
            data_root=data_root,
            source=SOURCE,
        )
        report = {
            "dataset": "video_world_bridge",
            "decode_method": "pyav_http_first_frame",
            "license_note": (
                "Respect the source dataset license (DROID). "
                "Reference frames live under data/external and are not committed to git."
            ),
        }
    if not rows:
        raise RuntimeError(f"No usable episodes from {dataset.repo_id!r}.")

    manifest_dir = video_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    eval_count = min(dataset.eval_limit, len(rows) - 1) if len(rows) > 1 else 0
    split_at = len(rows) - eval_count
    train_rows, eval_rows = rows[:split_at], rows[split_at:]
    train_manifest = manifest_dir / f"{args.name}_train.jsonl"
    eval_manifest = manifest_dir / f"{args.name}_eval.jsonl"
    write_jsonl(train_manifest, train_rows)
    if eval_rows:
        write_jsonl(eval_manifest, eval_rows)
    report = {
        **report,
        "source": SOURCE,
        "repo_id": dataset.repo_id,
        "camera": dataset.camera,
        "source_split": "main",
        "episode_count": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "data_root": data_root.as_posix(),
        "train_manifest": train_manifest.as_posix(),
        "eval_manifest": eval_manifest.as_posix() if eval_rows else "",
        "reference_dir": reference_dir.as_posix(),
    }
    write_json(video_root / f"{args.name}_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not dataset.text_video_view:
        return

    view = f"{args.name}_t2v"
    view_train, view_eval = derive_text_video_rows(train_rows), derive_text_video_rows(eval_rows)
    view_train_manifest = manifest_dir / f"{view}_train.jsonl"
    view_eval_manifest = manifest_dir / f"{view}_eval.jsonl"
    write_jsonl(view_train_manifest, view_train)
    write_jsonl(view_eval_manifest, view_eval)
    train_prompts = {row["prompt"] for row in view_train}
    eval_prompts = {row["prompt"] for row in view_eval}
    view_report = {
        "dataset": "text_video_targets",
        "source_train_manifest": train_manifest.as_posix(),
        "source_eval_manifest": eval_manifest.as_posix(),
        "data_root": data_root.as_posix(),
        "train_manifest": view_train_manifest.as_posix(),
        "eval_manifest": view_eval_manifest.as_posix(),
        "train_rows": len(view_train),
        "eval_rows": len(view_eval),
        "train_unique_prompts": len(train_prompts),
        "eval_unique_prompts": len(eval_prompts),
        "prompt_overlap": sorted(train_prompts & eval_prompts),
        "train_sha256": sha256_file(view_train_manifest),
        "eval_sha256": sha256_file(view_eval_manifest),
    }
    write_json(manifest_dir / f"{view}_report.json", view_report)
    print(json.dumps(view_report, indent=2, sort_keys=True))


# ── Manifest rows ─────────────────────────────────────────────────────────


def build_video_world_rows(
    episodes: Iterable[Mapping[str, Any]],
    *,
    reference_dir: Path,
    data_root: Path,
    source: str,
) -> list[dict[str, Any]]:
    """Write first-frame PNGs and return Video2World dataset index rows.

    Each normalized episode provides ``image``, ``prompt``, and ``episode_id``.
    Incomplete episodes are ignored so a partially corrupt public source does
    not produce invalid dataset index rows.
    """

    reference_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        prompt = episode.get("prompt")
        episode_id = episode.get("episode_id")
        if prompt is None or episode_id is None:
            continue
        prompt = str(prompt).strip()
        episode_id = str(episode_id).strip()
        image = episode.get("image")
        if not prompt or not episode_id or image is None:
            continue
        metadata = _dataset_metadata(
            episode,
            source=source,
            episode_id=episode_id,
        )
        ref_path = reference_dir / f"{source}_{episode_id}_first.png"
        write_png(image, ref_path)
        rows.append(
            {
                "prompt": prompt,
                "reference_image": os.path.relpath(ref_path, data_root),
                "task_type": "video2world",
                "metadata": metadata,
            },
        )
    return rows


def build_target_video_world_rows(
    episodes: Iterable[Mapping[str, Any]],
    *,
    reference_dir: Path,
    target_dir: Path,
    data_root: Path,
    source: str,
    fps: float,
) -> list[dict[str, Any]]:
    """Write first-frame PNGs and target clips from real demonstrations."""

    import imageio.v2 as imageio
    import numpy as np

    reference_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        prompt = episode.get("prompt")
        episode_id = episode.get("episode_id")
        if prompt is None or episode_id is None:
            continue
        prompt = str(prompt).strip()
        episode_id = str(episode_id).strip()
        frames = list(episode.get("frames") or [])
        if not prompt or not episode_id or not frames:
            continue
        metadata = _dataset_metadata(
            episode,
            source=source,
            episode_id=episode_id,
        )
        metadata_raw = episode.get("metadata") or {}
        source_fps = metadata_raw.get("source_fps")
        clip_fps = float(fps if source_fps is None else source_fps)
        if not math.isfinite(clip_fps) or clip_fps <= 0:
            raise ValueError(f"episode {episode_id!r} FPS must be finite and > 0")
        ref_path = reference_dir / f"{source}_{episode_id}_first.png"
        target_path = target_dir / f"{source}_{episode_id}_target.mp4"
        write_png(frames[0], ref_path)
        arrays = []
        for frame in frames:
            array = np.asarray(frame)
            if array.dtype != np.uint8:
                array = array.astype("uint8")
            arrays.append(array)
        imageio.mimsave(target_path, arrays, fps=clip_fps, macro_block_size=1)
        rows.append(
            {
                "prompt": prompt,
                "reference_image": os.path.relpath(ref_path, data_root),
                "target_video": os.path.relpath(target_path, data_root),
                "task_type": "video2world",
                "metadata": metadata,
            },
        )
    return rows


def _dataset_metadata(
    episode: Mapping[str, Any],
    *,
    source: str,
    episode_id: str,
) -> dict[str, Any]:
    metadata = episode.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError(f"episode {episode_id!r} metadata must be a mapping or None")
    source_metadata = {
        key: value
        for key, value in ({} if metadata is None else metadata).items()
        if value is not None and str(value).strip()
    }
    return {
        "source": source,
        "source_episode": episode_id,
        "conditioning": "first_frame",
        **source_metadata,
    }


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


# ── LeRobot v3.0 sources ──────────────────────────────────────────────────


def _hub_file(repo_id: str, rel: str) -> str:
    """One file of the LeRobot dataset repo, fetched into the Hugging Face cache."""

    return str(HuggingFaceRepoRevision(repo_id, "main").download_file(rel, repo_type="dataset"))


def _hub_info(repo_id: str) -> dict[str, Any]:
    return json.loads(Path(_hub_file(repo_id, "meta/info.json")).read_text(encoding="utf-8"))


def iter_lerobot_first_frames(
    repo_id: str, *, camera: str, limit: int
) -> Iterator[dict[str, Any]]:
    """Yield {image, prompt, episode_id} per episode of the first data/video file."""

    import pyarrow.parquet as pq
    from PIL import Image

    info = _hub_info(repo_id)
    video_tmpl = str(info["video_path"])
    data_tmpl = str(info["data_path"])

    task_rows = pq.read_table(_hub_file(repo_id, "meta/tasks.parquet")).to_pylist()
    if not task_rows:
        return
    caption_col = next(col for col in task_rows[0] if col != "task_index")
    captions = {int(r["task_index"]): str(r[caption_col]).strip() for r in task_rows}

    data_rows = pq.read_table(
        _hub_file(repo_id, data_tmpl.format(chunk_index=0, file_index=0)),
        columns=["episode_index", "frame_index", "task_index", "index"],
    ).to_pylist()

    firsts: list[tuple[int, int, int]] = []
    seen: set[int] = set()
    for row in data_rows:
        episode = int(row["episode_index"])
        if int(row["frame_index"]) == 0 and episode not in seen:
            seen.add(episode)
            firsts.append((episode, int(row["index"]), int(row["task_index"])))
            if len(firsts) >= limit:
                break
    if not firsts:
        return

    base = firsts[0][1]
    firsts_by_episode = {episode: idx for (episode, idx, _) in firsts}
    targets = {idx - base: (episode, task_index) for (episode, idx, task_index) in firsts}
    rel = video_tmpl.format(video_key=camera, chunk_index=0, file_index=0)
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rel}"
    for position, frame in _decode_frames(url, stop_after=max(targets)):
        if position in targets:
            episode, task_index = targets[position]
            prompt = captions.get(task_index, "")
            if prompt:
                yield {
                    "image": Image.fromarray(frame),
                    "prompt": prompt,
                    "episode_id": f"{episode:06d}",
                    "metadata": {
                        "source_repo": repo_id,
                        "source_split": "main",
                        "source_video": rel,
                        "source_frame_index": 0,
                        "source_global_index": firsts_by_episode[episode],
                        "source_task_index": task_index,
                        "source_camera": camera,
                        "decode_method": "pyav_http_first_frame",
                        "codebase_version": str(info.get("codebase_version", "")),
                    },
                }


def iter_lerobot_target_clips(
    repo_id: str, *, camera: str, limit: int
) -> Iterator[dict[str, Any]]:
    """Yield {frames, prompt, episode_id} target clips located by ``meta/episodes/*.parquet``."""

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    info = _hub_info(repo_id)
    video_tmpl = str(info["video_path"])
    fps = float(info.get("fps") or _video_fps(info, camera) or FALLBACK_FPS)
    repo_files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    episode_files = sorted(path for path in repo_files if path.startswith("meta/episodes/"))
    selected = []
    video_chunk_col = f"videos/{camera}/chunk_index"
    video_file_col = f"videos/{camera}/file_index"
    video_from_col = f"videos/{camera}/from_timestamp"
    for episode_file in episode_files:
        table = pq.read_table(
            _hub_file(repo_id, episode_file),
            columns=[
                "episode_index",
                "tasks",
                "dataset_from_index",
                video_chunk_col,
                video_file_col,
                video_from_col,
            ],
        )
        for row in table.to_pylist():
            # The first non-empty task, else a language_instruction* column.
            tasks = row.get("tasks") or []
            texts = [str(task).strip() for task in tasks] if isinstance(tasks, list) else []
            texts.extend(
                str(row.get(key, "") or "").strip()
                for key in (
                    "language_instruction",
                    "language_instruction_2",
                    "language_instruction_3",
                )
            )
            prompt = next((text for text in texts if text), "")
            if not prompt:
                continue
            selected.append(
                {
                    "episode_id": int(row["episode_index"]),
                    "prompt": prompt,
                    "video_chunk": int(row[video_chunk_col]),
                    "video_file": int(row[video_file_col]),
                    "start_timestamp": float(row[video_from_col]),
                    "dataset_from_index": int(row.get("dataset_from_index") or 0),
                },
            )
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break
    if not selected:
        return

    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for item in selected:
        groups.setdefault((int(item["video_chunk"]), int(item["video_file"])), []).append(item)

    for (chunk_index, file_index), group in groups.items():
        rel = video_tmpl.format(
            video_key=camera,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        yield from _decode_grouped_target_clips(
            repo_id,
            rel,
            group,
            fps=fps,
            metadata_base={
                "source_repo": repo_id,
                "source_split": "main",
                "source_video": rel,
                "source_fps": fps,
                "source_camera": camera,
                "decode_method": "pyav_http_target_clip",
                "codebase_version": str(info.get("codebase_version", "")),
            },
        )


def _decode_grouped_target_clips(
    repo_id: str,
    rel: str,
    group: Sequence[Mapping[str, Any]],
    *,
    fps: float,
    metadata_base: Mapping[str, Any],
) -> Iterator[dict[str, Any]]:
    entries = []
    for item in group:
        start_frame = round(float(item["start_timestamp"]) * fps)
        entries.append(
            {
                "episode_id": int(item["episode_id"]),
                "prompt": str(item["prompt"]),
                "start_frame": start_frame,
                "end_frame": start_frame + MAX_TARGET_FRAMES - 1,
                "dataset_from_index": int(item.get("dataset_from_index") or 0),
                "frames": [],
            },
        )
    if not entries:
        return

    starts: dict[int, list[dict[str, Any]]] = {}
    for entry in entries:
        starts.setdefault(int(entry["start_frame"]), []).append(entry)
    active: list[dict[str, Any]] = []
    max_stop = max(int(entry["end_frame"]) for entry in entries)
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rel}"

    for position, frame in _decode_frames(url, stop_after=max_stop):
        active.extend(starts.get(position, []))
        done: list[dict[str, Any]] = []
        keep: list[dict[str, Any]] = []
        for entry in active:
            if int(entry["start_frame"]) <= position <= int(entry["end_frame"]):
                entry["frames"].append(frame)
            if len(entry["frames"]) >= MAX_TARGET_FRAMES or position >= int(entry["end_frame"]):
                done.append(entry)
            else:
                keep.append(entry)
        active = keep
        for entry in done:
            frames = list(entry["frames"])
            if frames:
                metadata = dict(metadata_base)
                metadata.update(
                    {
                        "source_frame_index": int(entry["start_frame"]),
                        "source_dataset_from_index": int(entry["dataset_from_index"]),
                        "source_target_frame_count": len(frames),
                    },
                )
                yield {
                    "frames": frames,
                    "prompt": str(entry["prompt"]),
                    "episode_id": f"{int(entry['episode_id']):06d}",
                    "metadata": metadata,
                }


def _decode_frames(url: str, *, stop_after: int) -> Iterator[tuple[int, Any]]:
    import av

    with av.open(url) as container:
        for position, frame in enumerate(container.decode(video=0)):
            yield position, frame.to_ndarray(format="rgb24")
            if position >= stop_after:
                break


def _video_fps(info: Mapping[str, Any], video_key: str) -> float | None:
    feature = (info.get("features") or {}).get(video_key) or {}
    video_info = feature.get("video_info") or {}
    fps = video_info.get("video.fps") or feature.get("fps")
    return float(fps) if fps else None


__all__ = [
    "DATASETS",
    "VideoWorldDataset",
    "build_target_video_world_rows",
    "build_video_world_rows",
    "derive_text_video_rows",
    "iter_lerobot_first_frames",
    "iter_lerobot_target_clips",
    "main",
]


if __name__ == "__main__":
    main()
