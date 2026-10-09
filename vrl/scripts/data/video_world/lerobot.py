"""LeRobot source parsing and clip decoding for Video2World manifests.

- captions  <- ``meta/tasks.parquet`` (keyed by ``task_index``)
- episodes  <- ``data/*.parquet`` (per-episode ``frame_index==0`` row + global index)
- pixels    <- per-camera ``videos/.../*.mp4`` decoded with PyAV (streamed over HTTP)

Both supported LeRobot layouts are normalized into episode mappings consumed by
the manifest builders. No synthetic smoke data is shipped. Imports for optional
dataset and video dependencies stay inside the paths that use them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


def iter_lerobot_first_frames(
    repo_id: str,
    *,
    limit: int,
    cache_dir: Path,
    camera: str = "",
) -> Iterator[dict[str, Any]]:
    """Yield {image, prompt, episode_id} per episode from a real LeRobot dataset.

    Auto-detects the on-disk layout:
    - v2.1: aggregated ``meta/tasks.parquet`` + per-chunk data/video files.
    - v2.0: ``meta/tasks.jsonl`` + ``meta/episodes.jsonl`` + one parquet/mp4 per
      episode.

    Captions come from the dataset's task metadata; pixels are decoded from the
    per-camera mp4 with PyAV streamed over HTTP.
    """

    import json

    from huggingface_hub import hf_hub_download

    def _dl(rel: str) -> str:
        return hf_hub_download(repo_id, rel, repo_type="dataset", cache_dir=str(cache_dir))

    info = json.loads(Path(_dl("meta/info.json")).read_text(encoding="utf-8"))
    features = list(info.get("features", {}).keys())
    video_key = camera or next(
        (f for f in features if f.startswith("observation.images.")),
        "",
    )
    if not video_key:
        return

    version = str(info.get("codebase_version", "v2.1")).lower()
    if version.startswith("v2.0"):
        yield from _iter_lerobot_v20(repo_id, info, video_key=video_key, limit=limit, dl=_dl)
    else:
        yield from _iter_lerobot_v21(repo_id, info, video_key=video_key, limit=limit, dl=_dl)


def iter_lerobot_target_clips(
    repo_id: str,
    *,
    limit: int,
    cache_dir: Path,
    camera: str = "",
    max_target_frames: int = 33,
) -> Iterator[dict[str, Any]]:
    """Yield {frames, prompt, episode_id} target clips from a real LeRobot dataset."""

    import json

    from huggingface_hub import hf_hub_download

    if max_target_frames <= 0:
        raise ValueError("--max-target-frames must be positive")

    def _dl(rel: str) -> str:
        return hf_hub_download(repo_id, rel, repo_type="dataset", cache_dir=str(cache_dir))

    info = json.loads(Path(_dl("meta/info.json")).read_text(encoding="utf-8"))
    features = list(info.get("features", {}).keys())
    video_key = camera or next(
        (f for f in features if f.startswith("observation.images.")),
        "",
    )
    if not video_key:
        return

    version = str(info.get("codebase_version", "v2.1")).lower()
    if version.startswith("v2.0"):
        yield from _iter_lerobot_v20_target_clips(
            repo_id,
            info,
            video_key=video_key,
            limit=limit,
            max_target_frames=max_target_frames,
            dl=_dl,
        )
    else:
        yield from _iter_lerobot_v21_target_clips(
            repo_id,
            info,
            video_key=video_key,
            limit=limit,
            max_target_frames=max_target_frames,
            dl=_dl,
        )


def _iter_lerobot_v21(
    repo_id: str,
    info: Mapping[str, Any],
    *,
    video_key: str,
    limit: int,
    dl: Callable[[str], str],
) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq
    from PIL import Image

    video_tmpl = str(info["video_path"])
    data_tmpl = str(info["data_path"])

    task_rows = pq.read_table(dl("meta/tasks.parquet")).to_pylist()
    if not task_rows:
        return
    caption_col = next(col for col in task_rows[0] if col != "task_index")
    captions = {int(r["task_index"]): str(r[caption_col]).strip() for r in task_rows}

    data_rows = pq.read_table(
        dl(data_tmpl.format(chunk_index=0, file_index=0)),
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
    rel = video_tmpl.format(video_key=video_key, chunk_index=0, file_index=0)
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
                        "source_camera": video_key,
                        "decode_method": "pyav_http_first_frame",
                        "codebase_version": str(info.get("codebase_version", "v2.1")),
                    },
                }


def _iter_lerobot_v20(
    repo_id: str,
    info: Mapping[str, Any],
    *,
    video_key: str,
    limit: int,
    dl: Callable[[str], str],
) -> Iterator[dict[str, Any]]:
    import json

    from PIL import Image

    video_tmpl = str(info["video_path"])
    chunks_size = int(info.get("chunks_size", 1000))

    episodes: list[dict[str, Any]] = []
    with Path(dl("meta/episodes.jsonl")).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            episodes.append(json.loads(line))
            if len(episodes) >= limit:
                break

    for ep in episodes:
        episode = int(ep["episode_index"])
        tasks = ep.get("tasks") or []
        prompt = str(tasks[0]).strip() if tasks else ""
        if not prompt:
            continue
        rel = video_tmpl.format(
            episode_chunk=episode // chunks_size,
            video_key=video_key,
            episode_index=episode,
        )
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rel}"
        for _, frame in _decode_frames(url, stop_after=0):
            yield {
                "image": Image.fromarray(frame),
                "prompt": prompt,
                "episode_id": f"{episode:06d}",
                "metadata": {
                    "source_repo": repo_id,
                    "source_split": "main",
                    "source_video": rel,
                    "source_frame_index": 0,
                    "source_camera": video_key,
                    "decode_method": "pyav_http_first_frame",
                    "codebase_version": str(info.get("codebase_version", "v2.0")),
                },
            }


def _iter_lerobot_v21_target_clips(
    repo_id: str,
    info: Mapping[str, Any],
    *,
    video_key: str,
    limit: int,
    max_target_frames: int,
    dl: Callable[[str], str],
) -> Iterator[dict[str, Any]]:
    """Cut target clips located by LeRobot v3.0 ``meta/episodes/*.parquet`` rows."""

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    video_tmpl = str(info["video_path"])
    fps = float(info.get("fps") or _video_fps(info, video_key) or 15.0)
    repo_files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    episode_files = sorted(path for path in repo_files if path.startswith("meta/episodes/"))
    selected = []
    video_chunk_col = f"videos/{video_key}/chunk_index"
    video_file_col = f"videos/{video_key}/file_index"
    video_from_col = f"videos/{video_key}/from_timestamp"
    for episode_file in episode_files:
        table = pq.read_table(
            dl(episode_file),
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
            video_key=video_key,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        yield from _decode_grouped_target_clips(
            repo_id,
            rel,
            group,
            fps=fps,
            max_target_frames=max_target_frames,
            metadata_base={
                "source_repo": repo_id,
                "source_split": "main",
                "source_video": rel,
                "source_fps": fps,
                "source_camera": video_key,
                "decode_method": "pyav_http_target_clip",
                "codebase_version": str(info.get("codebase_version", "v2.1")),
            },
        )


def _iter_lerobot_v20_target_clips(
    repo_id: str,
    info: Mapping[str, Any],
    *,
    video_key: str,
    limit: int,
    max_target_frames: int,
    dl: Callable[[str], str],
) -> Iterator[dict[str, Any]]:
    import json

    video_tmpl = str(info["video_path"])
    chunks_size = int(info.get("chunks_size", 1000))
    fps = float(info.get("fps") or _video_fps(info, video_key) or 15.0)

    episodes: list[dict[str, Any]] = []
    with Path(dl("meta/episodes.jsonl")).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            episodes.append(json.loads(line))
            if len(episodes) >= limit:
                break

    for ep in episodes:
        episode = int(ep["episode_index"])
        tasks = ep.get("tasks") or []
        prompt = str(tasks[0]).strip() if tasks else ""
        if not prompt:
            continue
        rel = video_tmpl.format(
            episode_chunk=episode // chunks_size,
            video_key=video_key,
            episode_index=episode,
        )
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rel}"
        frames = [frame for _, frame in _decode_frames(url, stop_after=max_target_frames - 1)]
        if frames:
            yield {
                "frames": frames,
                "prompt": prompt,
                "episode_id": f"{episode:06d}",
                "metadata": {
                    "source_repo": repo_id,
                    "source_split": "main",
                    "source_video": rel,
                    "source_frame_index": 0,
                    "source_target_frame_count": len(frames),
                    "source_fps": fps,
                    "source_camera": video_key,
                    "decode_method": "pyav_http_target_clip",
                    "codebase_version": str(info.get("codebase_version", "v2.0")),
                },
            }


def _decode_grouped_target_clips(
    repo_id: str,
    rel: str,
    group: Sequence[Mapping[str, Any]],
    *,
    fps: float,
    max_target_frames: int,
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
                "end_frame": start_frame + max_target_frames - 1,
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
            if len(entry["frames"]) >= max_target_frames or position >= int(entry["end_frame"]):
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


__all__ = ["iter_lerobot_first_frames", "iter_lerobot_target_clips"]
