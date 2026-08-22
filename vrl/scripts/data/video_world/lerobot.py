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
    for line in Path(dl("meta/episodes.jsonl")).read_text(encoding="utf-8").splitlines():
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


_CANONICAL_ACTION_COLS = ("action.cartesian_velocity", "action.gripper_position")


def _select_action_columns(names: Sequence[str]) -> list[str]:
    """Pick the manifest's action label columns from a data parquet schema.

    droid_1.0.1 stores every representation side by side (cartesian/joint x
    position/velocity, the packed 8-dim ``action``, ``action.original``);
    flattening them all yields a redundant 43-dim label. The reward contract
    (SPRINT_idm_action_following_reward §2) is cartesian 6 + gripper 1 with
    the gripper LAST — and the cartesian block must be the VELOCITY command:
    DROID teleop commands are velocity setpoints, zero-centered and
    recoverable from a frame pair, while ``action.cartesian_position`` is the
    absolute workspace pose, unknowable from an uncalibrated per-episode
    camera (training on it memorizes scene->position and generalizes worse
    than predicting zero — measured 2026-08-22, eval_mse_z 1.28 vs the 1.0
    baseline). Fall back to the packed ``action`` column (droid_100's 7-dim
    delta layout), then to every ``action.*`` column for datasets with
    neither.
    """

    present = set(names)
    if set(_CANONICAL_ACTION_COLS) <= present:
        return list(_CANONICAL_ACTION_COLS)
    if "action" in present:
        return ["action"]
    return sorted(c for c in names if c.startswith("action."))


def _row_action(row: Mapping[str, Any], action_cols: Sequence[str]) -> list[float]:
    """Flatten a parquet row's action column(s) into one control-step vector.

    Handles both a single ``action`` array column and split
    ``action.cartesian_position`` / ``action.gripper_position`` columns. The exact
    layout is recorded in ``metadata['action_keys']`` so rewards can interpret the
    action vector without relying on dataset-specific hidden state.
    """

    vec: list[float] = []
    for col in action_cols:
        value = row.get(col)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            vec.extend(float(x) for x in value)
        else:
            vec.append(float(value))
    return vec


def _iter_lerobot_v21_target_clips(
    repo_id: str,
    info: Mapping[str, Any],
    *,
    video_key: str,
    limit: int,
    max_target_frames: int,
    dl: Callable[[str], str],
) -> Iterator[dict[str, Any]]:
    video_tmpl = str(info["video_path"])
    fps = float(info.get("fps") or _video_fps(info, video_key) or 15.0)
    episode_files = _repo_files(repo_id, prefix="meta/episodes/")
    if episode_files:
        yield from _iter_v21_target_clips_from_episode_metadata(
            repo_id,
            info,
            video_key=video_key,
            video_tmpl=video_tmpl,
            fps=fps,
            limit=limit,
            max_target_frames=max_target_frames,
            episode_files=episode_files,
            dl=dl,
        )
        return

    yield from _iter_v21_target_clips_from_data_rows(
        repo_id,
        info,
        video_key=video_key,
        fps=fps,
        limit=limit,
        max_target_frames=max_target_frames,
        dl=dl,
    )


def _iter_v21_target_clips_from_episode_metadata(
    repo_id: str,
    info: Mapping[str, Any],
    *,
    video_key: str,
    video_tmpl: str,
    fps: float,
    limit: int,
    max_target_frames: int,
    episode_files: Sequence[str],
    dl: Callable[[str], str],
) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    selected = []
    video_chunk_col = f"videos/{video_key}/chunk_index"
    video_file_col = f"videos/{video_key}/file_index"
    video_from_col = f"videos/{video_key}/from_timestamp"
    data_chunk_col = "data/chunk_index"
    data_file_col = "data/file_index"
    for episode_file in episode_files:
        local = dl(episode_file)
        schema_names = set(pq.read_schema(local).names)
        columns = [
            "episode_index",
            "tasks",
            "dataset_from_index",
            video_chunk_col,
            video_file_col,
            video_from_col,
        ]
        has_data_location = {data_chunk_col, data_file_col} <= schema_names
        if has_data_location:
            columns += [data_chunk_col, data_file_col]
        table = pq.read_table(local, columns=columns)
        for row in table.to_pylist():
            prompt = _episode_prompt(row)
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
                    "data_chunk": int(row[data_chunk_col]) if has_data_location else None,
                    "data_file": int(row[data_file_col]) if has_data_location else None,
                },
            )
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break
    if not selected:
        return
    _attach_episode_actions(selected, info, max_target_frames=max_target_frames, dl=dl)

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


def _iter_v21_target_clips_from_data_rows(
    repo_id: str,
    info: Mapping[str, Any],
    *,
    video_key: str,
    fps: float,
    limit: int,
    max_target_frames: int,
    dl: Callable[[str], str],
) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    video_tmpl = str(info["video_path"])
    data_tmpl = str(info["data_path"])

    task_rows = pq.read_table(dl("meta/tasks.parquet")).to_pylist()
    if not task_rows:
        return
    caption_col = next(col for col in task_rows[0] if col != "task_index")
    captions = {int(r["task_index"]): str(r[caption_col]).strip() for r in task_rows}

    data_path = dl(data_tmpl.format(chunk_index=0, file_index=0))
    base_cols = ["episode_index", "frame_index", "task_index", "index"]
    schema_names = set(pq.ParquetFile(data_path).schema_arrow.names)
    action_cols = _select_action_columns(schema_names)
    data_rows = pq.read_table(
        data_path,
        columns=base_cols + [c for c in action_cols if c not in base_cols],
    ).to_pylist()
    if not data_rows:
        return

    base_index = int(data_rows[0]["index"])
    selected_order: list[int] = []
    selected: dict[int, dict[str, Any]] = {}
    positions: dict[int, int] = {}
    for row in data_rows:
        episode = int(row["episode_index"])
        if episode not in selected:
            if len(selected_order) >= limit:
                continue
            selected_order.append(episode)
            selected[episode] = {
                "task_index": int(row["task_index"]),
                "start_global_index": int(row["index"]),
                "frames": [],
                "actions": [],
            }
        if episode in selected and int(row["frame_index"]) < max_target_frames:
            positions[int(row["index"]) - base_index] = episode
            if action_cols:
                selected[episode]["actions"].append(_row_action(row, action_cols))

    if not positions:
        return
    rel = video_tmpl.format(video_key=video_key, chunk_index=0, file_index=0)
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rel}"
    for position, frame in _decode_frames(url, stop_after=max(positions)):
        episode = positions.get(position)
        if episode is not None:
            selected[episode]["frames"].append(frame)

    for episode in selected_order:
        item = selected[episode]
        frames = item["frames"]
        task_index = int(item["task_index"])
        prompt = captions.get(task_index, "")
        if prompt and frames:
            metadata: dict[str, Any] = {
                "source_repo": repo_id,
                "source_split": "main",
                "source_video": rel,
                "source_frame_index": int(item["start_global_index"]) - base_index,
                "source_target_frame_count": len(frames),
                "source_fps": fps,
                "source_camera": video_key,
                "decode_method": "pyav_http_target_clip",
                "codebase_version": str(info.get("codebase_version", "v2.1")),
            }
            actions = item.get("actions") or []
            if actions:
                aligned = [list(a) for a in actions[: len(frames)]]
                metadata["target_actions"] = aligned
                metadata["action_keys"] = action_cols
                metadata["action_dim"] = len(aligned[0]) if aligned else 0
            yield {
                "frames": frames,
                "prompt": prompt,
                "episode_id": f"{episode:06d}",
                "metadata": metadata,
            }


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
    for line in Path(dl("meta/episodes.jsonl")).read_text(encoding="utf-8").splitlines():
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


def _repo_files(repo_id: str, *, prefix: str) -> tuple[str, ...]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    return tuple(sorted(path for path in files if path.startswith(prefix)))


def _episode_prompt(row: Mapping[str, Any]) -> str:
    tasks = row.get("tasks") or []
    if isinstance(tasks, list):
        for task in tasks:
            text = str(task).strip()
            if text:
                return text
    for key in ("language_instruction", "language_instruction_2", "language_instruction_3"):
        text = str(row.get(key, "") or "").strip()
        if text:
            return text
    return ""


def _attach_episode_actions(
    selected: Sequence[dict[str, Any]],
    info: Mapping[str, Any],
    *,
    max_target_frames: int,
    dl: Callable[[str], str],
) -> None:
    """Attach instruction actions to episodes selected via v3 episode metadata.

    The v2.1 data-rows path reads actions from the rows it already scans; this
    path never touches data parquet, so the IDM action labels were silently
    absent (the SPRINT_idm_action_following_reward Phase 1 gap). v3 episode
    metadata locates each episode's rows via ``dataset_from_index`` plus the
    ``data/{chunk,file}_index`` columns, and the 15Hz data rows are
    frame-aligned, so row ``dataset_from_index + k`` labels clip frame ``k``.
    Missing location columns or action columns degrade to no labels, never to
    an error — action-less manifests stay valid for pure-video consumers.
    """

    import pyarrow.parquet as pq

    data_tmpl = str(info.get("data_path") or "")
    if not data_tmpl:
        return
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for item in selected:
        if item.get("data_chunk") is None or item.get("data_file") is None:
            return
        groups.setdefault((int(item["data_chunk"]), int(item["data_file"])), []).append(item)

    action_cols: list[str] | None = None
    for (chunk_index, file_index), items in groups.items():
        path = dl(data_tmpl.format(chunk_index=chunk_index, file_index=file_index))
        if action_cols is None:
            action_cols = _select_action_columns(pq.read_schema(path).names)
            if not action_cols:
                return
        lo = min(int(it["dataset_from_index"]) for it in items)
        hi = max(int(it["dataset_from_index"]) for it in items) + max_target_frames
        table = pq.read_table(
            path,
            columns=["index", *action_cols],
            filters=[("index", ">=", lo), ("index", "<", hi)],
        )
        rows = {int(row["index"]): row for row in table.to_pylist()}
        for item in items:
            start = int(item["dataset_from_index"])
            actions: list[list[float]] = []
            for offset in range(max_target_frames):
                row = rows.get(start + offset)
                if row is None:
                    break
                actions.append(_row_action(row, action_cols))
            if actions:
                item["actions"] = actions
                item["action_keys"] = list(action_cols)


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
                "actions": list(item.get("actions") or []),
                "action_keys": list(item.get("action_keys") or []),
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
                actions = entry.get("actions") or []
                if actions:
                    aligned = [list(a) for a in actions[: len(frames)]]
                    metadata["target_actions"] = aligned
                    metadata["action_keys"] = list(entry.get("action_keys") or [])
                    metadata["action_dim"] = len(aligned[0]) if aligned else 0
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
