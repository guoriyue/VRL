"""Pure Video2World row builders and media writers.

Source adapters normalize public datasets into episode mappings. This module
owns the persisted manifest shape and writes the reference/target media without
knowing how an episode was fetched.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from vrl.utils.media import write_png


def build_video_world_rows(
    episodes: Iterable[Mapping[str, Any]],
    *,
    reference_dir: Path,
    data_root: Path,
    source: str,
    conditioning: str = "first_frame",
) -> list[dict[str, Any]]:
    """Write first-frame PNGs and return Video2World manifest rows.

    Each normalized episode provides ``image``, ``prompt``, and ``episode_id``.
    Incomplete episodes are ignored so a partially corrupt public source does
    not produce invalid manifest rows.
    """

    reference_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        prompt = str(episode.get("prompt", "")).strip()
        episode_id = str(episode.get("episode_id", "")).strip()
        image = episode.get("image")
        if not prompt or not episode_id or image is None:
            continue
        ref_path = reference_dir / f"{source}_{episode_id}_first.png"
        write_png(image, ref_path)
        rows.append(
            {
                "prompt": prompt,
                "reference_image": os.path.relpath(ref_path, data_root),
                "task_type": "video2world",
                "metadata": _manifest_metadata(
                    episode,
                    source=source,
                    episode_id=episode_id,
                    conditioning=conditioning,
                ),
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
    conditioning: str = "first_frame",
    video_writer: Callable[[Path, Sequence[Any], float], None] | None = None,
) -> list[dict[str, Any]]:
    """Write first-frame PNGs and target clips from real demonstrations."""

    reference_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    writer = video_writer or _write_mp4
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        prompt = str(episode.get("prompt", "")).strip()
        episode_id = str(episode.get("episode_id", "")).strip()
        frames = list(episode.get("frames") or [])
        if not prompt or not episode_id or not frames:
            continue
        metadata_raw = dict(episode.get("metadata") or {})
        clip_fps = float(metadata_raw.get("source_fps") or fps)
        ref_path = reference_dir / f"{source}_{episode_id}_first.png"
        target_path = target_dir / f"{source}_{episode_id}_target.mp4"
        write_png(frames[0], ref_path)
        writer(target_path, frames, clip_fps)
        rows.append(
            {
                "prompt": prompt,
                "reference_image": os.path.relpath(ref_path, data_root),
                "target_video": os.path.relpath(target_path, data_root),
                "task_type": "video2world",
                "metadata": _manifest_metadata(
                    episode,
                    source=source,
                    episode_id=episode_id,
                    conditioning=conditioning,
                ),
            },
        )
    return rows


def _manifest_metadata(
    episode: Mapping[str, Any],
    *,
    source: str,
    episode_id: str,
    conditioning: str,
) -> dict[str, Any]:
    source_metadata = {
        key: value
        for key, value in dict(episode.get("metadata") or {}).items()
        if value is not None and str(value).strip()
    }
    return {
        "source": source,
        "source_episode": episode_id,
        "conditioning": conditioning,
        **source_metadata,
    }


def _write_mp4(path: Path, frames: Sequence[Any], fps: float) -> None:
    import imageio.v2 as imageio
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = []
    for frame in frames:
        array = np.asarray(frame)
        if array.dtype != np.uint8:
            array = array.astype("uint8")
        arrays.append(array)
    if not arrays:
        raise ValueError(f"Cannot write empty target clip: {path}")
    imageio.mimsave(path, arrays, fps=fps, macro_block_size=1)


__all__ = ["build_target_video_world_rows", "build_video_world_rows"]
