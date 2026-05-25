"""Cosmos Video2World dataset population.

``video-world-bridge`` imports first frames from a real LeRobot v2.1 robotics
dataset (DROID / BridgeData V2 style on the HF Hub):

- captions  <- ``meta/tasks.parquet`` (keyed by ``task_index``)
- episodes  <- ``data/*.parquet`` (per-episode ``frame_index==0`` row + global index)
- pixels    <- per-camera ``videos/.../*.mp4`` decoded with PyAV (streamed over HTTP)

``build_video_world_rows`` is a pure transform so it stays offline-testable; the
fetch/decode lives in ``_iter_lerobot_first_frames``. No synthetic smoke data is
shipped. Bounded to the first data/video file (use ``--limit`` to cap episodes).
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from vrl.scripts.data.common import (
    default_cache_dir,
    default_data_root,
    emit,
    write_jsonl,
    write_report,
)
from vrl.trainers.data.artifacts import validate_source_backed_video_world_manifest_pair


def register(subparsers: Any) -> None:
    bridge = subparsers.add_parser("video-world-bridge")
    bridge.add_argument("--repo-id", default="lerobot/droid_100")
    bridge.add_argument("--limit", type=int, default=50)
    bridge.add_argument("--eval-limit", type=int, default=8)
    bridge.add_argument("--data-root", type=Path, default=None)
    bridge.add_argument("--manifest-dir", type=Path, default=None)
    bridge.add_argument("--name", default="robot")
    bridge.add_argument("--source", default="droid")
    bridge.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    bridge.add_argument(
        "--camera",
        default="",
        help="observation.images.<name> video key; default = first camera",
    )
    bridge.set_defaults(func=_cmd_video_world_bridge)


def build_video_world_rows(
    episodes: Iterable[Mapping[str, Any]],
    *,
    reference_dir: Path,
    data_root: Path,
    source: str,
    conditioning: str = "first_frame",
) -> list[dict[str, Any]]:
    """Pure transform: write first-frame PNGs and return Video2World manifest rows.

    Each episode is a normalized mapping with keys ``image`` (PIL image or array),
    ``prompt`` (str), and ``episode_id`` (str).
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
        _write_png(ref_path, image)
        source_metadata = {
            key: value
            for key, value in dict(episode.get("metadata") or {}).items()
            if value is not None and str(value).strip()
        }
        metadata = {
            "source": source,
            "source_episode": episode_id,
            "conditioning": conditioning,
            **source_metadata,
        }
        rows.append(
            {
                "prompt": prompt,
                "reference_image": os.path.relpath(ref_path, data_root),
                "task_type": "video2world",
                "metadata": metadata,
            },
        )
    return rows


def _iter_lerobot_first_frames(
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
    firsts_by_episode = {
        episode: {"global_index": idx, "task_index": task_index}
        for (episode, idx, task_index) in firsts
    }
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
                        "source_global_index": firsts_by_episode[episode]["global_index"],
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


def _decode_frames(url: str, *, stop_after: int) -> Iterator[tuple[int, Any]]:
    import av

    with av.open(url) as container:
        for position, frame in enumerate(container.decode(video=0)):
            yield position, frame.to_ndarray(format="rgb24")
            if position >= stop_after:
                break


def _write_png(path: Path, image: Any) -> None:
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, Image.Image):
        image.convert("RGB").save(path, format="PNG")
        return
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = array.astype("uint8")
    Image.fromarray(array).convert("RGB").save(path, format="PNG")


def _cmd_video_world_bridge(args: argparse.Namespace) -> None:
    data_root = args.data_root.expanduser().resolve() if args.data_root else default_data_root()
    video_root = data_root / "video_world"
    manifest_dir = args.manifest_dir or (video_root / "manifests")
    reference_dir = video_root / "references"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    try:
        episodes = _iter_lerobot_first_frames(
            args.repo_id,
            limit=args.limit,
            cache_dir=args.cache_dir.expanduser(),
            camera=args.camera,
        )
        rows = build_video_world_rows(
            episodes,
            reference_dir=reference_dir,
            data_root=data_root,
            source=args.source,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "video-world-bridge needs `pyarrow`, `av`, `pillow`, and `huggingface_hub`",
        ) from exc

    if not rows:
        raise RuntimeError(
            f"No usable episodes from {args.repo_id!r}. Check it is a LeRobot v2.1 "
            "dataset (meta/tasks.parquet + data/*.parquet + videos/*.mp4), or set "
            "--camera to a valid observation.images.<name> key.",
        )

    eval_count = min(args.eval_limit, len(rows) - 1) if len(rows) > 1 else 0
    split_at = len(rows) - eval_count
    train_rows, eval_rows = rows[:split_at], rows[split_at:]

    train_manifest = manifest_dir / f"{args.name}_train.jsonl"
    eval_manifest = manifest_dir / f"{args.name}_eval.jsonl"
    write_jsonl(train_manifest, train_rows)
    if eval_rows:
        write_jsonl(eval_manifest, eval_rows)

    validation_summary: dict[str, Any] = {}
    if eval_rows:
        validation = validate_source_backed_video_world_manifest_pair(
            train_manifest,
            eval_manifest,
            data_root=data_root,
        )
        validation_summary = validation.to_dict()

    report = {
        "dataset": "video_world_bridge",
        "source": args.source,
        "repo_id": args.repo_id,
        "camera": args.camera,
        "source_split": "main",
        "decode_method": "pyav_http_first_frame",
        "episode_count": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "data_root": data_root.as_posix(),
        "train_manifest": train_manifest.as_posix(),
        "eval_manifest": eval_manifest.as_posix() if eval_rows else "",
        "reference_dir": reference_dir.as_posix(),
        "validation_summary": validation_summary,
        "license_note": (
            "Respect the source dataset license (BridgeData V2 / DROID). "
            "Reference frames live under data/external and are not committed to git."
        ),
    }
    write_report(video_root / f"{args.name}_report.json", report)
    emit(report)


__all__ = ["build_video_world_rows", "register"]
