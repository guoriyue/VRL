"""Cosmos Video2World dataset population.

``video-world-bridge`` is a real first-frame importer over a LeRobot/HF robotics
dataset (BridgeData V2 / DROID style). Network access lives in the fetch adapter
(``_iter_lerobot_first_frames``); the ``build_video_world_rows`` transform is pure
so it stays offline-testable. No synthetic smoke data is shipped.
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

_BRIDGE_IMAGE_COLUMNS = (
    "observation.images.image",
    "observation.images.image_0",
    "observation.image",
    "image",
)
_BRIDGE_LANGUAGE_COLUMNS = (
    "task",
    "language_instruction",
    "prompt",
    "language",
    "instruction",
)


def register(subparsers: Any) -> None:
    bridge = subparsers.add_parser("video-world-bridge")
    bridge.add_argument("--repo-id", default="lerobot/bridge_orig")
    bridge.add_argument("--split", default="train")
    bridge.add_argument("--limit", type=int, default=200)
    bridge.add_argument("--eval-limit", type=int, default=16)
    bridge.add_argument("--data-root", type=Path, default=None)
    bridge.add_argument("--manifest-dir", type=Path, default=None)
    bridge.add_argument("--name", default="bridge")
    bridge.add_argument("--source", default="bridge")
    bridge.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    bridge.add_argument("--image-column", default="")
    bridge.add_argument("--language-column", default="")
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
        rows.append(
            {
                "prompt": prompt,
                "reference_image": os.path.relpath(ref_path, data_root),
                "task_type": "video2world",
                "metadata": {
                    "source": source,
                    "source_episode": episode_id,
                    "conditioning": conditioning,
                },
            },
        )
    return rows


def _iter_lerobot_first_frames(
    repo_id: str,
    *,
    split: str,
    limit: int,
    cache_dir: Path,
    image_column: str = "",
    language_column: str = "",
) -> Iterator[dict[str, Any]]:
    """Stream a LeRobot/HF robotics dataset and yield the first frame per episode."""

    from datasets import load_dataset

    dataset = load_dataset(repo_id, split=split, streaming=True, cache_dir=str(cache_dir))
    image_key = image_column
    language_key = language_column
    seen: set[str] = set()
    for index, record in enumerate(dataset):
        if not image_key:
            image_key = _detect_column(record, _BRIDGE_IMAGE_COLUMNS, _is_image_value)
        if not language_key:
            language_key = _detect_column(record, _BRIDGE_LANGUAGE_COLUMNS, _is_text_value)
        episode_index = record.get("episode_index", record.get("episode_id", index))
        episode_id = (
            f"{int(episode_index):06d}" if isinstance(episode_index, int) else str(episode_index)
        )
        if episode_id in seen:
            continue
        image = record.get(image_key) if image_key else None
        prompt = record.get(language_key) if language_key else ""
        if image is None or not str(prompt).strip():
            continue
        seen.add(episode_id)
        yield {"image": image, "prompt": str(prompt).strip(), "episode_id": episode_id}
        if len(seen) >= limit:
            break


def _detect_column(
    record: Mapping[str, Any],
    candidates: tuple[str, ...],
    predicate: Callable[[Any], bool],
) -> str:
    for name in candidates:
        if name in record and predicate(record[name]):
            return name
    for name, value in record.items():
        if predicate(value):
            return name
    return ""


def _is_image_value(value: Any) -> bool:
    try:
        from PIL.Image import Image as PILImage
    except ImportError:  # pragma: no cover - pillow is a hard dependency here
        PILImage = ()  # type: ignore[assignment]
    if PILImage and isinstance(value, PILImage):
        return True
    return hasattr(value, "shape") and getattr(value, "ndim", 0) >= 2


def _is_text_value(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


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
            split=args.split,
            limit=args.limit,
            cache_dir=args.cache_dir.expanduser(),
            image_column=args.image_column,
            language_column=args.language_column,
        )
        rows = build_video_world_rows(
            episodes,
            reference_dir=reference_dir,
            data_root=data_root,
            source=args.source,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "video-world-bridge needs `datasets`, `pillow`, and `numpy` installed",
        ) from exc

    if not rows:
        raise RuntimeError(
            f"No usable episodes from {args.repo_id!r} split={args.split!r}. "
            "Override --repo-id/--split, or set --image-column/--language-column "
            "if the dataset schema is non-standard.",
        )

    eval_count = min(args.eval_limit, len(rows) - 1) if len(rows) > 1 else 0
    split_at = len(rows) - eval_count
    train_rows, eval_rows = rows[:split_at], rows[split_at:]

    train_manifest = manifest_dir / f"{args.name}_train.jsonl"
    eval_manifest = manifest_dir / f"{args.name}_eval.jsonl"
    write_jsonl(train_manifest, train_rows)
    if eval_rows:
        write_jsonl(eval_manifest, eval_rows)

    report = {
        "dataset": "video_world_bridge",
        "source": args.source,
        "repo_id": args.repo_id,
        "split": args.split,
        "episode_count": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "data_root": data_root.as_posix(),
        "train_manifest": train_manifest.as_posix(),
        "eval_manifest": eval_manifest.as_posix() if eval_rows else "",
        "reference_dir": reference_dir.as_posix(),
        "license_note": (
            "Respect the source dataset license (BridgeData V2 / DROID). "
            "Reference frames live under data/external and are not committed to git."
        ),
    }
    write_report(video_root / f"{args.name}_report.json", report)
    emit(report)


__all__ = ["build_video_world_rows", "register"]
