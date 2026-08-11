"""Positive image manifests, hand crops, and Danbooru asset downloads."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from vrl.scripts.data.common import default_data_root, write_jsonl
from vrl.scripts.data.danbooru.anatomy import is_anatomy_positive
from vrl.scripts.data.danbooru.config import (
    DOMAIN,
    POSITIVE_IMAGES_OUTPUT,
)
from vrl.scripts.data.danbooru.metadata import (
    iter_metadata,
    normalize_tags,
    record_id,
    record_score,
)


def build_positive_images(
    *,
    metadata: str | Path,
    output: str | Path = POSITIVE_IMAGES_OUTPUT,
    image_root: str | Path | None = None,
    hand_crops_output: str | Path | None = None,
    source: str = "danbooru2023",
    min_score: float = 20.0,
    limit: int | None = 10_000,
    fetch_images: bool = False,
    overwrite: bool = False,
    fetch: Callable[[str, Path], None] | None = None,
) -> dict[str, Any]:
    resolved_image_root = Path(
        image_root or (default_data_root() / "danbooru" / "images"),
    ).expanduser()
    fetched = {"selected": 0, "downloaded": 0, "skipped_existing": 0, "failed": 0}
    if fetch_images:
        resolved_image_root.mkdir(parents=True, exist_ok=True)
        targets = select_positive_targets(
            metadata,
            resolved_image_root,
            min_score=int(min_score),
            limit=None if limit is None else int(limit),
            source=source,
        )
        downloaded, skipped, failed = download_danbooru_images(
            metadata,
            targets,
            fetch=fetch or http_download,
            overwrite=overwrite,
        )
        fetched = {
            "selected": len(targets),
            "downloaded": downloaded,
            "skipped_existing": skipped,
            "failed": failed,
        }

    positive_rows = positive_image_rows(
        metadata,
        image_root=resolved_image_root,
        min_score=min_score,
        limit=limit,
        source=source,
    )
    write_jsonl(output, positive_rows)

    hand_crops_written = 0
    if hand_crops_output:
        crop_rows = hand_crop_rows(
            [output],
            label="hand_ok",
            source="anime_positive",
            fallback_whole_image=True,
        )
        hand_crops_written = write_jsonl(hand_crops_output, crop_rows)

    return {
        "dataset": "danbooru_positives",
        "image_root": str(resolved_image_root.resolve()),
        "positive_manifest": str(output),
        "positives_written": len(positive_rows),
        "hand_crops_output": str(hand_crops_output) if hand_crops_output else "",
        "hand_crops_written": hand_crops_written,
        "fetched": fetched,
    }


def download_danbooru_images(
    metadata_path: str | Path,
    targets: Mapping[str, Path],
    *,
    fetch: Callable[[str, Path], None],
    overwrite: bool = False,
) -> tuple[int, int, int]:
    """Download selected Danbooru images into their positive target paths."""

    remaining: dict[str, Path] = {str(post_id): Path(path) for post_id, path in targets.items()}
    downloaded = skipped = failed = 0
    for row in iter_metadata(metadata_path):
        post_id = str(record_id(row))
        target = remaining.pop(post_id, None)
        if target is None:
            continue
        if target.exists() and not overwrite:
            skipped += 1
        else:
            url = _danbooru_file_url(row)
            if not url:
                failed += 1
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    fetch(url, target)
                    downloaded += 1
                except Exception:
                    failed += 1
        if not remaining:
            break
    return downloaded, skipped, failed


def _danbooru_file_url(row: Mapping[str, Any]) -> str:
    for key in ("file_url", "large_file_url", "preview_file_url"):
        value = row.get(key)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    return ""


def http_download(url: str, target: Path) -> None:
    import requests

    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 16):
                handle.write(chunk)


def positive_image_rows(
    metadata_path: str | Path,
    *,
    image_root: str | Path | None = None,
    min_score: float = 0.0,
    limit: int | None = None,
    source: str = "danbooru",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in iter_metadata(metadata_path):
        tags = normalize_tags(row)
        if not is_anatomy_positive(set(tags), min_score=min_score, score=record_score(row)):
            continue
        image_path = _image_path(row, image_root=image_root)
        if image_path is None:
            continue
        out.append(
            {
                "image_path": image_path,
                "source": source,
                "post_id": record_id(row),
                "score": record_score(row),
                "tags": tags,
                "domain": DOMAIN,
            },
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def hand_crop_rows(
    manifests: Sequence[str | Path],
    *,
    label: str,
    source: str,
    fallback_whole_image: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        for row in iter_metadata(manifest):
            image_path = row.get("image_path")
            if not image_path:
                continue
            boxes = _hand_boxes(row)
            if not boxes and fallback_whole_image:
                boxes = [None]
            for index, box in enumerate(boxes):
                crop_row = {
                    "image_path": str(image_path),
                    "source": row.get("source", source),
                    "parent_image_path": str(image_path),
                    "labels": [label],
                    "domain": row.get("domain", DOMAIN),
                }
                if box is not None:
                    crop_row["bbox"] = box
                    crop_row["crop_id"] = f"{Path(str(image_path)).stem}_{index:02d}"
                rows.append(crop_row)
    return rows


def _image_path(row: Mapping[str, Any], *, image_root: str | Path | None) -> str | None:
    for key in ("image_path", "file_path", "path"):
        value = row.get(key)
        if value:
            return str(Path(str(value)).expanduser())
    if image_root is None:
        return None
    post_id = record_id(row)
    if post_id is None:
        return None
    extension = str(row.get("file_ext") or row.get("ext") or "jpg").lstrip(".")
    return str(Path(image_root).expanduser() / f"{post_id}.{extension}")


def _hand_boxes(row: Mapping[str, Any]) -> list[Any]:
    for key in ("hand_boxes", "hands", "bboxes", "boxes"):
        value = row.get(key)
        if isinstance(value, list):
            return value
    bbox = row.get("bbox")
    return [bbox] if bbox is not None else []


def select_positive_targets(
    metadata_path: str | Path,
    image_root: Path,
    *,
    min_score: int,
    limit: int | None,
    source: str,
) -> dict[str, Path]:
    selected = positive_image_rows(
        metadata_path,
        image_root=image_root,
        min_score=min_score,
        limit=limit,
        source=source,
    )
    return {
        str(row["post_id"]): Path(row["image_path"])
        for row in selected
        if row.get("post_id") is not None and row.get("image_path")
    }
