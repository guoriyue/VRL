"""Danbooru metadata download, parsing, and record normalization."""

from __future__ import annotations

import gzip
import io
import json
import tarfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from vrl.scripts.data.danbooru.config import DANBOORU_METADATA_FILE, DANBOORU_REPO_ID


def resolve_metadata_path(
    *,
    metadata: str | Path | None,
    download_metadata: bool,
    hf_repo: str,
    hf_file: str,
    hf_cache_dir: str | Path | None,
) -> str:
    if metadata:
        return str(metadata)
    if download_metadata:
        return download_metadata_file(
            repo_id=hf_repo,
            filename=hf_file,
            cache_dir=hf_cache_dir,
        )
    raise ValueError("Provide metadata or set download_metadata=True")


def download_metadata_file(
    *,
    repo_id: str = DANBOORU_REPO_ID,
    filename: str = DANBOORU_METADATA_FILE,
    cache_dir: str | Path | None = None,
) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download Danbooru metadata") from exc

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=str(Path(cache_dir).expanduser()) if cache_dir else None,
    )


def iter_metadata(path: str | Path) -> Iterator[dict[str, Any]]:
    metadata_path = Path(path)
    suffixes = metadata_path.suffixes
    if suffixes[-2:] == [".tar", ".gz"] or metadata_path.suffix == ".tgz":
        yield from _iter_tarred_json_metadata(metadata_path)
        return

    if metadata_path.suffix == ".gz":
        with gzip.open(metadata_path, "rt", encoding="utf-8") as handle:
            yield from _iter_json_records(handle, source=str(metadata_path))
        return

    if metadata_path.suffix in {".json", ".jsonl"}:
        with metadata_path.open(encoding="utf-8") as handle:
            yield from _iter_json_records(handle, source=str(metadata_path))
        return

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    yield from _iter_json_payload(payload, source=str(metadata_path))


def _iter_tarred_json_metadata(path: Path) -> Iterator[dict[str, Any]]:
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            if not member.name.endswith((".json", ".jsonl", ".json.gz")):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            if member.name.endswith(".gz"):
                with gzip.open(extracted, "rt", encoding="utf-8") as handle:
                    yield from _iter_json_records(handle, source=f"{path}:{member.name}")
            else:
                with io.TextIOWrapper(extracted, encoding="utf-8") as handle:
                    yield from _iter_json_records(handle, source=f"{path}:{member.name}")
            return
    raise ValueError(f"{path}: expected a JSON or JSONL file inside the archive")


def _iter_json_records(handle: io.TextIOBase, *, source: str) -> Iterator[dict[str, Any]]:
    first = ""
    for line in handle:
        if line.strip():
            first = line
            break
    if not first:
        return
    try:
        first_payload = json.loads(first)
    except json.JSONDecodeError:
        payload = json.loads(first + handle.read())
        yield from _iter_json_payload(payload, source=source)
        return

    if isinstance(first_payload, dict):
        yield first_payload
        for line_number, line in enumerate(handle, start=2):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_number}: JSONL rows must be objects")
            yield row
        return

    yield from _iter_json_payload(first_payload, source=source)


def _iter_json_payload(payload: Any, *, source: str) -> Iterator[dict[str, Any]]:
    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                raise ValueError(f"{source}: JSON array entries must be objects")
            yield row
        return
    if isinstance(payload, dict):
        for key in ("posts", "items", "data", "rows"):
            rows = payload.get(key)
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        raise ValueError(f"{source}: {key} entries must be objects")
                    yield row
                return
        yield payload
        return
    raise ValueError(f"{source}: expected JSON object, JSON array, or JSONL")


def normalize_tags(row: Mapping[str, Any]) -> list[str]:
    tags: list[str] = []
    for key in (
        "tags",
        "tag_string",
        "tag_string_general",
        "tag_string_character",
        "tag_string_meta",
    ):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            tags.extend(value.split())
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            tags.extend(str(item) for item in value)
    out: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        normalized = str(tag).strip().lower().replace(" ", "_")
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def record_id(row: Mapping[str, Any]) -> Any:
    for key in ("id", "post_id", "danbooru_id"):
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def record_score(row: Mapping[str, Any]) -> float:
    try:
        return float(row.get("score", 0) or 0)
    except (TypeError, ValueError):
        return 0.0
