"""Hashing helpers for the opt-in inference-quality tests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON at {target}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {target}")
    return value


def source_tree_sha256() -> str:
    """Hash runtime source plus this test contract; live edits invalidate evidence."""

    project_root = Path(__file__).resolve().parents[2]
    candidates: list[Path] = []
    for root in (project_root / "vrl", project_root / "tests" / "quality"):
        candidates.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix in {".py", ".yaml", ".yml", ".json"}
        )
    for project_file in ("pyproject.toml", "uv.lock"):
        path = project_root / project_file
        if path.is_file():
            candidates.append(path)
    snapshot: list[tuple[str, str, int, int]] = []
    for path in sorted(candidates):
        stat = path.stat()
        snapshot.append(
            (
                str(path),
                path.relative_to(project_root).as_posix(),
                stat.st_size,
                stat.st_mtime_ns,
            ),
        )
    return _source_tree_snapshot_sha256(tuple(snapshot))


def inference_environment() -> dict[str, str]:
    """Bind evidence to dependency versions that can change inference behavior."""

    versions: dict[str, str] = {}
    for package in ("torch", "diffusers", "transformers", "huggingface-hub", "peft"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


@lru_cache(maxsize=4)
def _source_tree_snapshot_sha256(
    snapshot: tuple[tuple[str, str, int, int], ...],
) -> str:
    digest = hashlib.sha256()
    for absolute, relative_name, _, _ in snapshot:
        relative = relative_name.encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with Path(absolute).open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()
