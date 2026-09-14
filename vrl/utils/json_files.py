"""Read and write JSON / JSONL files the same way everywhere.

Every writer here is atomic: the content goes to a temporary file in the
destination directory, is fsynced, then published by rename or exclusive link.
Readers do not see partially written destination files. Normal completion and
exceptions remove temporary files; abrupt process termination may leave one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from vrl.utils.artifacts import atomic_file


def write_json(path: str | Path, value: Any, *, overwrite: bool = True) -> Path:
    """Write ``value`` as pretty-printed, key-sorted JSON.

    With ``overwrite=False`` an existing file is left untouched and
    ``FileExistsError`` is raised, for records that must never be replaced.
    """

    path = Path(path)
    with atomic_file(path, overwrite=overwrite) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_json_sha256(value: Any, *, ensure_ascii: bool = True, allow_nan: bool = True) -> str:
    """SHA-256 of ``value`` as compact, key-sorted JSON.

    The one canonical form behind every persisted JSON digest (run evidence,
    evaluation protocols, reward request fingerprints). ``ensure_ascii`` and
    ``allow_nan`` are exposed because existing digests were minted with
    specific settings; a producer and its verifier must pass the same ones.
    """

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=ensure_ascii,
        allow_nan=allow_nan,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    sort_keys: bool = True,
) -> int:
    """Write one JSON object per line; return how many rows were written."""

    count = 0

    with atomic_file(path) as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=sort_keys) + "\n")
            count += 1

    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read one JSON object per line, skipping blank lines."""

    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_number} must be a JSON object")
            rows.append(row)
    return rows


__all__ = ["read_json", "read_jsonl", "write_json", "write_jsonl"]
