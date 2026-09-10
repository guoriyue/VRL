"""Read and write JSON / JSONL files the same way everywhere.

Every writer here is atomic: the content goes to a temporary file in the
destination directory, is fsynced, then renamed into place. A reader never
sees a half-written file, and a crashed writer leaves nothing behind.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, TextIO


def write_json(path: str | Path, value: Any, *, overwrite: bool = True) -> Path:
    """Write ``value`` as pretty-printed, key-sorted JSON.

    With ``overwrite=False`` an existing file is left untouched and
    ``FileExistsError`` is raised, for records that must never be replaced.
    """

    def emit(handle: TextIO) -> None:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return _write_atomically(Path(path), emit, overwrite=overwrite)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    sort_keys: bool = True,
) -> int:
    """Write one JSON object per line; return how many rows were written."""

    count = 0

    def emit(handle: TextIO) -> None:
        nonlocal count
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=sort_keys) + "\n")
            count += 1

    _write_atomically(Path(path), emit, overwrite=True)
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


def _write_atomically(path: Path, emit: Callable[[TextIO], None], *, overwrite: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            emit(handle)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # A hard link fails if ``path`` exists, so the earlier record survives.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


__all__ = ["read_json", "read_jsonl", "write_json", "write_jsonl"]
