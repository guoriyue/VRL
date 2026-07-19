"""Tamper-evident identity checks for packaged upstream model runtimes."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path


def runtime_source_tree_sha256(
    source_root: Path,
    *,
    include_globs: Sequence[str],
    exclude_prefixes: Sequence[str] = (),
) -> str:
    """Hash the deterministic runtime subset of a vendored source tree.

    Ray deliberately omits nested Git databases and research assets from its
    working-directory archive. The parent project therefore pins the exact
    files that can execute in that archive. Paths and byte lengths are included
    in the digest so renames and concatenation ambiguities cannot preserve it.
    """

    root = source_root.resolve()
    excluded = tuple(prefix.strip("/") for prefix in exclude_prefixes)
    selected: dict[str, Path] = {}
    for pattern in include_globs:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if any(relative == prefix or relative.startswith(f"{prefix}/") for prefix in excluded):
                continue
            if path.is_symlink():
                raise ValueError(
                    f"runtime source integrity does not permit symlinks: {path}",
                )
            selected[relative] = path

    if not selected:
        raise ValueError(
            f"runtime source integrity selection is empty under {source_root}",
        )

    digest = hashlib.sha256()
    for relative, path in sorted(selected.items()):
        payload = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = ["runtime_source_tree_sha256"]
