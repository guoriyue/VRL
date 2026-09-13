"""Put back the extras `uv export` drops from requirement lines.

`uv export --format requirements-txt` flattens `cuda-toolkit[cublas,...]` to
`cuda-toolkit`. rules_python only wires a wheel's optional dependencies when
the requirement line names the extra, so torch would lose its NVIDIA runtime
libraries. uv.lock records which extras each dependency edge requests; this
copies them onto the exported lines. It adds nothing that the lock does not
already say.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?(.*)$")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requested_extras(lock: dict) -> dict[str, set[str]]:
    extras: dict[str, set[str]] = {}
    for package in lock.get("package", []):
        edges = list(package.get("dependencies", []))
        for group in package.get("optional-dependencies", {}).values():
            edges.extend(group)
        for group in package.get("dev-dependencies", {}).values():
            edges.extend(group)
        for edge in edges:
            names = edge.get("extra", [])
            if names:
                extras.setdefault(_normalize(edge["name"]), set()).update(names)
    return extras


def restore(lines: list[str], extras: dict[str, set[str]]) -> list[str]:
    out = []
    for line in lines:
        match = _NAME.match(line)
        if match and not line.startswith(("#", "-")):
            name, existing, rest = match.groups()
            wanted = set(extras.get(_normalize(name), ()))
            if existing:
                wanted.update(existing[1:-1].split(","))
            if wanted:
                line = f"{name}[{','.join(sorted(wanted))}]{rest}"
        out.append(line)
    return out


def main(lock_path: str, *requirement_files: str) -> None:
    with open(lock_path, "rb") as handle:
        extras = requested_extras(tomllib.load(handle))
    for path in map(Path, requirement_files):
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(restore(lines, extras)) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
