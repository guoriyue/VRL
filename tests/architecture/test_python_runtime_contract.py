"""Python runtime floor consistency checks."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_python_runtime_floor_matches_tooling_and_lockfile() -> None:
    """Keep package, lint, and resolved dependency contracts on Python 3.12."""

    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    with (ROOT / "uv.lock").open("rb") as stream:
        lock = tomllib.load(stream)

    requires_python = project["project"]["requires-python"]
    assert requires_python == ">=3.12"
    assert project["tool"]["ruff"]["target-version"] == "py312"
    assert lock["requires-python"] == requires_python
