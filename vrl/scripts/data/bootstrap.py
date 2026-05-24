"""Per-experiment dataset bootstrap.

``for-experiment <name>`` resolves the dataset an experiment config consumes
(via the real config loader), reports whether each manifest is already present
with row counts, and prints the exact populate command to fetch anything missing.
This is the user-facing front door: pick an experiment, learn what to run.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vrl.scripts.data.common import emit
from vrl.scripts.data.common import repo_root as _repo_root

_MANIFEST_POPULATE_HINTS = (
    ("datasets/danbooru/anatomy/", "python -m vrl.scripts.data.populate anime-prompts"),
    ("data/external/video_world/", "python -m vrl.scripts.data.populate video-world-bridge"),
)


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser("for-experiment")
    parser.add_argument("experiment")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--run", action="store_true")
    parser.set_defaults(func=_cmd_for_experiment)


def resolve_experiment_dataset_plan(
    data: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Resolve the dataset an experiment needs into a present/get-it plan.

    Pure function: takes a config ``data`` mapping (loader + manifest paths) and a
    repo root, and reports per manifest whether it is already present (with row
    count) plus the populate command to fetch it when missing.
    """

    loader = str(data.get("loader", "") or "")
    steps: list[dict[str, Any]] = []
    ready = True
    for role in ("manifest", "eval_manifest"):
        path = str(data.get(role, "") or "").strip()
        if not path:
            continue
        resolved = Path(path) if os.path.isabs(path) else (repo_root / path)
        present = resolved.exists()
        if not present:
            ready = False
        steps.append(
            {
                "role": role,
                "path": path,
                "present": present,
                "rows": _count_rows(resolved) if present else 0,
                "get": "" if present else _populate_hint_for_path(path),
            },
        )
    if loader == "pickapic_preference":
        ready = False
        steps.append(
            {
                "role": "images",
                "path": "(huggingface cache)",
                "present": False,
                "rows": 0,
                "get": "python -m vrl.scripts.data.populate pickapic --with-images",
            },
        )
    return {"loader": loader, "ready": ready, "steps": steps}


def _count_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _populate_hint_for_path(path: str) -> str:
    for prefix, hint in _MANIFEST_POPULATE_HINTS:
        if path.startswith(prefix):
            return hint
    return f"{path} is not present and no populate command maps to it; see datasets/ docs"


def _cmd_for_experiment(args: argparse.Namespace) -> None:
    from vrl.config.loading import load_config

    cfg = load_config(f"experiment/{args.experiment}", overrides=list(args.override))
    data = cfg.get("data", {})
    data_map = {
        key: data.get(key)
        for key in ("loader", "manifest", "eval_manifest", "source_report")
    }
    plan = resolve_experiment_dataset_plan(data_map, repo_root=_repo_root())
    plan["experiment"] = args.experiment
    emit(plan)
    if args.run and not plan["ready"]:
        for step in plan["steps"]:
            if step["get"].startswith("python -m vrl.scripts.data.populate pickapic"):
                from vrl.scripts.data.populate import main as populate_main

                populate_main(["pickapic", "--with-images"])


__all__ = ["register", "resolve_experiment_dataset_plan"]
