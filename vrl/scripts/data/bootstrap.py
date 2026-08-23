"""Per-experiment dataset bootstrap.

``for-experiment <name>`` resolves the dataset an experiment config consumes
(via the real config loader), reports whether each manifest is already present
with row counts, and prints the exact populate command to fetch anything missing.
This is the user-facing front door: pick an experiment, learn what to run.
"""

from __future__ import annotations

import argparse
import os
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vrl.scripts.data.common import emit
from vrl.scripts.data.common import repo_root as _repo_root


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

    from vrl.config.data import manifest_sources

    loader = str(data.get("loader", "") or "")
    role_paths: list[tuple[str, str]] = []
    for role in ("manifest", "eval_manifest", "source_report"):
        value = data.get(role)
        # data.manifest may declare a {path: count} mixture; every source
        # manifest of it must be present before the experiment can run.
        paths = manifest_sources(value) if role == "manifest" and value else [value]
        for path in paths:
            text = str(path or "").strip()
            if text:
                role_paths.append((role, text))

    steps: list[dict[str, Any]] = []
    ready = True
    for role, path in role_paths:
        resolved = Path(path) if os.path.isabs(path) else (repo_root / path)
        present = resolved.exists()
        rows = _count_rows(resolved) if present and role != "source_report" else 0
        expected_rows = _expected_rows_for_path(path, repo_root) if role != "source_report" else 0
        complete = present and (expected_rows <= 0 or rows >= expected_rows)
        if not complete:
            ready = False
        steps.append(
            {
                "role": role,
                "path": path,
                "present": present,
                "rows": rows,
                "expected_rows": expected_rows,
                "complete": complete,
                "get": "" if complete else _populate_hint_for_path(path),
            },
        )
    if loader == "pickapic_preference":
        from vrl.scripts.data import pickapic

        ready = False
        steps.append(
            {
                "role": "images",
                "path": "(huggingface cache)",
                "present": False,
                "rows": 0,
                "get": _setup_command(
                    pickapic.image_setup_argv(
                        str(data.get("dataset_name") or "") or None,
                    ),
                ),
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
    for prefix, argv in _manifest_setup_hints():
        if path.startswith(prefix):
            return _setup_command(argv)
    return f"{path} is not present and no populate command maps to it; see datasets/ docs"


def _manifest_setup_hints() -> tuple[tuple[str, tuple[str, ...]], ...]:
    from vrl.scripts.data import (
        danbooru,
        derive_text_video_targets,
        video_world,
        videophy_i2v,
    )

    return (
        *danbooru.manifest_setup_hints(),
        *videophy_i2v.manifest_setup_hints(),
        *derive_text_video_targets.manifest_setup_hints(),
        *video_world.manifest_setup_hints(),
    )


def _setup_command(argv: tuple[str, ...]) -> str:
    return "python -m vrl.scripts.data.setup " + shlex.join(argv)


def _expected_rows_for_path(path: str, repo_root: Path) -> int:
    from vrl.scripts.data import videophy_i2v

    expected_source = videophy_i2v.expected_manifest_sources().get(path)
    if not expected_source:
        return 0
    return _count_rows(repo_root / expected_source)


def _cmd_for_experiment(args: argparse.Namespace) -> None:
    from vrl.config.loading import load_config

    cfg = load_config(f"experiment/{args.experiment}", overrides=list(args.override))
    data = cfg.get("data", {})
    data_map = {
        key: data.get(key) for key in ("loader", "manifest", "eval_manifest", "source_report")
    }
    plan = resolve_experiment_dataset_plan(data_map, repo_root=_repo_root())
    plan["experiment"] = args.experiment
    emit(plan)
    if args.run and not plan["ready"]:
        # Paired train/eval/report paths usually share one producer. Run each
        # canonical setup command once instead of rebuilding the same dataset
        # for every missing output in the pre-run plan.
        commands = dict.fromkeys(str(step.get("get", "")) for step in plan["steps"])
        for command in commands:
            _run_setup_command(command)


def _run_setup_command(command: str) -> None:
    parts = shlex.split(command)
    if parts[:4] != ["python", "-m", "vrl.scripts.data.setup"]:
        return

    from vrl.scripts.data.setup import main as setup_main

    setup_main(parts[4:])


__all__ = ["register", "resolve_experiment_dataset_plan"]
