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
    from vrl.scripts.data import (
        danbooru,
        derive_text_video_targets,
        video_world,
        videophy_i2v,
    )

    # Which populate command produces a manifest under each path prefix.
    setup_hints = (
        *danbooru.manifest_setup_hints(),
        *videophy_i2v.manifest_setup_hints(),
        *derive_text_video_targets.manifest_setup_hints(),
        *video_world.manifest_setup_hints(),
    )
    expected_sources = videophy_i2v.expected_manifest_sources()

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
        expected_rows = 0
        if role != "source_report" and expected_sources.get(path):
            expected_rows = _count_rows(repo_root / expected_sources[path])
        complete = present and (expected_rows <= 0 or rows >= expected_rows)
        get = ""
        if not complete:
            ready = False
            get = f"{path} is not present and no populate command maps to it; see manifests/ docs"
            for prefix, argv in setup_hints:
                if path.startswith(prefix):
                    get = _setup_command(argv)
                    break
        steps.append(
            {
                "role": role,
                "path": path,
                "present": present,
                "rows": rows,
                "expected_rows": expected_rows,
                "complete": complete,
                "get": get,
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


def _setup_command(argv: tuple[str, ...]) -> str:
    return "python -m vrl.scripts.data.setup " + shlex.join(argv)


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
        from vrl.scripts.data import setup

        commands = dict.fromkeys(str(step.get("get", "")) for step in plan["steps"])
        for command in commands:
            parts = shlex.split(command)
            if parts[:3] == ["python", "-m", "vrl.scripts.data.setup"]:
                setup.main(parts[3:])


__all__ = ["register", "resolve_experiment_dataset_plan"]
