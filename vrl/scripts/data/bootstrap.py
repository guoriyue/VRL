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
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from vrl.scripts.data.common import emit
from vrl.utils.paths import repo_root as _repo_root


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser("for-experiment")
    parser.add_argument("experiment")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--run", action="store_true")
    parser.set_defaults(func=_cmd_for_experiment)


@dataclass(frozen=True, slots=True)
class DataFile:
    """One file an experiment needs: whether it is here, and how to fetch it."""

    role: str
    path: str
    present: bool
    rows: int
    # Rows the manifest must reach to be complete (0 = any), from the prompt
    # file the populate command derives it from.
    expected_rows: int = 0
    # The populate command that produces it, or a note that none does; empty
    # when the file is complete.
    fetch_command: str = ""

    @property
    def complete(self) -> bool:
        return self.present and (self.expected_rows <= 0 or self.rows >= self.expected_rows)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "complete": self.complete}


@dataclass(frozen=True, slots=True)
class ExperimentDataFiles:
    """Every file an experiment needs, each with its status and fetch command."""

    loader: str
    files: tuple[DataFile, ...]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, repo_root: Path) -> ExperimentDataFiles:
        """Read the config ``data`` section and look at the manifests under ``repo_root``."""

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

        def count_rows(path: Path) -> int:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    return sum(1 for line in handle if line.strip())
            except OSError:
                return 0

        loader = str(data.get("loader", "") or "")
        files: list[DataFile] = []
        for role in ("manifest", "eval_manifest", "source_report"):
            value = data.get(role)
            # data.manifest may declare a {path: count} mixture; every source
            # manifest of it must be present before the experiment can run.
            paths = manifest_sources(value) if role == "manifest" and value else [value]
            for raw in paths:
                path = str(raw or "").strip()
                if not path:
                    continue
                resolved = Path(path) if os.path.isabs(path) else (repo_root / path)
                present = resolved.exists()
                rows = count_rows(resolved) if present and role != "source_report" else 0
                expected_rows = 0
                if role != "source_report" and expected_sources.get(path):
                    expected_rows = count_rows(repo_root / expected_sources[path])
                data_file = DataFile(role, path, present, rows, expected_rows)
                if not data_file.complete:
                    fetch = f"{path} is not present and no populate command maps to it; see manifests/ docs"
                    for prefix, argv in setup_hints:
                        if path.startswith(prefix):
                            fetch = _setup_command(argv)
                            break
                    data_file = replace(data_file, fetch_command=fetch)
                files.append(data_file)
        if loader == "pickapic_preference":
            from vrl.scripts.data import pickapic

            files.append(
                DataFile(
                    role="images",
                    path="(huggingface cache)",
                    present=False,
                    rows=0,
                    fetch_command=_setup_command(
                        pickapic.image_setup_argv(str(data.get("dataset_name") or "") or None),
                    ),
                ),
            )
        return cls(loader=loader, files=tuple(files))

    @property
    def ready(self) -> bool:
        return all(data_file.complete for data_file in self.files)

    def to_dict(self) -> dict[str, Any]:
        return {
            "loader": self.loader,
            "ready": self.ready,
            "files": [data_file.to_dict() for data_file in self.files],
        }

    def run_missing(self) -> None:
        """Run each distinct populate command the incomplete files name, once.

        Paired train/eval/report paths usually share one producer, so one
        command serves every missing output it covers.
        """

        from vrl.scripts.data import setup

        commands = dict.fromkeys(f.fetch_command for f in self.files if not f.complete)
        for command in commands:
            parts = shlex.split(command)
            if parts[:3] == ["python", "-m", "vrl.scripts.data.setup"]:
                setup.main(parts[3:])


def _setup_command(argv: tuple[str, ...]) -> str:
    return "python -m vrl.scripts.data.setup " + shlex.join(argv)


def _cmd_for_experiment(args: argparse.Namespace) -> None:
    from vrl.config.loading import load_config

    cfg = load_config(f"experiment/{args.experiment}", overrides=list(args.override))
    plan = ExperimentDataFiles.from_mapping(cfg.get("data", {}), repo_root=_repo_root())
    emit({"experiment": args.experiment, **plan.to_dict()})
    if args.run and not plan.ready:
        plan.run_missing()


__all__ = ["DataFile", "ExperimentDataFiles", "register"]
