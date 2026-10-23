"""Dataset setup CLI — the single entrypoint for preparing training datasets.

Each dataset has its own concrete implementation (``video_world/``, ``bootstrap.py``, ``videophy_i2v.py``, and
``derive_text_video_targets.py``); this module wires their subcommands together
and adds ``init-dirs`` for creating the empty artifact directories a dataset
downloads into. No generic artifact-manifest framework lives here.

    python -m vrl.scripts.data.setup <command>

Commands: videophy-i2v, video-world-bridge, video-world-targets,
derive-text-video-targets, for-experiment, init-dirs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vrl.scripts.data import (
    bootstrap,
    derive_text_video_targets,
    video_world,
    videophy_i2v,
)
from vrl.scripts.data.common import emit
from vrl.utils.paths import RootedPaths

# init-dirs CLI: dataset -> directories to create under the data root. This is the
# init-dirs command's own small wiring table, not a generic artifact registry (the
# module docstring rules that out); keep it local rather than a per-dataset protocol.
_INIT_DIRS_BY_DATASET = {
    "video-world": ("video_world/references", "video_world/targets"),
}


def _cmd_init_dirs(args: argparse.Namespace) -> None:
    data_root = RootedPaths.data_root(args.data_root).root
    created: list[Path] = []
    for rel in _INIT_DIRS_BY_DATASET[args.dataset]:
        path = data_root / rel
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    emit(
        {
            "dataset": args.dataset,
            "data_root": data_root.as_posix(),
            "created": [path.as_posix() for path in created],
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    videophy_i2v.register(subparsers)
    video_world.register(subparsers)
    derive_text_video_targets.register(subparsers)
    bootstrap.register(subparsers)
    init_dirs = subparsers.add_parser(
        "init-dirs",
        help="Create the empty artifact directories a dataset downloads into.",
    )
    init_dirs.add_argument("dataset", choices=tuple(_INIT_DIRS_BY_DATASET))
    init_dirs.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Artifact root. Defaults to VRL_DATA_ROOT or ./data/external.",
    )
    init_dirs.set_defaults(func=_cmd_init_dirs)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    main()
