"""Dataset setup CLI — the single entrypoint for preparing training datasets.

Each dataset has its own concrete script (``pickapic.py``, ``danbooru.py``,
``video_world.py``, ``bootstrap.py``, ``videophy_i2v.py``); this module wires
their subcommands together and adds ``init-dirs`` for creating the empty
artifact directories a dataset downloads into. No generic artifact-manifest
framework lives here.

    python -m vrl.scripts.data.setup <command>

Commands: pickapic, anime-prompts, anime-safety-prompts, anime-positives,
anime-fetch-images, videophy-i2v, video-world-bridge, video-world-targets,
for-experiment, init-dirs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vrl.scripts.data import bootstrap, danbooru, pickapic, video_world, videophy_i2v

# init-dirs CLI: dataset -> directories to create under the data root. This is the
# init-dirs command's own small wiring table, not a generic artifact registry (the
# module docstring rules that out); keep it local rather than a per-dataset protocol.
_INIT_DIRS_BY_DATASET = {
    "pickapic": ("pickapic",),
    "anime": ("danbooru/images", "danbooru/hand_crops"),
    "video-world": ("video_world/references", "video_world/targets"),
}


def _cmd_init_dirs(args: argparse.Namespace) -> None:
    from vrl.trainers.data.artifacts import default_data_root, repo_root

    data_root = args.data_root.expanduser().resolve() if args.data_root else default_data_root()
    created: list[Path] = []
    for rel in _INIT_DIRS_BY_DATASET[args.dataset]:
        path = data_root / rel
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    if args.dataset == "pickapic":
        # pickapic also caches HF downloads under the repo's data/ tree.
        cache = repo_root() / "data" / "cache" / "hf"
        cache.mkdir(parents=True, exist_ok=True)
        created.append(cache)
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "data_root": data_root.as_posix(),
                "created": [path.as_posix() for path in created],
            },
            indent=2,
            sort_keys=True,
        ),
    )


def _register_init_dirs(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "init-dirs",
        help="Create the empty artifact directories a dataset downloads into.",
    )
    parser.add_argument("dataset", choices=tuple(_INIT_DIRS_BY_DATASET))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Artifact root. Defaults to VRL_DATA_ROOT or ./data/external.",
    )
    parser.set_defaults(func=_cmd_init_dirs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pickapic.register(subparsers)
    danbooru.register(subparsers)
    videophy_i2v.register(subparsers)
    video_world.register(subparsers)
    bootstrap.register(subparsers)
    _register_init_dirs(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    main()
