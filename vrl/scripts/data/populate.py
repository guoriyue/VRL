"""Dataset population CLI dispatcher.

Each dataset has its own concrete script (``pickapic.py``, ``danbooru.py``,
``video_world.py``, ``bootstrap.py``); this module only wires their subcommands
together. No generic artifact-manifest framework lives here.

    python -m vrl.scripts.data.populate <command>

Commands: pickapic, anime-prompts, anime-safety-prompts, anime-positives,
anime-fetch-images, videophy-i2v, video-world-bridge, for-experiment.
"""

from __future__ import annotations

import argparse

from vrl.scripts.data import bootstrap, danbooru, pickapic, video_world, videophy_i2v


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pickapic.register(subparsers)
    danbooru.register(subparsers)
    videophy_i2v.register(subparsers)
    video_world.register(subparsers)
    bootstrap.register(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    main()
