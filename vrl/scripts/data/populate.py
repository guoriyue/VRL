"""Dataset population CLI dispatcher.

Each dataset has its own concrete script (``pickapic.py``, ``anime.py``,
``video_world.py``) mirroring ``danbooru.py``; this module only wires their
subcommands together. No generic artifact-manifest framework lives here.

    python -m vrl.scripts.data.populate <command>
"""

from __future__ import annotations

import argparse

from vrl.scripts.data import bootstrap, danbooru, pickapic, video_world


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pickapic.register(subparsers)
    danbooru.register(subparsers)
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
