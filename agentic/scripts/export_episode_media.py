"""Export a validated visual episode's image states for independent reward scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic.export import export_episode_media


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = export_episode_media(json.loads(args.episode.read_text()), args.output_dir)
    print(json.dumps({"export_id": result["export_id"], "sample_order": result["sample_order"]}))


if __name__ == "__main__":
    main()
