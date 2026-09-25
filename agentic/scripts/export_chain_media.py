"""Export a chain run's image states for independent reward scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic.chains import ChainRun


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="A chain run.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = ChainRun.load(args.run).export(args.output_dir)
    print(json.dumps({"run_id": result["run_id"], "sample_order": result["sample_order"]}))


if __name__ == "__main__":
    main()
