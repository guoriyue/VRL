"""Score existing media without loading a generation model or trainer."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from vrl.rewards.evaluation import Evaluation, ScoringConfig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path, help="Standalone scoring YAML")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    import yaml

    config = ScoringConfig.model_validate(yaml.safe_load(args.config.read_text()))
    evaluation = asyncio.run(
        Evaluation.score(args.manifest, config, args.output_dir, resume=args.resume)
    )
    print(json.dumps({"run_id": evaluation.run_id, **evaluation.summary}, sort_keys=True))


if __name__ == "__main__":
    main()
