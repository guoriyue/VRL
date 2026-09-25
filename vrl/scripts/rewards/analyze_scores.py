"""Report reward health and ranking disagreements without loading models."""

from __future__ import annotations

import argparse
from pathlib import Path

from vrl.rewards.diagnostics import compare_rankings, health_report, read_evaluation
from vrl.rewards.evaluation import atomic_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    health = subparsers.add_parser("health")
    health.add_argument("evaluation", type=Path)
    health.add_argument("--tie-epsilon", type=float, default=0.0)
    compare = subparsers.add_parser("compare")
    compare.add_argument("first", type=Path)
    compare.add_argument("second", type=Path)
    compare.add_argument("--first-axis", required=True)
    compare.add_argument("--second-axis", required=True)
    compare.add_argument("--first-direction", type=int, choices=(-1, 1), default=1)
    compare.add_argument("--second-direction", type=int, choices=(-1, 1), default=1)
    compare.add_argument("--first-tie-epsilon", type=float, default=0.0)
    compare.add_argument("--second-tie-epsilon", type=float, default=0.0)
    for command in (health, compare):
        command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "health":
        result = health_report(read_evaluation(args.evaluation), tie_epsilon=args.tie_epsilon)
    else:
        result = compare_rankings(
            read_evaluation(args.first),
            read_evaluation(args.second),
            first_axis=args.first_axis,
            second_axis=args.second_axis,
            first_direction=args.first_direction,
            second_direction=args.second_direction,
            first_tie_epsilon=args.first_tie_epsilon,
            second_tie_epsilon=args.second_tie_epsilon,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
