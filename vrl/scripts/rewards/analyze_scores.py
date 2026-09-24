"""Report reward health and ranking disagreements without loading models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vrl.rewards.diagnostics import (
    compare_paired_outputs,
    compare_rankings,
    health_report,
    read_evaluation,
    repeatability_report,
    stress_report,
)
from vrl.rewards.sequences import EditSequenceSpec, sequence_report
from vrl.utils.json_files import write_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    health = subparsers.add_parser("health")
    health.add_argument("evaluation", type=Path)
    health.add_argument("--tie-epsilon", type=float, default=0.0)
    stress = subparsers.add_parser("stress")
    stress.add_argument("evaluation", type=Path)
    compare = subparsers.add_parser("compare")
    compare.add_argument("first", type=Path)
    compare.add_argument("second", type=Path)
    compare.add_argument("--first-axis", required=True)
    compare.add_argument("--second-axis", required=True)
    compare.add_argument("--first-direction", type=int, choices=(-1, 1), default=1)
    compare.add_argument("--second-direction", type=int, choices=(-1, 1), default=1)
    compare.add_argument("--first-tie-epsilon", type=float, default=0.0)
    compare.add_argument("--second-tie-epsilon", type=float, default=0.0)
    paired = subparsers.add_parser("paired")
    paired.add_argument("--baseline", action="append", required=True, type=Path)
    paired.add_argument("--candidate", action="append", required=True, type=Path)
    paired.add_argument("--axis", required=True)
    paired.add_argument("--direction", type=int, choices=(-1, 1), default=1)
    paired.add_argument("--stratify-by", help="Categorical field in sample metadata")
    repeat = subparsers.add_parser("repeat")
    repeat.add_argument("evaluations", nargs="+", type=Path)
    sequence = subparsers.add_parser("sequence")
    sequence.add_argument("evaluation", type=Path)
    sequence.add_argument("--spec", required=True, type=Path)
    for command in (health, compare, stress, paired, repeat, sequence):
        command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "health":
        result = health_report(read_evaluation(args.evaluation), tie_epsilon=args.tie_epsilon)
    elif args.command == "stress":
        result = stress_report(read_evaluation(args.evaluation))
    elif args.command == "sequence":
        result = sequence_report(
            read_evaluation(args.evaluation),
            EditSequenceSpec.model_validate(json.loads(args.spec.read_text())),
        )
    elif args.command == "repeat":
        paths = [path.resolve() for path in args.evaluations]
        if len(set(paths)) != len(paths):
            raise ValueError("repeatability requires distinct scoring directories")
        result = repeatability_report([read_evaluation(path) for path in paths])
        result["evaluation_directories"] = [str(path) for path in paths]
    elif args.command == "paired":
        result = compare_paired_outputs(
            [read_evaluation(path) for path in args.baseline],
            [read_evaluation(path) for path in args.candidate],
            axis=args.axis,
            direction=args.direction,
            stratify_by=args.stratify_by,
        )
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
    write_json(args.output, result)


if __name__ == "__main__":
    main()
