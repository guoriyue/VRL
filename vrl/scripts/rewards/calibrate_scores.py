"""Fit or evaluate a frozen reward combination using explicit preference splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vrl.rewards.calibration import evaluate_combination, fit_combination, load_preferences
from vrl.rewards.diagnostics import read_evaluation
from vrl.rewards.evaluation import atomic_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--axes", nargs="+", required=True)
    fit.add_argument("--dimension", default="overall")
    fit.add_argument("--l2", type=float, default=0.1)
    fit.add_argument("--tie-margin", type=float, default=0.1)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--combination", required=True, type=Path)
    for command in (fit, evaluate):
        command.add_argument("--evaluation", required=True, type=Path)
        command.add_argument("--preferences", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    evaluation = read_evaluation(args.evaluation)
    pairs = load_preferences(args.preferences)
    if args.command == "fit":
        result = fit_combination(
            evaluation,
            pairs,
            axes=args.axes,
            dimension=args.dimension,
            l2=args.l2,
            tie_margin=args.tie_margin,
        )
    else:
        result = evaluate_combination(evaluation, pairs, json.loads(args.combination.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
