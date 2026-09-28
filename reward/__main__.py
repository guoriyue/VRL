"""``python -m reward <command>``: analyze rewards offline.

Every command reads scoring runs written by ``vrl.scripts.rewards.rescore_media``
and writes one JSON report. ``--evaluation DIR`` names one run; ``--component
NAME=DIR`` (repeatable) joins several scorers of the same samples into one run
whose axes are ``NAME/axis``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reward.analysis import Analysis
from reward.labels import Contrast, OutcomeLabel, agreement
from reward.shortcuts import build_shortcut_manifest
from reward.stress import build_stress_manifest
from vrl.rewards.evaluation import Evaluation
from vrl.rewards.sequences import EditSequenceSpec
from vrl.utils.json_files import write_json


def _load(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Evaluation:
    if getattr(args, "evaluation", None) is not None:
        return Evaluation.load(args.evaluation)
    components = {}
    for spec in args.component or ():
        name, separator, directory = spec.partition("=")
        if not separator or not directory or name in components:
            parser.error("components must be unique NAME=DIRECTORY entries")
        components[name] = Evaluation.load(Path(directory))
    if not components:
        parser.error("one of --evaluation or --component is required")
    return Analysis.join(components)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="reward", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, *, source: bool = True, output: bool = True) -> argparse.ArgumentParser:
        sub = commands.add_parser(name)
        if source:
            group = sub.add_mutually_exclusive_group()
            group.add_argument("--evaluation", type=Path)
            group.add_argument("--component", action="append", metavar="NAME=DIRECTORY")
        if output:
            sub.add_argument("--output", required=True, type=Path)
        return sub

    health = command("health")
    health.add_argument("--tie-epsilon", type=float, default=0.0)
    command("stress")
    sequence = command("sequence")
    sequence.add_argument("--spec", required=True, type=Path)
    compare = command("compare")
    compare.add_argument("--other", required=True, type=Path)
    compare.add_argument("--first-axis", required=True)
    compare.add_argument("--second-axis", required=True)
    compare.add_argument("--first-direction", type=int, choices=(-1, 1), default=1)
    compare.add_argument("--second-direction", type=int, choices=(-1, 1), default=1)
    compare.add_argument("--first-tie-epsilon", type=float, default=0.0)
    compare.add_argument("--second-tie-epsilon", type=float, default=0.0)
    paired = command("paired", source=False)
    paired.add_argument("--baseline", action="append", required=True, type=Path)
    paired.add_argument("--candidate", action="append", required=True, type=Path)
    paired.add_argument("--axis", required=True)
    paired.add_argument("--direction", type=int, choices=(-1, 1), default=1)
    paired.add_argument("--stratify-by", help="Categorical field in sample metadata")
    repeat = command("repeat", source=False)
    repeat.add_argument("evaluations", nargs="+", type=Path)
    stress_manifest = command("stress-manifest", source=False, output=False)
    stress_manifest.add_argument("--manifest", required=True, type=Path)
    stress_manifest.add_argument("--output-dir", required=True, type=Path)
    stress_manifest.add_argument("--seed", type=int, default=42)
    shortcut_manifest = command("shortcut-manifest", source=False, output=False)
    shortcut_manifest.add_argument("--manifest", required=True, type=Path)
    shortcut_manifest.add_argument("--output-dir", required=True, type=Path)
    shortcut_manifest.add_argument("--shortcuts", nargs="+")
    shortcut_manifest.add_argument("--seed", type=int, default=42)
    agree = command("agreement")
    agree.add_argument("--labels", required=True, type=Path)
    agree.add_argument("--contrasts", required=True, type=Path)
    agree.add_argument("--axis", required=True)
    agree.add_argument("--direction", type=int, choices=(-1, 1), default=1)
    agree.add_argument("--gate", type=float, default=0.85)
    agree.add_argument("--seed", type=int, default=0)
    spread = command("spread")
    spread.add_argument("--axis", required=True)
    spread.add_argument("--threshold", required=True, type=float)
    spread.add_argument("--band", nargs=2, type=float, default=(0.2, 0.6), metavar=("LOW", "HIGH"))
    spread.add_argument("--tie-epsilon", type=float, default=0.0)
    args = parser.parse_args(argv)

    if args.command == "stress-manifest":
        print(build_stress_manifest(args.manifest, args.output_dir, seed=args.seed))
        return
    if args.command == "shortcut-manifest":
        print(
            build_shortcut_manifest(
                args.manifest, args.output_dir, shortcuts=args.shortcuts, seed=args.seed
            )
        )
        return
    if args.command == "paired":
        result = Analysis.paired(
            [Evaluation.load(path) for path in args.baseline],
            [Evaluation.load(path) for path in args.candidate],
            axis=args.axis,
            direction=args.direction,
            stratify_by=args.stratify_by,
        )
    elif args.command == "repeat":
        result = Analysis.repeatability([Evaluation.load(path) for path in args.evaluations])
    else:
        evaluation = _load(args, parser)
        analysis = Analysis(evaluation)
        if args.command == "health":
            result = analysis.health(tie_epsilon=args.tie_epsilon)
        elif args.command == "stress":
            result = analysis.stress()
        elif args.command == "sequence":
            result = analysis.sequence(
                EditSequenceSpec.model_validate(json.loads(args.spec.read_text()))
            )
        elif args.command == "agreement":
            result = agreement(
                evaluation,
                OutcomeLabel.load_jsonl(args.labels),
                Contrast.load_json(args.contrasts),
                axis=args.axis,
                direction=args.direction,
                gate=args.gate,
                seed=args.seed,
            )
        elif args.command == "spread":
            result = analysis.spread(
                axis=args.axis,
                threshold=args.threshold,
                band=tuple(args.band),
                tie_epsilon=args.tie_epsilon,
            )
        else:
            result = analysis.compare_rankings(
                Evaluation.load(args.other),
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
