"""``python -m reward <command>``: analyze, calibrate and qualify rewards offline.

Every command reads scoring runs written by ``vrl.scripts.rewards.rescore_media``
and writes one JSON report. ``--evaluation DIR`` names one run; ``--component
NAME=DIR`` (repeatable) joins several scorers of the same samples into one run
whose axes are ``NAME/axis``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from reward.analysis import Analysis
from reward.calibration import Calibration, PreferencePair
from reward.stress import build_stress_manifest
from vrl.rewards.evaluation import Evaluation
from vrl.rewards.sequences import EditSequenceSpec
from vrl.utils.artifacts import atomic_file
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
    fit = command("fit")
    fit.add_argument("--preferences", required=True, type=Path)
    fit.add_argument("--axes", nargs="+", required=True)
    fit.add_argument("--dimension", default="overall")
    fit.add_argument("--l2", type=float, default=0.1)
    fit.add_argument("--tie-margin", type=float, default=0.1)
    evaluate = command("evaluate")
    evaluate.add_argument("--preferences", required=True, type=Path)
    evaluate.add_argument("--combination", required=True, type=Path)
    apply = command("apply")
    apply.add_argument("--combination", required=True, type=Path)
    qualify = command("qualify")
    qualify.add_argument("--combination", required=True, type=Path)
    qualify.add_argument("--reward-config", required=True, type=Path)
    qualify.add_argument("--axis-mapping", required=True, type=Path)
    qualify.add_argument("--atol", required=True, type=float)
    qualify.add_argument("--rtol", required=True, type=float)
    review_export = command("review-export")
    review_export.add_argument("--pairs", required=True, type=Path)
    review_export.add_argument("--seed", required=True, type=int)
    review_import = command("review-import", source=False)
    review_import.add_argument("--review", required=True, type=Path)
    review_import.add_argument("--answers", required=True, type=Path)
    stress_manifest = command("stress-manifest", source=False, output=False)
    stress_manifest.add_argument("--manifest", required=True, type=Path)
    stress_manifest.add_argument("--output-dir", required=True, type=Path)
    stress_manifest.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.command == "stress-manifest":
        print(build_stress_manifest(args.manifest, args.output_dir, seed=args.seed))
        return
    if args.command == "review-import":
        pairs = PreferencePair.from_review(
            json.loads(args.review.read_text()), json.loads(args.answers.read_text())
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with atomic_file(args.output, overwrite=False) as handle:
            for pair in pairs:
                handle.write(pair.model_dump_json() + "\n")
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
        analysis, calibration = Analysis(evaluation), Calibration(evaluation)
        if args.command == "health":
            result = analysis.health(tie_epsilon=args.tie_epsilon)
        elif args.command == "stress":
            result = analysis.stress()
        elif args.command == "sequence":
            result = analysis.sequence(
                EditSequenceSpec.model_validate(json.loads(args.spec.read_text()))
            )
        elif args.command == "compare":
            result = analysis.compare_rankings(
                Evaluation.load(args.other),
                first_axis=args.first_axis,
                second_axis=args.second_axis,
                first_direction=args.first_direction,
                second_direction=args.second_direction,
                first_tie_epsilon=args.first_tie_epsilon,
                second_tie_epsilon=args.second_tie_epsilon,
            )
        elif args.command == "fit":
            result = calibration.fit(
                PreferencePair.load_jsonl(args.preferences),
                axes=args.axes,
                dimension=args.dimension,
                l2=args.l2,
                tie_margin=args.tie_margin,
            )
        elif args.command == "evaluate":
            result = calibration.evaluate(
                PreferencePair.load_jsonl(args.preferences),
                json.loads(args.combination.read_text()),
            )
        elif args.command == "apply":
            result = calibration.apply(json.loads(args.combination.read_text()))
        elif args.command == "review-export":
            pairs = [
                json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()
            ]
            result = calibration.review_packet(pairs, args.output, seed=args.seed)
            print(json.dumps(result))
            return
        else:
            import yaml

            from vrl.config.builders import RewardRuntimeConfig
            from vrl.config.schema import RewardConfig
            from vrl.rewards.deployment import RewardDeployment

            if args.output.exists():
                raise FileExistsError("use a new output path for qualification")
            config = RewardRuntimeConfig.from_cfg(
                RewardConfig.model_validate(yaml.safe_load(args.reward_config.read_text()))
            )
            deployment = asyncio.run(
                RewardDeployment.qualify(
                    evaluation,
                    json.loads(args.combination.read_text()),
                    config,
                    axis_mapping=json.loads(args.axis_mapping.read_text()),
                    atol=args.atol,
                    rtol=args.rtol,
                )
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            deployment.write(args.output)
            return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
