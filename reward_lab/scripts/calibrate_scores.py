"""Fit, apply, or evaluate a frozen reward combination from persisted scores."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from reward_lab.annotation import export_preference_review, import_preference_review
from reward_lab.calibration import (
    apply_combination,
    evaluate_combination,
    fit_combination,
    load_preferences,
)
from reward_lab.diagnostics import join_evaluations
from vrl.rewards.evaluation import read_evaluation
from vrl.utils.artifacts import atomic_file
from vrl.utils.json_files import write_json


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
    apply = subparsers.add_parser("apply")
    apply.add_argument("--combination", required=True, type=Path)
    qualify = subparsers.add_parser("qualify")
    qualify.add_argument("--combination", required=True, type=Path)
    qualify.add_argument("--reward-config", required=True, type=Path)
    qualify.add_argument("--axis-mapping", required=True, type=Path)
    qualify.add_argument("--atol", required=True, type=float)
    qualify.add_argument("--rtol", required=True, type=float)
    prepare = subparsers.add_parser("prepare-review")
    prepare.add_argument("--pairs", required=True, type=Path)
    prepare.add_argument("--seed", required=True, type=int)
    import_review = subparsers.add_parser("import-review")
    import_review.add_argument("--review", required=True, type=Path)
    import_review.add_argument("--answers", required=True, type=Path)
    import_review.add_argument("--output", required=True, type=Path)
    for command in (fit, evaluate):
        command.add_argument("--preferences", required=True, type=Path)
    for command in (fit, evaluate, apply, prepare, qualify):
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--evaluation", type=Path)
        source.add_argument(
            "--component",
            action="append",
            metavar="NAME=DIRECTORY",
            help="Repeat to join independent evaluations; axes become NAME/axis",
        )
        command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "import-review":
        pairs = import_preference_review(
            json.loads(args.review.read_text()), json.loads(args.answers.read_text())
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with atomic_file(args.output, overwrite=False) as handle:
            for pair in pairs:
                handle.write(pair.model_dump_json() + "\n")
        return
    if args.evaluation is not None:
        evaluation = read_evaluation(args.evaluation)
    else:
        components = {}
        for spec in args.component:
            name, separator, directory = spec.partition("=")
            if not separator or not directory or name in components:
                parser.error("components must be unique NAME=DIRECTORY entries")
            components[name] = read_evaluation(Path(directory))
        evaluation = join_evaluations(components)
    if args.command == "prepare-review":
        pairs = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()]
        export_preference_review(evaluation, pairs, args.output, seed=args.seed)
        return
    if args.command == "qualify":
        import yaml

        from reward_lab.qualification import qualify_reward_deployment
        from vrl.config.builders import RewardRuntimeConfig
        from vrl.config.schema import RewardConfig

        if args.output.exists():
            raise FileExistsError("use a new output path for qualification")
        config = RewardRuntimeConfig.from_cfg(
            RewardConfig.model_validate(yaml.safe_load(args.reward_config.read_text()))
        )
        result = asyncio.run(
            qualify_reward_deployment(
                evaluation,
                json.loads(args.combination.read_text()),
                config,
                axis_mapping=json.loads(args.axis_mapping.read_text()),
                atol=args.atol,
                rtol=args.rtol,
            )
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with atomic_file(args.output, overwrite=False) as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
        return
    if args.command == "fit":
        result = fit_combination(
            evaluation,
            load_preferences(args.preferences),
            axes=args.axes,
            dimension=args.dimension,
            l2=args.l2,
            tie_margin=args.tie_margin,
        )
    elif args.command == "evaluate":
        result = evaluate_combination(
            evaluation,
            load_preferences(args.preferences),
            json.loads(args.combination.read_text()),
        )
    else:
        result = apply_combination(evaluation, json.loads(args.combination.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
