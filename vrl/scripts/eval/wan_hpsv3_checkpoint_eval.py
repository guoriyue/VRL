"""Fixed-prompt HPSv3 comparison across Wan2.1 T2V LoRA checkpoints.

Online GRPO rotates prompts every update, so ``metrics.csv`` reward is a moving
target: a batch of hard prompts and a degraded policy look identical there.
This evaluation freezes the prompt set, the seeds, and the sampling protocol so
the only thing that varies between arms is the adapter, and reports the PAIRED
delta against the base arm — the statistic that removes prompt difficulty from
the comparison.

Two subcommands so a reward-side failure never costs the generation, which is
the expensive half (81 frames x 20 steps per video):

    generate  base + each checkpoint over the same prompt/seed grid -> mp4s
    score     HPSv3 over those mp4s -> scores.jsonl/csv + report.json

The base arm is produced by disabling the adapter on the already-built model
rather than by a second build. That matters for correctness, not just cost: a
training run's ``resolved_config.yaml`` may carry a warm-start
``model.lora.path`` (run50 warm-started from an earlier run), so rebuilding
from that config without overriding the path would silently make "base" the
warm-start adapter instead of the base model.

All three HPSv3 keys are recorded per video. ``top_frame_mean`` is the training
objective, but a rising ``top_frame_mean`` beside a falling ``frame_min`` is the
"best frames improve while the worst rot" reward-hacking signature, and only the
whole-video keys can show it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vrl.scripts.eval._checkpoint_comparison import (
    BASE_LABEL,
    CheckpointComparison,
    GeneratedVideo,
    load_wan_run_config,
    reward_worker_config,
    score_rows,
)
from vrl.scripts.eval._device import resolve_eval_device
from vrl.scripts.eval.score_report import summarize_paired_scores, write_scores
from vrl.trainers.checkpointing import (
    RESOLVED_CONFIG_NAME,
    TrainingCheckpoint,
)
from vrl.utils.artifacts import sha256_file
from vrl.utils.cuda_memory import release_cuda_memory
from vrl.utils.json_files import write_json, write_jsonl

logger = logging.getLogger(__name__)

REPORT_SCHEMA = "vrl.wan_hpsv3_checkpoint_eval/v1"
# Every key the HPSv3 worker returns. The selected training objective is
# top_frame_mean; the other two exist to expose reward hacking (see module doc).
SCORE_KEYS = ("top_frame_mean", "frame_mean", "frame_min")
DEFAULT_BASE_SEED = 2_026_082_500


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Wan2.1 T2V LoRA checkpoints on a fixed HPSv3 prompt grid.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Generate the fixed prompt/seed grid.")
    generate.add_argument("--run-dir", type=Path, required=True, help="Training run directory.")
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="[LABEL=]PATH",
        help="Checkpoint to evaluate; repeatable. Bare paths take the directory name as label.",
    )
    generate.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="Held-out prompt manifest (.txt or .jsonl), disjoint from the training set.",
    )
    generate.add_argument("--limit", type=int, default=24, help="Prompts to take from the head.")
    generate.add_argument("--samples-per-prompt", type=int, default=2)
    generate.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    generate.add_argument("--device", default="auto")
    generate.add_argument(
        "--denoise-mode",
        default="native",
        choices=("native", "sde"),
        help="Sampler for the comparison. Default 'native' (deterministic) because that "
        "is what inference deploys; the run's own rollout SDE settings are exploration "
        "machinery, not an evaluation protocol.",
    )
    generate.add_argument(
        "--no-base",
        action="store_true",
        help="Skip the adapter-disabled arm (paired scoring then needs a base elsewhere).",
    )

    score = sub.add_parser("score", help="Score generated videos with HPSv3.")
    score.add_argument("--run-dir", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    result = generate_grid(args) if args.command == "generate" else score_grid(args)
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def generate_grid(args: argparse.Namespace) -> dict[str, Any]:
    comparison = CheckpointComparison.from_args(
        args, requires_trainer="Wan HPSv3 checkpoint evaluation"
    )
    bundle = comparison.entry.build_rollout(comparison.build)
    model = bundle.model.eval()
    videos: list[GeneratedVideo] = []
    try:
        # Base FIRST, while no checkpoint has been restored. Ordering is the
        # guarantee: once a checkpoint is loaded, "adapter disabled" no longer
        # means the base model for any later arm.
        if not args.no_base:
            with model.disable_adapter():
                videos += comparison.generate_arm(model, BASE_LABEL)
        for target in comparison.targets:
            TrainingCheckpoint.load(target.path).restore_model(
                bundle=bundle,
                family=comparison.entry.family,
                expected_model_identity=comparison.identity,
            )
            videos += comparison.generate_arm(model, target.label)
    finally:
        del model, bundle
        release_cuda_memory()

    write_jsonl(args.output_dir / "generated.jsonl", [asdict(video) for video in videos])
    provenance = {
        "schema": REPORT_SCHEMA,
        "run_dir": str(args.run_dir),
        "resolved_config_sha256": sha256_file(args.run_dir / RESOLVED_CONFIG_NAME),
        "prompts": str(args.prompts),
        "limit": args.limit,
        "samples_per_prompt": args.samples_per_prompt,
        "base_seed": args.base_seed,
        "seed_formula": comparison.seed_formula,
        "sampling": comparison.sampling,
        "arms": [BASE_LABEL] * (not args.no_base)
        + [target.label for target in comparison.targets],
        "checkpoints": {
            target.label: {"path": str(target.path), "meta": target.meta}
            for target in comparison.targets
        },
    }
    write_json(args.output_dir / "provenance.json", provenance)
    return {"videos": len(videos), "arms": provenance["arms"], "output_dir": str(args.output_dir)}


# --- score --------------------------------------------------------------------


def score_grid(args: argparse.Namespace) -> dict[str, Any]:
    from vrl.rewards.models.hpsv3 import HPSv3Model

    # The run's own reward block, so eval scores on training's terms.
    worker_config = reward_worker_config(
        load_wan_run_config(args.run_dir),
        "hpsv3",
        device=str(resolve_eval_device(args.device)),
        default_reward_model_name="MizzenAI/HPSv3@main",
    )
    scored = score_rows(args.output_dir, lambda: HPSv3Model(worker_config), score_keys=SCORE_KEYS)
    write_scores(scored, args.output_dir)
    summary = summarize_paired_scores(
        scored,
        score_keys=SCORE_KEYS,
        schema=REPORT_SCHEMA,
        base_label=BASE_LABEL,
    )
    report = {"schema": REPORT_SCHEMA, "scored": len(scored), **summary}
    write_json(args.output_dir / "report.json", report)
    return report


if __name__ == "__main__":
    main()
