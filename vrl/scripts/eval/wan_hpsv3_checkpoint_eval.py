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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from vrl.config.builders import RewardRuntimeConfig
from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import parse_config
from vrl.models.checkpoint_identity import resolve_checkpoint_model_identity
from vrl.models.families.registry import get_model_family_entry
from vrl.rewards.inference import RewardInferenceArtifact
from vrl.scripts.eval._device import resolve_eval_device, resolve_eval_dtype
from vrl.scripts.eval._sampling import resolve_eval_sampling
from vrl.scripts.eval.denoise_generation import generate_one_video, seed_for
from vrl.scripts.eval.score_report import summarize_paired_scores, write_scores
from vrl.trainers.checkpointing import (
    RESOLVED_CONFIG_NAME,
    CheckpointTarget,
    TrainingCheckpoint,
    load_resolved_run_config,
    restore_model_checkpoint,
    validate_checkpoint_meta_compatibility,
)
from vrl.trainers.data.prompts import load_prompt_dataset_index
from vrl.utils.artifacts import sha256_file
from vrl.utils.cuda_memory import release_cuda_memory
from vrl.utils.json_files import read_jsonl, write_json, write_jsonl
from vrl.utils.media import write_mp4

logger = logging.getLogger(__name__)

REPORT_SCHEMA = "vrl.wan_hpsv3_checkpoint_eval/v1"
BASE_LABEL = "base"
# Every key the HPSv3 worker returns. The selected training objective is
# top_frame_mean; the other two exist to expose reward hacking (see module doc).
SCORE_KEYS = ("top_frame_mean", "frame_mean", "frame_min")
DEFAULT_BASE_SEED = 2_026_082_500


@dataclass(frozen=True, slots=True)
class GeneratedVideo:
    checkpoint_label: str
    prompt_index: int
    sample_index: int
    seed: int
    prompt: str
    path: str
    size_bytes: int
    sha256: str


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


# --- generate -----------------------------------------------------------------


def generate_grid(args: argparse.Namespace) -> dict[str, Any]:
    if args.limit < 1 or args.samples_per_prompt < 1:
        raise ValueError("--limit and --samples-per-prompt must be >= 1")
    targets = CheckpointTarget.from_cli_values(args.checkpoint, reserved_label=BASE_LABEL)
    if not targets and args.no_base:
        raise ValueError("nothing to generate: pass --checkpoint or drop --no-base")

    cfg = _load_run_config(args.run_dir)
    root = parse_config(cfg)
    device = resolve_eval_device(args.device)
    precision = PrecisionPolicy.from_section(root.precision)
    dtype = resolve_eval_dtype(
        "auto",
        root,
        precision=precision,
        device=device,
        requires_trainer="Wan HPSv3 checkpoint evaluation",
    )
    entry = get_model_family_entry(str(root.model.family))
    build = entry.resolve_model_build(
        root,
        device,
        precision=precision,
        parameter_dtype_override=dtype,
    )
    identity = resolve_checkpoint_model_identity(build)

    # Preflight every checkpoint against the runtime identity BEFORE paying for
    # model construction: a mismatched arm should fail in seconds, not after the
    # pipeline is resident on the GPU. CLI parsing only resolved label/path;
    # pinning the published metadata happens here, once, for every arm.
    targets = [CheckpointTarget.load(target.path, label=target.label) for target in targets]
    for target in targets:
        validate_checkpoint_meta_compatibility(
            target.meta,
            family=entry.family,
            expected_model_identity=identity,
            strict=True,
        )

    examples = load_prompt_dataset_index(args.prompts)[: args.limit]
    if len(examples) < args.limit:
        raise ValueError(f"manifest has {len(examples)} prompts, fewer than --limit {args.limit}")
    # The eval pins its own sampler rather than inheriting the training rollout's.
    # A GRPO rollout's SDE knobs describe EXPLORATION: this recipe makes exactly one
    # of 20 steps stochastic (sde.window_size=1) and runs the rest as ODE, but the
    # window never reaches the sampling dict, so inheriting denoise_mode='sde' would
    # inject full noise at every step — a regime neither training nor deployment
    # uses, and one that buried the weight difference under sampler noise (measured:
    # absolute HPSv3 fell from +3.6 to -7.3 for the SAME base model).
    sampling = resolve_eval_sampling(cfg)
    sampling["denoise_mode"] = str(args.denoise_mode)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = entry.build_rollout(build)
    model = bundle.model.eval()
    videos: list[GeneratedVideo] = []
    try:
        # Base FIRST, while no checkpoint has been restored. Ordering is the
        # guarantee: once a checkpoint is loaded, "adapter disabled" no longer
        # means the base model for any later arm.
        if not args.no_base:
            with model.disable_adapter():
                videos += _generate_arm(model, BASE_LABEL, examples, sampling, args)
        for target in targets:
            restore_model_checkpoint(
                TrainingCheckpoint.load(target.path),
                bundle=bundle,
                family=entry.family,
                expected_model_identity=identity,
                strict=True,
            )
            videos += _generate_arm(model, target.label, examples, sampling, args)
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
        "seed_formula": "base_seed + prompt_index * samples_per_prompt + sample_index",
        "sampling": sampling,
        "arms": [BASE_LABEL] * (not args.no_base) + [target.label for target in targets],
        "checkpoints": {
            target.label: {"path": str(target.path), "meta": target.meta} for target in targets
        },
    }
    write_json(args.output_dir / "provenance.json", provenance)
    return {"videos": len(videos), "arms": provenance["arms"], "output_dir": str(args.output_dir)}


def _generate_arm(
    model: Any,
    label: str,
    examples: list[Any],
    sampling: dict[str, Any],
    args: argparse.Namespace,
) -> list[GeneratedVideo]:
    arm_dir = args.output_dir / "videos" / label
    arm_dir.mkdir(parents=True, exist_ok=True)
    produced: list[GeneratedVideo] = []
    for prompt_index, example in enumerate(examples):
        for sample_index in range(args.samples_per_prompt):
            seed = seed_for(
                base_seed=args.base_seed,
                prompt_index=prompt_index,
                sample_index=sample_index,
                samples_per_prompt=args.samples_per_prompt,
            )
            video = generate_one_video(
                model,
                prompt=example.prompt,
                seed=seed,
                sampling=sampling,
            )
            # A silently broken adapter yields constant frames that still score.
            # Refuse here rather than spend reward GPU-hours scoring noise.
            if not bool(torch.isfinite(video).all()) or float(video.float().std()) < 1e-3:
                raise RuntimeError(
                    f"degenerate video for {label} prompt {prompt_index} sample {sample_index}",
                )
            path = arm_dir / f"p{prompt_index:04d}_s{sample_index:02d}.mp4"
            write_mp4(video, path, fps=float(sampling["fps"]))
            produced.append(
                GeneratedVideo(
                    checkpoint_label=label,
                    prompt_index=prompt_index,
                    sample_index=sample_index,
                    seed=seed,
                    prompt=example.prompt,
                    path=str(path),
                    size_bytes=path.stat().st_size,
                    sha256=sha256_file(path),
                ),
            )
        logger.info("%s: %d/%d prompts", label, prompt_index + 1, len(examples))
    return produced


# --- score --------------------------------------------------------------------


def score_grid(args: argparse.Namespace) -> dict[str, Any]:
    from vrl.rewards.models.hpsv3 import HPSv3Model

    rows = read_jsonl(args.output_dir / "generated.jsonl")
    if not rows:
        raise ValueError(f"no generated.jsonl rows under {args.output_dir}")
    device = resolve_eval_device(args.device)
    worker_config = _hpsv3_worker_config(_load_run_config(args.run_dir), device=device)

    model = HPSv3Model(worker_config)
    scored: list[dict[str, Any]] = []
    try:
        for row in rows:
            path = Path(row["path"])
            # The grid is the contract: a video edited or lost between generate
            # and score would silently compare two different things.
            if sha256_file(path) != row["sha256"]:
                raise ValueError(f"video changed since generation: {path}")
            scores = model(
                RewardInferenceArtifact(
                    artifact_id=f"{row['checkpoint_label']}-p{row['prompt_index']:04d}"
                    f"-s{row['sample_index']:02d}",
                    sample_id=f"{row['checkpoint_label']}-p{row['prompt_index']:04d}"
                    f"-s{row['sample_index']:02d}",
                    path=str(path),
                    prompt=str(row["prompt"]),
                    size_bytes=int(row["size_bytes"]),
                    sha256=str(row["sha256"]),
                ),
            )
            missing = [key for key in SCORE_KEYS if key not in scores]
            if missing:
                raise ValueError(f"HPSv3 returned no {missing} for {path}")
            scored.append(
                {**row, **{f"r_{key}": float(scores[key]) for key in SCORE_KEYS}},
            )
    finally:
        del model
        release_cuda_memory()

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


def _hpsv3_worker_config(cfg: DictConfig, *, device: torch.device) -> dict[str, Any]:
    """Project the run's own reward block so eval scores on training's terms."""

    worker_config = RewardRuntimeConfig.from_cfg(cfg).worker_config(
        "hpsv3",
        default_reward_model_name="MizzenAI/HPSv3@main",
    )
    worker_config["device"] = str(device)
    return worker_config


# --- shared -------------------------------------------------------------------


def _load_run_config(run_dir: Path) -> DictConfig:
    """Load the run's resolved config with the eval-only overrides applied.

    ``model.lora.path`` is cleared because a warm-started run records its
    donor adapter there, and every arm here supplies its own weights (base by
    disabling the adapter, checkpoints by restoring state). Compile is off so
    the comparison is not also a kernel comparison.
    """

    cfg, _ = load_resolved_run_config(
        run_dir,
        overrides=["model.lora.path=", "model.torch_compile.enable=false"],
    )
    # The registry resolves the config's family alias (a run records "wan"),
    # so the canonical entry name is what this check can trust.
    entry = get_model_family_entry(str(OmegaConf.select(cfg, "model.family", default="")))
    if entry.family != "wan_2_1":
        raise ValueError(f"this evaluation requires a wan_2_1 run; got {entry.family!r}")
    return cfg
