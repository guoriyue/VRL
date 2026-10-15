"""Fixed-prompt Kling VideoReward comparison across Wan 2.2 T2V LoRA checkpoints.

Same protocol as ``wan_hpsv3_checkpoint_eval``: freeze the prompt set, the
seeds and the sampler so the only thing that varies between arms is the
adapter, produce the base arm by disabling the adapter on the already-built
model, and report PAIRED deltas against it. Two differences:

* the scorer is Kling VideoReward; all four keys (``visual_quality`` /
  ``motion_quality`` / ``text_alignment`` / ``overall_reward``) are recorded
  per video so the trained objective can be read beside the other dimensions;
* ``generate`` MERGES into an existing ``generated.jsonl`` instead of
  overwriting it, so the base arm is generated once and later checkpoints are
  added to the same grid as training produces them.

    generate  base and/or checkpoints over the same prompt/seed grid -> mp4s
    score     Kling over every video in the grid -> scores.jsonl/csv + report.json
    probe     reward sanity on the base arm: repeat determinism, response to
              obvious degradations (gaussian noise, frame shuffle, frozen
              frame) and the spread across real generations

The probe answers "is this reward usable at this geometry" before GPU-hours go
into training: a dimension that a frozen or shuffled clip does NOT lower is not
measuring what its name says at 320x320x17.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from vrl.scripts.eval._checkpoint_comparison import (
    BASE_LABEL,
    CheckpointComparison,
    GeneratedVideo,
    load_wan_run_config,
    reward_worker_config,
    score_rows,
    score_video,
)
from vrl.scripts.eval._device import resolve_eval_device
from vrl.scripts.eval.score_report import (
    summarize_paired_scores,
    summarize_prompt_level,
    write_scores,
)
from vrl.trainers.checkpointing import (
    TrainingCheckpoint,
)
from vrl.utils.artifacts import sha256_file
from vrl.utils.cuda_memory import release_cuda_memory
from vrl.utils.json_files import read_jsonl, write_json, write_jsonl

logger = logging.getLogger(__name__)

REPORT_SCHEMA = "vrl.wan22_kling_checkpoint_eval/v1"
SCORE_KEYS = ("visual_quality", "motion_quality", "text_alignment", "overall_reward")
DEFAULT_BASE_SEED = 2_026_091_400
# Degradations an honest quality/motion reward must rank below the original.
PROBE_NOISE_SIGMAS = (20.0, 60.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Wan 2.2 T2V LoRA checkpoints on a fixed Kling prompt grid.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Generate (part of) the fixed prompt/seed grid.")
    generate.add_argument("--run-dir", type=Path, required=True, help="Training run directory.")
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="[LABEL=]PATH",
        help="Checkpoint to evaluate; repeatable. Bare paths take the directory name as label.",
    )
    generate.add_argument("--prompts", type=Path, required=True, help="Prompt manifest.")
    generate.add_argument("--limit", type=int, default=16, help="Prompts to take from the head.")
    generate.add_argument("--samples-per-prompt", type=int, default=2)
    generate.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    generate.add_argument("--device", default="auto")
    generate.add_argument(
        "--fps",
        type=int,
        default=16,
        help="Frame rate for every arm; always replaces the run's sampling.fps.",
    )
    generate.add_argument(
        "--denoise-mode",
        default="native",
        choices=("native", "sde"),
        help="Deterministic native sampler by default; that is what inference deploys.",
    )
    generate.add_argument("--no-base", action="store_true", help="Skip the adapter-disabled arm.")

    score = sub.add_parser("score", help="Score every generated video with Kling.")
    score.add_argument("--run-dir", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--device", default="auto")

    probe = sub.add_parser("probe", help="Reward sanity probe over the base arm videos.")
    probe.add_argument("--run-dir", type=Path, required=True)
    probe.add_argument("--output-dir", type=Path, required=True)
    probe.add_argument("--device", default="auto")
    probe.add_argument("--limit", type=int, default=16, help="Base videos to probe.")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        result = generate_grid(args)
    elif args.command == "score":
        result = score_grid(args)
    else:
        result = probe_reward(args)
    json.dump(result, sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")


# --- shared -------------------------------------------------------------------


def _build_kling(run_dir: Path, device: str) -> Any:
    """Kling built from the run's own reward block so eval scores on training's terms."""

    from vrl.rewards.models.kling_video_reward import KlingVideoRewardModel

    return KlingVideoRewardModel(
        reward_worker_config(
            load_wan_run_config(run_dir),
            "kling_video_reward",
            device=str(resolve_eval_device(device)),
            default_reward_model_name="KlingTeam/VideoReward@main",
        ),
    )


# --- generate -----------------------------------------------------------------


def generate_grid(args: argparse.Namespace) -> dict[str, Any]:
    grid_path = args.output_dir / "generated.jsonl"
    existing = read_jsonl(grid_path) if grid_path.is_file() else []
    present = {str(row["checkpoint_label"]) for row in existing}
    provenance_path = args.output_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else None

    comparison = CheckpointComparison.from_args(
        args,
        requires_trainer="Wan 2.2 Kling checkpoint evaluation",
        extra_overrides=[f"sampling.fps={int(args.fps)}"],
    )
    for target in comparison.targets:
        if target.label in present:
            raise ValueError(f"arm {target.label!r} already exists in {grid_path}")
    protocol = {
        "prompts": str(args.prompts.resolve()),
        "prompts_sha256": sha256_file(args.prompts),
        "limit": args.limit,
        "samples_per_prompt": args.samples_per_prompt,
        "base_seed": args.base_seed,
        "seed_formula": comparison.seed_formula,
        "sampling": comparison.sampling,
    }
    if provenance is not None and provenance["grid"] != protocol:
        raise ValueError(f"grid differs from the existing {provenance_path}: {provenance['grid']}")

    bundle = comparison.entry.build_rollout(
        comparison.build, expected_identity=comparison.identity
    )
    model = bundle.model.eval()
    videos: list[GeneratedVideo] = []
    try:
        if not args.no_base:
            if BASE_LABEL in present:
                raise ValueError(f"base arm already exists in {grid_path}; pass --no-base")
            with model.disable_adapter():
                videos += comparison.generate_arm(model, BASE_LABEL)
        for target in comparison.targets:
            checkpoint = TrainingCheckpoint.load(target.path)
            # Install the adapters through the family's weight-sync path: the
            # generic restore writes into rollout modules that sit behind Wan's
            # sequential-offload hooks (meta tensors) and its strict readback
            # cannot run there. load_trainable_state suspends the hooks,
            # copies the bytes and reads them back before re-arming offload.
            model.load_trainable_state(
                {
                    f"{expert}.{name}": value
                    for expert, state in checkpoint.checkpoint_state.items()
                    for name, value in state.items()
                },
            )
            videos += comparison.generate_arm(model, target.label)
    finally:
        del model, bundle
        release_cuda_memory()

    rows = existing + [asdict(video) for video in videos]
    write_jsonl(grid_path, rows)
    checkpoints = dict(provenance["checkpoints"]) if provenance else {}
    checkpoints.update(
        {
            target.label: {"path": str(target.path), "meta": target.meta}
            for target in comparison.targets
        },
    )
    write_json(
        provenance_path,
        {
            "schema": REPORT_SCHEMA,
            "run_dir": str(args.run_dir),
            "resolved_config_sha256": sha256_file(args.run_dir / "resolved_config.yaml"),
            "grid": protocol,
            "arms": sorted({str(row["checkpoint_label"]) for row in rows}),
            "checkpoints": checkpoints,
        },
    )
    return {"videos": len(videos), "arms": sorted(present | {v.checkpoint_label for v in videos})}


# --- score --------------------------------------------------------------------


def score_grid(args: argparse.Namespace) -> dict[str, Any]:
    scored = score_rows(
        args.output_dir,
        lambda: _build_kling(args.run_dir, args.device),
        score_keys=SCORE_KEYS,
        reuse_cached=True,
    )
    write_scores(scored, args.output_dir)
    summary = summarize_paired_scores(
        scored,
        score_keys=SCORE_KEYS,
        schema=REPORT_SCHEMA,
        base_label=BASE_LABEL,
    )
    report = {
        "schema": REPORT_SCHEMA,
        "scored": len(scored),
        "prompt_level": summarize_prompt_level(
            scored, score_keys=SCORE_KEYS, base_label=BASE_LABEL
        ),
        **summary,
    }
    write_json(args.output_dir / "report.json", report)
    return {k: v for k, v in report.items() if k in {"schema", "scored", "prompt_level"}}


# --- probe --------------------------------------------------------------------


def probe_reward(args: argparse.Namespace) -> dict[str, Any]:
    rows = [
        row
        for row in read_jsonl(args.output_dir / "generated.jsonl")
        if row["checkpoint_label"] == BASE_LABEL
    ][: args.limit]
    if not rows:
        raise ValueError("probe needs base-arm rows in generated.jsonl")
    model = _build_kling(args.run_dir, args.device)
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for row in rows:
            path = Path(row["path"])
            prompt = str(row["prompt"])
            cell = f"p{row['prompt_index']:04d}-s{row['sample_index']:02d}"
            record: dict[str, Any] = {"cell": cell, "prompt": prompt, "variants": {}}
            record["variants"]["original"] = score_video(
                model, path, prompt=prompt, artifact_id=cell, score_keys=SCORE_KEYS
            )
            record["variants"]["repeat"] = score_video(
                model, path, prompt=prompt, artifact_id=cell + "-r", score_keys=SCORE_KEYS
            )
            frames = np.asarray(iio.imread(path, plugin="pyav"))
            fps = float(iio.immeta(path, plugin="pyav").get("fps", 16.0))
            # Degradations an honest quality/motion reward must rank below the
            # original: gaussian noise, shuffled frames, one frozen frame.
            rng = np.random.default_rng(0)
            degradations: dict[str, np.ndarray] = {}
            for sigma in PROBE_NOISE_SIGMAS:
                noisy = frames.astype(np.float32) + rng.normal(0.0, sigma, frames.shape)
                degradations[f"noise_{int(sigma)}"] = np.clip(noisy, 0, 255).astype(np.uint8)
            degradations["shuffle"] = frames[rng.permutation(len(frames))]
            degradations["frozen"] = np.repeat(frames[:1], len(frames), axis=0)
            for name, degraded in degradations.items():
                dst = Path(tmp) / f"{cell}_{name}.mp4"
                iio.imwrite(
                    dst, degraded.astype(np.uint8), fps=fps, plugin="pyav", codec="libx264"
                )
                record["variants"][name] = score_video(
                    model, dst, prompt=prompt, artifact_id=f"{cell}-{name}", score_keys=SCORE_KEYS
                )
            records.append(record)
            logger.info("probed %s", cell)

    variants = list(records[0]["variants"])
    means = {
        v: {k: statistics.fmean(r["variants"][v][k] for r in records) for k in SCORE_KEYS}
        for v in variants
    }
    repeat_max_abs = max(
        abs(r["variants"]["original"][k] - r["variants"]["repeat"][k])
        for r in records
        for k in SCORE_KEYS
    )
    lower_rate = {
        v: {
            k: sum(r["variants"][v][k] < r["variants"]["original"][k] for r in records)
            / len(records)
            for k in SCORE_KEYS
        }
        for v in variants
        if v not in {"original", "repeat"}
    }
    spread = {
        k: {
            "std": statistics.pstdev([r["variants"]["original"][k] for r in records]),
            "min": min(r["variants"]["original"][k] for r in records),
            "max": max(r["variants"]["original"][k] for r in records),
        }
        for k in SCORE_KEYS
    }
    report = {
        "schema": REPORT_SCHEMA + "/probe",
        "videos": len(records),
        "repeat_max_abs_diff": repeat_max_abs,
        "variant_means": means,
        "degradation_scored_lower_rate": lower_rate,
        "original_spread": spread,
        "records": records,
    }
    write_json(args.output_dir / "reward_probe.json", report)
    return {k: v for k, v in report.items() if k != "records"}


if __name__ == "__main__":
    main()
