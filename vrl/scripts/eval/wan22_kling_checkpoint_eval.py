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
import torch
from omegaconf import DictConfig, OmegaConf

from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import parse_config
from vrl.models.checkpoint_identity import resolve_checkpoint_model_identity
from vrl.models.families.registry import get_model_family_entry
from vrl.rewards.inference import RewardInferenceArtifact
from vrl.scripts.eval._device import resolve_eval_device, resolve_eval_dtype
from vrl.scripts.eval._sampling import resolve_eval_sampling
from vrl.scripts.eval.denoise_generation import generate_one_video, seed_for
from vrl.scripts.eval.score_report import summarize_paired_scores, write_scores
from vrl.scripts.eval.wan_hpsv3_checkpoint_eval import BASE_LABEL, GeneratedVideo
from vrl.trainers.checkpointing import (
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
        "--fps", type=int, default=16, help="sampling.fps when the run omits it."
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


def _load_run_config(run_dir: Path, *, extra_overrides: list[str] = ()) -> DictConfig:
    """The run's resolved config with the eval-only overrides (see the HPSv3 twin)."""

    cfg, _ = load_resolved_run_config(
        run_dir,
        overrides=["model.lora.path=", "model.torch_compile.enable=false", *extra_overrides],
    )
    entry = get_model_family_entry(str(OmegaConf.select(cfg, "model.family", default="")))
    if entry.family != "wan_2_1":
        raise ValueError(f"this evaluation requires a wan family run; got {entry.family!r}")
    return cfg


def _kling_worker_config(cfg: DictConfig, *, device: torch.device) -> dict[str, Any]:
    """Project the run's own Kling block so eval scores on training's terms."""

    selected = OmegaConf.select(cfg, "reward.kwargs.kling_video_reward", default={})
    reward_cfg = OmegaConf.to_container(selected, resolve=True) or {}
    if not isinstance(reward_cfg, dict):
        raise ValueError("reward.kwargs.kling_video_reward must be a mapping")
    worker_config = dict(reward_cfg.get("worker_config") or {})
    worker_config.setdefault("reward_model_name", "KlingTeam/VideoReward@main")
    # Training-only lifecycle knob; the eval owns the whole GPU.
    worker_config.pop("memory_parking_mode", None)
    worker_config["device"] = str(device)
    return worker_config


def _build_kling(run_dir: Path, device: str) -> Any:
    from vrl.rewards.models.kling_video_reward import KlingVideoRewardModel

    return KlingVideoRewardModel(
        _kling_worker_config(_load_run_config(run_dir), device=resolve_eval_device(device)),
    )


def _score_path(model: Any, path: Path, prompt: str, artifact_id: str) -> dict[str, float]:
    scores = model(
        RewardInferenceArtifact(
            artifact_id=artifact_id,
            sample_id=artifact_id,
            path=str(path),
            prompt=prompt,
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
        ),
    )
    missing = [key for key in SCORE_KEYS if key not in scores]
    if missing:
        raise ValueError(f"Kling returned no {missing} for {path}")
    return {key: float(scores[key]) for key in SCORE_KEYS}


# --- generate -----------------------------------------------------------------


def generate_grid(args: argparse.Namespace) -> dict[str, Any]:
    if args.limit < 1 or args.samples_per_prompt < 1:
        raise ValueError("--limit and --samples-per-prompt must be >= 1")
    targets = CheckpointTarget.from_cli_values(args.checkpoint, reserved_label=BASE_LABEL)
    if not targets and args.no_base:
        raise ValueError("nothing to generate: pass --checkpoint or drop --no-base")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = args.output_dir / "generated.jsonl"
    existing = read_jsonl(grid_path) if grid_path.is_file() else []
    present = {str(row["checkpoint_label"]) for row in existing}
    provenance_path = args.output_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else None

    cfg = _load_run_config(args.run_dir, extra_overrides=[f"sampling.fps={int(args.fps)}"])
    root = parse_config(cfg)
    device = resolve_eval_device(args.device)
    precision = PrecisionPolicy.from_section(root.precision)
    dtype = resolve_eval_dtype(
        "auto",
        root,
        precision=precision,
        device=device,
        requires_trainer="Wan 2.2 Kling checkpoint evaluation",
    )
    entry = get_model_family_entry(str(root.model.family))
    build = entry.resolve_model_build(
        root,
        device,
        precision=precision,
        parameter_dtype_override=dtype,
    )
    identity = resolve_checkpoint_model_identity(build)

    targets = [CheckpointTarget.load(target.path, label=target.label) for target in targets]
    for target in targets:
        if target.label in present:
            raise ValueError(f"arm {target.label!r} already exists in {grid_path}")
        validate_checkpoint_meta_compatibility(
            target.meta,
            family=entry.family,
            expected_model_identity=identity,
            strict=True,
        )

    examples = load_prompt_dataset_index(args.prompts)[: args.limit]
    if len(examples) < args.limit:
        raise ValueError(f"manifest has {len(examples)} prompts, fewer than --limit {args.limit}")
    sampling = resolve_eval_sampling(root)
    sampling["denoise_mode"] = str(args.denoise_mode)
    grid = {
        "prompts": str(args.prompts.resolve()),
        "prompts_sha256": sha256_file(args.prompts),
        "limit": args.limit,
        "samples_per_prompt": args.samples_per_prompt,
        "base_seed": args.base_seed,
        "seed_formula": "base_seed + prompt_index * samples_per_prompt + sample_index",
        "sampling": sampling,
    }
    if provenance is not None and provenance["grid"] != grid:
        raise ValueError(f"grid differs from the existing {provenance_path}: {provenance['grid']}")

    bundle = entry.build_rollout(build)
    model = bundle.model.eval()
    videos: list[GeneratedVideo] = []
    try:
        if not args.no_base:
            if BASE_LABEL in present:
                raise ValueError(f"base arm already exists in {grid_path}; pass --no-base")
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

    rows = existing + [asdict(video) for video in videos]
    write_jsonl(grid_path, rows)
    checkpoints = dict(provenance["checkpoints"]) if provenance else {}
    checkpoints.update(
        {target.label: {"path": str(target.path), "meta": target.meta} for target in targets},
    )
    write_json(
        provenance_path,
        {
            "schema": REPORT_SCHEMA,
            "run_dir": str(args.run_dir),
            "resolved_config_sha256": sha256_file(args.run_dir / "resolved_config.yaml"),
            "grid": grid,
            "arms": sorted({str(row["checkpoint_label"]) for row in rows}),
            "checkpoints": checkpoints,
        },
    )
    return {"videos": len(videos), "arms": sorted(present | {v.checkpoint_label for v in videos})}


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
            video = generate_one_video(model, prompt=example.prompt, seed=seed, sampling=sampling)
            # A silently broken adapter yields constant frames that still score.
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
    rows = read_jsonl(args.output_dir / "generated.jsonl")
    if not rows:
        raise ValueError(f"no generated.jsonl rows under {args.output_dir}")
    scores_path = args.output_dir / "scores.jsonl"
    cached = {
        (str(r["checkpoint_label"]), int(r["prompt_index"]), int(r["sample_index"])): r
        for r in (read_jsonl(scores_path) if scores_path.is_file() else [])
    }
    model = None
    scored: list[dict[str, Any]] = []
    try:
        for row in rows:
            key = (
                str(row["checkpoint_label"]),
                int(row["prompt_index"]),
                int(row["sample_index"]),
            )
            hit = cached.get(key)
            if hit is not None and hit.get("sha256") == row["sha256"]:
                scored.append(hit)
                continue
            path = Path(row["path"])
            if sha256_file(path) != row["sha256"]:
                raise ValueError(f"video changed since generation: {path}")
            if model is None:
                model = _build_kling(args.run_dir, args.device)
            cell = f"{key[0]}-p{key[1]:04d}-s{key[2]:02d}"
            scores = _score_path(model, path, str(row["prompt"]), cell)
            scored.append({**row, **{f"r_{k}": v for k, v in scores.items()}})
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
    report = {
        "schema": REPORT_SCHEMA,
        "scored": len(scored),
        "prompt_level": _prompt_level_paired(scored),
        **summary,
    }
    write_json(args.output_dir / "report.json", report)
    return {k: v for k, v in report.items() if k in {"schema", "scored", "prompt_level"}}


def _prompt_level_paired(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Paired deltas with the PROMPT as the unit: seeds of one prompt are correlated.

    Per arm and key: average the samples of each prompt, difference against the
    base arm's same-prompt average, then report mean / stderr / win rate over
    prompts. This is the statistic a learning claim rests on.
    """

    by_arm: dict[str, dict[int, dict[str, list[float]]]] = {}
    for row in rows:
        arm = by_arm.setdefault(str(row["checkpoint_label"]), {})
        prompt = arm.setdefault(int(row["prompt_index"]), {key: [] for key in SCORE_KEYS})
        for key in SCORE_KEYS:
            prompt[key].append(float(row[f"r_{key}"]))
    base = by_arm[BASE_LABEL]
    out: dict[str, Any] = {}
    for label, prompts in by_arm.items():
        if label == BASE_LABEL:
            continue
        if set(prompts) != set(base):
            raise ValueError(f"arm {label!r} covers different prompts than the base arm")
        per_key: dict[str, Any] = {}
        for key in SCORE_KEYS:
            deltas = [
                statistics.fmean(prompts[p][key]) - statistics.fmean(base[p][key])
                for p in sorted(base)
            ]
            n = len(deltas)
            per_key[key] = {
                "prompts": n,
                "base_mean": statistics.fmean(statistics.fmean(base[p][key]) for p in base),
                "arm_mean": statistics.fmean(statistics.fmean(prompts[p][key]) for p in base),
                "delta_mean": statistics.fmean(deltas),
                "delta_stderr": statistics.stdev(deltas) / n**0.5 if n > 1 else float("nan"),
                "win_rate": sum(d > 0 for d in deltas) / n,
            }
        out[label] = per_key
    return out


# --- probe --------------------------------------------------------------------


def _read_frames(path: Path) -> tuple[np.ndarray, float]:
    frames = iio.imread(path, plugin="pyav")
    meta = iio.immeta(path, plugin="pyav")
    return np.asarray(frames), float(meta.get("fps", 16.0))


def _write_frames(frames: np.ndarray, fps: float, dst: Path) -> None:
    iio.imwrite(dst, frames.astype(np.uint8), fps=fps, plugin="pyav", codec="libx264")


def _degradations(frames: np.ndarray) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    out: dict[str, np.ndarray] = {}
    for sigma in PROBE_NOISE_SIGMAS:
        noisy = frames.astype(np.float32) + rng.normal(0.0, sigma, frames.shape)
        out[f"noise_{int(sigma)}"] = np.clip(noisy, 0, 255).astype(np.uint8)
    out["shuffle"] = frames[rng.permutation(len(frames))]
    out["frozen"] = np.repeat(frames[:1], len(frames), axis=0)
    return out


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
            record["variants"]["original"] = _score_path(model, path, prompt, cell)
            record["variants"]["repeat"] = _score_path(model, path, prompt, cell + "-r")
            frames, fps = _read_frames(path)
            for name, degraded in _degradations(frames).items():
                dst = Path(tmp) / f"{cell}_{name}.mp4"
                _write_frames(degraded, fps, dst)
                record["variants"][name] = _score_path(model, dst, prompt, f"{cell}-{name}")
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
