"""The fixed prompt/seed grid every checkpoint comparison generates and scores.

A comparison freezes the prompt set, the seeds and the sampler so the only
thing that varies between arms is the checkpoint, writes one media file per
(arm, prompt, sample) cell, and scores every cell with the run's own reward.
The Wan Kling / HPSv3 tools and the Cosmos Kling tool share this grid; each
keeps only its scorer and its report.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from vrl.config.builders import RewardRuntimeConfig
from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import RewardConfig, parse_config
from vrl.models.families.registry import get_model_family_entry
from vrl.rewards.inference import RewardInferenceArtifact
from vrl.scripts.eval._device import resolve_eval_device, resolve_eval_dtype
from vrl.scripts.eval._sampling import resolve_eval_sampling
from vrl.scripts.eval.denoise_generation import generate_one_video, seed_for
from vrl.trainers.checkpointing import CheckpointTarget, load_resolved_run_config
from vrl.trainers.data.prompts import load_prompt_dataset_index
from vrl.utils.artifacts import sha256_file
from vrl.utils.cuda_memory import release_cuda_memory
from vrl.utils.json_files import read_jsonl
from vrl.utils.media import write_mp4

logger = logging.getLogger(__name__)

BASE_LABEL = "base"


@dataclass(frozen=True, slots=True)
class GeneratedVideo:
    """One materialized (arm, prompt, sample) cell of the grid."""

    checkpoint_label: str
    prompt_index: int
    sample_index: int
    seed: int
    prompt: str
    path: str
    size_bytes: int
    sha256: str

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        checkpoint_label: str,
        prompt_index: int,
        sample_index: int,
        seed: int,
        prompt: str,
    ) -> GeneratedVideo:
        """Record a written file with the size and digest the scorer re-checks."""

        return cls(
            checkpoint_label=checkpoint_label,
            prompt_index=prompt_index,
            sample_index=sample_index,
            seed=seed,
            prompt=prompt,
            path=str(path),
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )

    @property
    def artifact_id(self) -> str:
        return cell_id(self.checkpoint_label, self.prompt_index, self.sample_index)


def cell_id(checkpoint_label: str, prompt_index: int, sample_index: int) -> str:
    """The grid cell's identity, as the reward artifact id and the scores key."""

    return f"{checkpoint_label}-p{prompt_index:04d}-s{sample_index:02d}"


def load_wan_run_config(run_dir: Path, *, extra_overrides: Sequence[str] = ()) -> DictConfig:
    """The run's resolved config with the eval-only overrides applied.

    ``model.lora.path`` is cleared because a warm-started run records its
    donor adapter there, and every arm here supplies its own weights (base by
    disabling the adapter, checkpoints by restoring state). Compile is off so
    the comparison is not also a kernel comparison.
    """

    cfg, _ = load_resolved_run_config(
        run_dir,
        overrides=["model.lora.path=", "model.torch_compile.enable=false", *extra_overrides],
    )
    # The registry resolves the config's family alias (a run records "wan"),
    # so the canonical entry name is what this check can trust.
    entry = get_model_family_entry(str(OmegaConf.select(cfg, "model.family", default="")))
    if entry.family != "wan_2_1":
        raise ValueError(f"this evaluation requires a wan_2_1 run; got {entry.family!r}")
    return cfg


def reward_worker_config(
    cfg: DictConfig,
    component: str,
    *,
    device: str,
    default_reward_model_name: str | None = None,
) -> dict[str, Any]:
    """Project the run's own reward block into a scorer's worker config.

    Evaluation scores on training's terms, on the evaluation device, owning the
    whole GPU; parking lives in distributed.resources, never in this block.
    """

    worker_config = RewardRuntimeConfig.from_section(RewardConfig.from_cfg(cfg)).worker_config(
        component, default_reward_model_name=default_reward_model_name
    )
    worker_config["device"] = device
    return worker_config


@dataclass(frozen=True, slots=True)
class CheckpointComparison:
    """A Wan comparison ready to generate: the run, the arms and the protocol."""

    cfg: DictConfig
    entry: Any
    build: Any
    identity: dict[str, Any]
    targets: tuple[CheckpointTarget, ...]
    examples: list[Any]
    sampling: dict[str, Any]
    output_dir: Path
    samples_per_prompt: int
    base_seed: int

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        *,
        requires_trainer: str,
        extra_overrides: Sequence[str] = (),
    ) -> CheckpointComparison:
        """Resolve the run and preflight every arm before paying for the model.

        A mismatched checkpoint should fail in seconds, not after the pipeline
        is resident on the GPU, so each target's published metadata is pinned
        and checked against the runtime identity here.
        """

        if args.limit < 1 or args.samples_per_prompt < 1:
            raise ValueError("--limit and --samples-per-prompt must be >= 1")
        targets = CheckpointTarget.from_cli_values(args.checkpoint, reserved_label=BASE_LABEL)
        if not targets and args.no_base:
            raise ValueError("nothing to generate: pass --checkpoint or drop --no-base")

        cfg = load_wan_run_config(args.run_dir, extra_overrides=extra_overrides)
        root = parse_config(cfg)
        device = resolve_eval_device(args.device)
        precision = PrecisionPolicy.from_section(root.precision)
        dtype = resolve_eval_dtype(
            "auto", root, precision=precision, device=device, requires_trainer=requires_trainer
        )
        entry = get_model_family_entry(str(root.model.family))
        build = entry.resolve_model_build(
            root, device, precision=precision, parameter_dtype_override=dtype
        )
        identity = build.checkpoint_identity()
        loaded = tuple(
            CheckpointTarget.load(target.path, label=target.label) for target in targets
        )
        for target in loaded:
            target.validate_compatibility(family=entry.family, expected_model_identity=identity)

        examples = load_prompt_dataset_index(args.prompts)[: args.limit]
        if len(examples) < args.limit:
            raise ValueError(
                f"manifest has {len(examples)} prompts, fewer than --limit {args.limit}"
            )
        # The eval pins its own sampler rather than inheriting the training
        # rollout's: a GRPO rollout's SDE knobs describe exploration, not an
        # evaluation protocol (measured: inheriting denoise_mode='sde' buried
        # the weight difference under sampler noise).
        sampling = resolve_eval_sampling(cfg)
        sampling["denoise_mode"] = str(args.denoise_mode)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            cfg=cfg,
            entry=entry,
            build=build,
            identity=identity,
            targets=loaded,
            examples=examples,
            sampling=sampling,
            output_dir=args.output_dir,
            samples_per_prompt=int(args.samples_per_prompt),
            base_seed=int(args.base_seed),
        )

    @property
    def seed_formula(self) -> str:
        return "base_seed + prompt_index * samples_per_prompt + sample_index"

    def generate_arm(self, model: Any, label: str) -> list[GeneratedVideo]:
        """Generate every (prompt, sample) cell of one arm with the model as it stands."""

        import torch

        arm_dir = self.output_dir / "videos" / label
        arm_dir.mkdir(parents=True, exist_ok=True)
        produced: list[GeneratedVideo] = []
        for prompt_index, example in enumerate(self.examples):
            for sample_index in range(self.samples_per_prompt):
                seed = seed_for(
                    base_seed=self.base_seed,
                    prompt_index=prompt_index,
                    sample_index=sample_index,
                    samples_per_prompt=self.samples_per_prompt,
                )
                video = generate_one_video(
                    model, prompt=example.prompt, seed=seed, sampling=self.sampling
                )
                # A silently broken adapter yields constant frames that still
                # score; refuse here rather than spend reward GPU-hours on noise.
                if not bool(torch.isfinite(video).all()) or float(video.float().std()) < 1e-3:
                    raise RuntimeError(
                        f"degenerate video for {label} prompt {prompt_index} sample {sample_index}",
                    )
                path = arm_dir / f"p{prompt_index:04d}_s{sample_index:02d}.mp4"
                write_mp4(video, path, fps=float(self.sampling["fps"]))
                produced.append(
                    GeneratedVideo.from_file(
                        path,
                        checkpoint_label=label,
                        prompt_index=prompt_index,
                        sample_index=sample_index,
                        seed=seed,
                        prompt=example.prompt,
                    ),
                )
            logger.info("%s: %d/%d prompts", label, prompt_index + 1, len(self.examples))
        return produced


def score_video(
    model: Any,
    path: Path,
    *,
    prompt: str,
    artifact_id: str,
    score_keys: Sequence[str],
) -> dict[str, float]:
    """Score one file with a reward model and require every declared key."""

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
    missing = [key for key in score_keys if key not in scores]
    if missing:
        raise ValueError(f"{type(model).__name__} returned no {missing} for {path}")
    return {key: float(scores[key]) for key in score_keys}


def score_rows(
    output_dir: Path,
    build_model: Callable[[], Any],
    *,
    score_keys: Sequence[str],
    reuse_cached: bool = False,
) -> list[dict[str, Any]]:
    """Score every generated.jsonl row, merging ``r_<key>`` columns into it.

    The grid is the contract: a video edited or lost between generate and
    score would silently compare two different things, so each file's digest
    is re-checked. With ``reuse_cached`` a prior scores.jsonl row whose digest
    still matches is kept and the model is built only if some row needs it.
    """

    rows = read_jsonl(output_dir / "generated.jsonl")
    if not rows:
        raise ValueError(f"no generated.jsonl rows under {output_dir}")
    scores_path = output_dir / "scores.jsonl"
    cached = {
        (str(r["checkpoint_label"]), int(r["prompt_index"]), int(r["sample_index"])): r
        for r in (read_jsonl(scores_path) if reuse_cached and scores_path.is_file() else [])
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
                model = build_model()
            scores = score_video(
                model,
                path,
                prompt=str(row["prompt"]),
                artifact_id=cell_id(*key),
                score_keys=score_keys,
            )
            scored.append({**row, **{f"r_{name}": value for name, value in scores.items()}})
    finally:
        del model
        release_cuda_memory()
    return scored


__all__ = [
    "BASE_LABEL",
    "CheckpointComparison",
    "GeneratedVideo",
    "cell_id",
    "load_wan_run_config",
    "reward_worker_config",
    "score_rows",
    "score_video",
]
