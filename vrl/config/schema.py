"""Pydantic typed boundary for merged training configs.

OmegaConf handles YAML defaults, interpolation, and CLI overrides.
Pydantic validates the fully-resolved, merged container after OmegaConf finishes.
Unknown YAML keys load fine and are reported loudly by the single whole-tree
walker in vrl.config.unknown_keys — a typo, a dead key, and a removed legacy
key all get the same treatment: one warning naming the dotted path.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vrl.config.unknown_keys import OPEN, ConfigBlock
from vrl.generation.ray.config import RayGenerationConfig
from vrl.ray.resources import (
    RewardResourceConfig,
    RoleResourceConfig,
    RolloutResourceConfig,
)
from vrl.trainers.core.types import (
    DebugConfig,
    EMAConfig,
    OptimConfig,
    PrecisionDriftGuardConfig,
    RolloutOrchestrationConfig,
)
from vrl.trajectory.storage import TrajectoryStoragePolicy
from vrl.utils.profiling import TorchProfilerConfig


class ConfigBase(BaseModel):
    """Shared typed-boundary base. Field declarations double as the known-key
    registry consumed by vrl.config.unknown_keys (the single whole-tree
    unknown-key reporter); unknown keys are tolerated here and reported there.
    """

    model_config = ConfigDict(extra="ignore")




# ── Reward section ────────────────────────────────────────────────────────────


class RewardConfig(ConfigBase):
    # reward names are user-chosen — open by design
    components: Annotated[dict[str, Any], OPEN]
    # each reward's kwargs contract is owned and validated by the reward class
    # itself at construction (vrl/rewards/), same as model families — the
    # config layer does not duplicate per-reward knowledge
    kwargs: Annotated[dict[str, Any], OPEN] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_reward(self) -> RewardConfig:
        # All kwargs entries must be mappings (or null)
        for name, sub in self.kwargs.items():
            if sub is not None and not isinstance(sub, dict):
                raise ValueError(
                    f"reward.kwargs.{name} must be a mapping, got {type(sub).__name__}",
                )

        # Per-component: weight must be numeric and >= 0
        for name, weight_raw in self.components.items():
            try:
                weight = float(weight_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"reward.components.{name} must be numeric, got {weight_raw!r}",
                ) from exc
            if weight < 0:
                raise ValueError(f"reward.components.{name} must be >= 0, got {weight}")
            if weight == 0:
                continue
        return self


# ── Algorithm section ─────────────────────────────────────────────────────────


class AlgorithmConfig(ConfigBase):
    kind: Literal[
        "grpo", "token_grpo", "token_grpo_multisegment", "diffusion_dpo", "diffusion_nft"
    ]

    # Key registry: values are validated by the algorithm dataclasses in
    # vrl/algorithms/* (build_algorithm_config), not here.
    adv_clip_max: Any = None
    advantage_high: Any = None
    advantage_low: Any = None
    beta: Any = None
    eps: Any = None
    eps_clip: Any = None
    flow_kl_use_dt: Any = None
    global_std: Any = None
    init_kl_coef: Any = None
    kl_beta: Any = None
    kl_estimator: Any = None
    kl_reward: Any = None
    mask_key: Any = None
    nft_beta: Any = None
    segment_weights: Any = None
    sft_weight: Any = None
    train_segments: Any = None
    uncentralized_training: Any = None
    weight_copy_decay: Any = None


# ── Data section ──────────────────────────────────────────────────────────────


class DataConfig(ConfigBase):
    loader: Literal["pickapic_preference", "prompt_manifest", "prompt_image_manifest"]
    manifest: str | None = None
    eval_manifest: str | None = None
    # readers: _validate_data + loader tooling
    preprocessing: Annotated[
        dict[str, Any] | None,
        ConfigBlock((
            "resolution", "random_crop", "horizontal_flip",
            "format", "image_field", "caption_field", "media_type",
            "conditioning", "metadata_schema", "target_text",
        )),
    ] = None
    sampler: Annotated[
        dict[str, Any] | None,
        ConfigBlock(("type", "shuffle", "drop_last", "dataloader_num_workers")),
    ] = None
    dataset_name: str | None = None
    split: str | None = None
    cache_dir: str | None = None
    max_train_samples: Any = None
    task_type: str | None = None
    # Key registry: consumed by data/eval tooling, not validated here.
    allow_absolute_artifact_paths: Any = None
    artifact_data_root: Any = None
    source: Any = None
    source_report: Any = None

    @model_validator(mode="after")
    def _validate_data(self) -> DataConfig:
        # loader validity already enforced by the Literal field type
        if self.loader == "prompt_manifest":
            if not self.manifest:
                raise ValueError("config missing required field: data.manifest")
            if self.preprocessing is None:
                raise ValueError("config missing required field: data.preprocessing")
            sampler = self.sampler or {}
            sampler_type = str(sampler.get("type", "")) if "type" in sampler else ""
            if not sampler_type:
                raise ValueError("config missing required field: data.sampler.type")
            valid_samplers = {"random_without_replacement", "sequential_window"}
            if sampler_type not in valid_samplers:
                expected = " / ".join(sorted(valid_samplers))
                raise ValueError(
                    f"unknown data.sampler.type={sampler_type!r}; expected {expected}"
                )

        if self.loader == "prompt_image_manifest":
            if not self.manifest:
                raise ValueError("config missing required field: data.manifest")
            if not self.eval_manifest:
                raise ValueError("config missing required field: data.eval_manifest")
            if self.preprocessing is None:
                raise ValueError("config missing required field: data.preprocessing")
            for field in ("format", "image_field", "caption_field", "media_type", "conditioning"):
                if field not in self.preprocessing:
                    raise ValueError(
                        f"config missing required field: data.preprocessing.{field}"
                    )
            sampler = self.sampler or {}
            sampler_type = str(sampler.get("type", "")) if "type" in sampler else ""
            if not sampler_type:
                raise ValueError("config missing required field: data.sampler.type")
            valid_samplers = {"random_without_replacement", "sequential_window"}
            if sampler_type not in valid_samplers:
                expected = " / ".join(sorted(valid_samplers))
                raise ValueError(
                    f"unknown data.sampler.type={sampler_type!r}; expected {expected}"
                )

        if self.loader == "pickapic_preference":
            for field in ("dataset_name", "split", "cache_dir"):
                # Allow empty strings (matches require()'s semantics — only None/absent is invalid)
                if getattr(self, field) is None:
                    raise ValueError(f"config missing required field: data.{field}")
            if self.preprocessing is None:
                raise ValueError("config missing required field: data.preprocessing")
            for field in ("resolution", "random_crop", "horizontal_flip"):
                if field not in self.preprocessing:
                    raise ValueError(
                        f"config missing required field: data.preprocessing.{field}"
                    )
            sampler = self.sampler or {}
            for field in ("shuffle", "drop_last", "dataloader_num_workers"):
                if field not in sampler:
                    raise ValueError(
                        f"config missing required field: data.sampler.{field}"
                    )

        return self


# ── Supporting sections for cross-field validation ────────────────────────────


class RolloutConfig(ConfigBase):
    # readers: math/diffusion flow_matching window + RootConfig check
    sde: Annotated[
        dict[str, Any] | None,
        ConfigBlock(("type", "window_size", "window_range")),
    ] = None
    noise_level: float | None = None
    final_image_policy: str | None = None
    n_samples_per_prompt: int | None = None
    rollout_batch_size: int | None = None
    # Key registry: validated by their reader layers (generation/trainers).
    # reader: generation/ray/launcher.py compile override
    denoise_compile: Annotated[Any, ConfigBlock(("enable", "mode"))] = None
    denoise_mode: Any = None
    max_reflect_len: Any = None
    max_text_length: Any = None
    same_latent: Any = None
    sample_batch_size: Any = None
    temperature: Any = None
    torch_profiler: Annotated[Any, ConfigBlock(TorchProfilerConfig)] = None
    trajectory_storage: Annotated[Any, ConfigBlock(TrajectoryStoragePolicy)] = None


class SamplingConfig(ConfigBase):
    # reader: rollouts/collector/config.py + RootConfig cross-field check
    r1: Annotated[
        dict[str, Any] | None,
        ConfigBlock(("final_image_policy", "train_segments")),
    ] = None
    # Key registry: parsed by family layout/runtime-spec extractors.
    ar_scheduler_batch_size: Any = None
    cfg: Any = None
    cfg_scale: Any = None
    cfg_weight: Any = None
    fps: Any = None
    guidance_scale: Any = None
    height: Any = None
    image_size: Any = None
    image_token_num: Any = None
    max_reflect_len: Any = None
    max_sequence_length: Any = None
    noise_level: Any = None
    num_flow_steps: Any = None
    num_frames: Any = None
    num_steps: Any = None
    temperature: Any = None
    width: Any = None


class ModelConfig(ConfigBase):
    family: str | None = None
    # Key registry: consumed by family runtime loaders.
    dtype: Any = None
    enable_model_cpu_offload: Any = None
    freeze_aligner: Any = None
    freeze_image_head: Any = None
    freeze_vae: Any = None
    freeze_vision_encoder: Any = None
    freeze_vq: Any = None
    # readers: models/interfaces/runtime.py + family runtime.py lora blocks
    lora: Annotated[
        Any,
        ConfigBlock((
            "rank", "alpha", "path", "target_modules",
            "init_lora_weights", "dropout", "init",
        )),
    ] = None
    # vae_decode self-validates strictly; frozen_offload: sd3_5 train entrypoint
    memory: Annotated[Any, ConfigBlock(("vae_decode", "frozen_offload"))] = None
    path: Any = None
    qwen_tokenizer_path: Any = None
    reference_image: Any = None
    revision: Any = None
    scheduler_shift: Any = None
    skip_text_encoder: Any = None
    t5_tokenizer_path: Any = None
    task_variant: Any = None
    text_encoder_file: Any = None
    torch_compile: Annotated[Any, ConfigBlock(("enable", "mode"))] = None
    transformer_file: Any = None
    use_lora: Any = None
    vae_file: Any = None
    vae_path: Any = None


# ── Section key registries (values validated by their own layers) ────────────


class TrainerSection(ConfigBase):
    """Key registry for trainer.*; values validated by vrl.config.builders."""

    entrypoint: Any = None
    total_epochs: Any = None
    save_freq: Any = None
    log_freq: Any = None
    output_dir: Any = None
    seed: Any = None
    profile: Any = None
    resume_from: Any = None
    resume_strict: Any = None
    debug: Annotated[Any, ConfigBlock(DebugConfig)] = None
    precision_drift_guard: Annotated[Any, ConfigBlock(PrecisionDriftGuardConfig)] = None
    # continuous sub-block nests automatically from the dataclass field type
    rollout_orchestration: Annotated[Any, ConfigBlock(RolloutOrchestrationConfig)] = None
    torch_profiler: Annotated[Any, ConfigBlock(TorchProfilerConfig)] = None
    # offline DPO entrypoint (vrl/scripts/diffusion/wan_2_1/train_dpo.py)
    checkpointing_steps: Any = None
    log_interval: Any = None
    max_train_steps: Any = None


class ActorSection(ConfigBase):
    """Key registry for actor.*; values validated by vrl.config.builders."""

    optim: Annotated[Any, ConfigBlock(OptimConfig)] = None
    ema: Annotated[Any, ConfigBlock(EMAConfig)] = None
    max_norm: Any = None
    ppo_epochs: Any = None
    gradient_accumulation_steps: Any = None
    drop_zero_advantage: Any = None
    gradient_checkpointing: Any = None
    timestep_fraction: Any = None
    # offline DPO entrypoint (vrl/scripts/diffusion/wan_2_1/train_dpo.py)
    prediction_type: Any = None
    scale_lr: Any = None
    train_batch_size: Any = None
    use_adafactor: Any = None


class DistributedSection(ConfigBase):
    """Key registry for distributed.*; values validated by vrl.ray.resources."""

    # reader: vrl/ray/resources.py resolve_distributed_resources
    resources: Annotated[
        Any,
        ConfigBlock(
            ("visible_devices", "trainer", "rollout", "reward",
             "allow_overlap", "cross_node"),
            {
                "trainer": ConfigBlock(RoleResourceConfig),
                "rollout": ConfigBlock(RolloutResourceConfig),
                "reward": ConfigBlock(RewardResourceConfig),
            },
        ),
    ] = None
    # RayGenerationConfig is the typed consumer of this block
    rollout: Annotated[Any, ConfigBlock(RayGenerationConfig)] = None
    # reader: vrl/ray/resources.py reward runtime block
    reward: Annotated[
        Any,
        ConfigBlock(("cpus_per_worker", "placement_strategy",
                     "max_inflight_batches", "release_after_score")),
    ] = None


# ── Root config ───────────────────────────────────────────────────────────────


class RootConfig(ConfigBase):
    """Top-level typed boundary for all training config sections.

    Sections with leaf validation are modelled; trainer/actor/distributed are
    key registries (values validated by builders / ray.resources); precision
    is parsed by vrl.config.precision; cosmos by the cosmos entrypoint.
    Anything else warns via ConfigBase.
    """

    algorithm: AlgorithmConfig | None = None
    data: DataConfig | None = None
    reward: RewardConfig | None = None
    rollout: RolloutConfig | None = None
    model: ModelConfig | None = None
    sampling: SamplingConfig | None = None
    # per-component production gates; contract checks live in
    # vrl/config/validation.py validate_production_* (raw-cfg checks)
    production: Annotated[dict[str, Any] | None, OPEN] = None
    trainer: TrainerSection | None = None
    actor: ActorSection | None = None
    distributed: DistributedSection | None = None
    # reader: vrl/config/precision.py (scalar form skips the block walk)
    precision: Annotated[Any, ConfigBlock(("forward", "math", "frozen"))] = None
    # reader: vrl/scripts/diffusion/cosmos/train.py
    cosmos: Annotated[Any, ConfigBlock(("reference_mode",))] = None

    @model_validator(mode="after")
    def _cross_field_validate(self) -> RootConfig:
        algo = self.algorithm
        if algo is None:
            return self

        kind = algo.kind
        rollout = self.rollout

        # grpo / diffusion_nft: SDE type must be sde or cps
        if kind in {"grpo", "diffusion_nft"}:
            sde = rollout.sde if rollout else None
            sde_type = str(sde.get("type", "")) if isinstance(sde, dict) else ""
            if sde_type not in {"sde", "cps"}:
                raise ValueError("rollout.sde.type must be 'sde' or 'cps'")

        # token_grpo: nextstep_1 family requires rollout.noise_level
        if kind == "token_grpo":
            model_family = self.model.family if self.model else None
            if model_family == "nextstep_1" and (
                rollout is None or rollout.noise_level is None
            ):
                raise ValueError("config missing required field: rollout.noise_level")

        # token_grpo_multisegment: janus_pro only, final_image_policy must match sampling
        if kind == "token_grpo_multisegment":
            model_family = (self.model.family or "") if self.model else ""
            if model_family != "janus_pro":
                raise ValueError(
                    "token_grpo_multisegment currently requires model.family=janus_pro"
                )
            final_image_policy = (rollout.final_image_policy or "") if rollout else ""
            if final_image_policy not in {"always_generate", "use_selfcheck"}:
                raise ValueError(
                    "rollout.final_image_policy must be 'always_generate' or 'use_selfcheck'"
                )
            sampling_r1 = (self.sampling.r1 or {}) if self.sampling else {}
            sampling_final_policy = str(sampling_r1.get("final_image_policy", ""))
            if sampling_final_policy != final_image_policy:
                raise ValueError(
                    "sampling.r1.final_image_policy must match rollout.final_image_policy"
                )

        return self

# ── Parse boundary ────────────────────────────────────────────────────────────


def _extract_error_message(exc: ValidationError) -> str:
    """Extract a clean ValueError message from a Pydantic ValidationError."""
    first = exc.errors(include_url=False)[0]
    error_type = first["type"]
    msg = first["msg"]
    loc = ".".join(str(p) for p in first["loc"])
    # Missing required field — remap to repo-standard message format
    if error_type == "missing":
        return f"config missing required field: {loc}"
    # Literal enum mismatch — reformat to "unknown {loc}={input!r}; expected ..."
    if error_type == "literal_error":
        input_val = first.get("input", "")
        expected = msg.replace("Input should be", "expected")
        return f"unknown {loc}={input_val!r}; {expected}"
    # ValueError raised inside a validator — strip Pydantic's "Value error, " prefix
    if msg.startswith("Value error, "):
        return msg[len("Value error, "):]
    return msg


def parse_config(cfg: DictConfig) -> RootConfig:
    """Validate a fully-merged, resolved DictConfig through the typed schema.

    OmegaConf resolves interpolations and enforces ??? missing-value semantics;
    Pydantic validates structure, enum discriminators, and cross-field rules.
    """
    try:
        raw = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    except MissingMandatoryValue as exc:
        missing_path = getattr(exc, "full_key", None) or str(exc)
        raise ValueError(f"config missing required field: {missing_path}") from exc

    if not isinstance(raw, dict):
        raise ValueError("config must be a top-level mapping")

    try:
        return RootConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(_extract_error_message(exc)) from exc


__all__ = [
    "AlgorithmConfig",
    "DataConfig",
    "ModelConfig",
    "RewardConfig",
    "RolloutConfig",
    "RootConfig",
    "SamplingConfig",
    "parse_config",
]
