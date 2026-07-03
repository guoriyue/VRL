"""Pydantic typed boundary for merged training configs.

OmegaConf handles YAML defaults, interpolation, and CLI overrides.
Pydantic validates the fully-resolved, merged container after OmegaConf finishes.
Unknown YAML keys load fine and are reported loudly by the single whole-tree
walker in vrl.config.unknown_keys — a typo, a dead key, and a removed legacy
key all get the same treatment: one warning naming the dotted path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields as dataclass_fields
from typing import Annotated, Any, Literal

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
from vrl.config.unknown_keys import OPEN, ConfigBlock
from vrl.models.interfaces.runtime import MODEL_MEMORY_SECTIONS
from vrl.ray.resources import (
    RewardResourceConfig,
    RoleResourceConfig,
    RolloutResourceConfig,
)
from vrl.trainers.core.types import (
    DebugConfig,
    EMAConfig,
    EvalConfig,
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
        "grpo",
        "dance_grpo",
        "flow_dppo",
        "grpo_guard",
        "token_grpo",
        "token_grpo_multisegment",
        "diffusion_dpo",
        "diffusion_nft",
    ]

    # Key registry: values are validated by the algorithm dataclasses in
    # vrl/algorithms/* (build_algorithm_config), not here.
    adv_clip_max: Any = None
    add_kl_coefficient: Any = None  # flow_dppo
    advantage_scale: Any = None
    beta: Any = None
    eps: Any = None
    clip_ratio: Any = None
    flow_kl_use_dt: Any = None
    global_std: Any = None
    kl_coef: Any = None
    kl_estimator: Any = None
    kl_mask_threshold: Any = None  # flow_dppo
    kl_reward_coef: Any = None
    nft_beta: Any = None
    segment_weights: Any = None
    sft_weight: Any = None
    train_segments: Any = None
    weight_copy_decay: Any = None


# ── Data section ──────────────────────────────────────────────────────────────


class DataConfig(ConfigBase):
    loader: Literal["pickapic_preference", "prompt_manifest", "prompt_image_manifest"] | None = None
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
        # loader is optional for the prompt-* family: when omitted, derive it from
        # preprocessing.format (image_caption_jsonl is the only image-caption
        # schema; everything else is the plain prompt manifest). pickapic_preference
        # cannot be derived and must be set explicitly. Keep this rule in sync with
        # _load_prompt_examples_from_config in vrl/trainers/data/prompts.py.
        # Explicit values (enforced by the Literal type) are unchanged.
        if self.loader is None:
            fmt = (self.preprocessing or {}).get("format", "")
            self.loader = (
                "prompt_image_manifest" if fmt == "image_caption_jsonl" else "prompt_manifest"
            )
        if self.loader == "prompt_manifest":
            if not self.manifest:
                raise ValueError("config missing required field: data.manifest")
            if self.preprocessing is None:
                raise ValueError("config missing required field: data.preprocessing")
            self._validate_sampler_type()

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
            self._validate_sampler_type()

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

    def _validate_sampler_type(self) -> None:
        """Shared sampler.type check for the prompt-manifest loaders."""
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


# ── Supporting sections for cross-field validation ────────────────────────────


class SdeConfig(ConfigBase):
    """Typed rollout.sde block. ``type`` is the user-facing allow-list, replacing
    the hand-written {flow_grpo, cps} membership checks previously duplicated in the
    schema cross-validator, layout, and flow_matching. The layout request-boundary
    guard stays for over-the-wire request dicts; ``window_*`` stay permissive.

    ``type`` names the reverse-SDE log-prob distribution (flow_grpo vs cps); it is
    orthogonal to ``denoise_mode`` (native/sde), which owns the word ``sde``."""

    type: Literal["flow_grpo", "cps"]
    window_size: Any = None
    window_range: Any = None


class RolloutConfig(ConfigBase):
    # readers: math/diffusion flow_matching window + RootConfig check
    sde: SdeConfig | None = None
    noise_level: float | None = None
    # janus_pro R1 only; the sole source for final_image_policy. Validated for
    # legality in RootConfig._cross_field_validate (which requires it for that kind).
    final_image_policy: Literal["always_generate", "use_selfcheck"] | None = None
    n_samples_per_prompt: int | None = None
    prompts_per_batch: int | None = None
    # "Set the slice once" size knob: prompt groups per streamed microbatch.
    # reader: TrainerConfig.__post_init__ derives actor.gradient_accumulation_steps
    # from it (vrl/trainers/core/types.py).
    microbatch_size: int | None = None
    # Fail-fast host-RAM guard fraction for streaming accumulation (0.0 = off).
    # reader: vrl/scripts/common/online.py:_run_streaming_optimizer_update.
    host_memory_budget_fraction: float | None = None
    # reader: vrl/generation/diffusion/layout.py _parse_denoise_mode (request boundary).
    # Allowed set is the type; the layout guard stays for over-the-wire request dicts.
    denoise_mode: Literal["native", "sde"] | None = None
    max_reflect_len: Any = None
    max_text_length: Any = None
    # reader: vrl/generation/diffusion/layout.py — opt-in to storing each denoise
    # step's rollout proposal mean for trust-region replay (flow_dppo/grpo_guard).
    return_prev_sample_mean: Any = None
    # reader: vrl/generation/diffusion/layout.py — opt-in to caching the frozen
    # reference (LoRA-disabled) noise_pred at collect, so KL replay never reruns
    # the ref forward. Lossless: replay applies the same sde_step_with_logprob.
    cache_ref_noise_pred: Any = None
    same_latent: Any = None
    samples_per_chunk: Any = None
    temperature: Any = None
    torch_profiler: Annotated[Any, ConfigBlock(TorchProfilerConfig)] = None
    trajectory_storage: Annotated[Any, ConfigBlock(TrajectoryStoragePolicy)] = None


class SamplingConfig(ConfigBase):
    # reader: rollouts/collector/config.py + RootConfig cross-field check
    r1: Annotated[
        dict[str, Any] | None,
        ConfigBlock(("train_segments",)),
    ] = None
    # reader: vrl/nn/modules/ar_attention_backends.py attention_backend_name
    # (read from the request dict; default "vllm_paged"). Declared here so the
    # allowed set is the type and the key is registered (no false unknown-key warning).
    attention_backend: Literal["vllm_paged", "torch_native"] = "vllm_paged"
    # Key registry: parsed by family layout/runtime-spec extractors.
    ar_scheduler_batch_size: Any = None
    # do-CFG boolean switch (whether to apply classifier-free guidance at all);
    # distinct from guidance_scale (the float strength). Diffusion models AND it
    # with guidance_scale > 1.0 to derive their internal do_cfg flag.
    cfg: Any = None
    fps: Any = None
    guidance_scale: Any = None
    height: Any = None
    image_size: Any = None
    image_token_num: Any = None
    max_reflect_len: Any = None
    max_sequence_length: Any = None
    # AR families declare noise_level under sampling; the collector flat-merges
    # it into rollout values. Same knob as rollout.noise_level (the canonical
    # owner the cross-field validator enforces for token_grpo + nextstep_1).
    noise_level: Any = None
    num_frames: Any = None
    num_steps: Any = None
    temperature: Any = None
    width: Any = None


class ModelConfig(ConfigBase):
    """Shared model keys; family-owned keys are validated by model.family."""

    model_config = ConfigDict(extra="allow")

    family: str | None = None
    dtype: Any = None
    # readers: models/interfaces/runtime.py + family runtime.py lora blocks
    lora: Annotated[
        Any,
        ConfigBlock((
            "rank", "alpha", "path", "target_modules",
            "init_lora_weights", "dropout", "init",
        )),
    ] = None
    # model.memory sections (today only vae_decode, which self-validates strictly)
    memory: Annotated[Any, ConfigBlock(MODEL_MEMORY_SECTIONS)] = None
    path: Any = None
    torch_compile: Annotated[Any, ConfigBlock(("enable", "mode"))] = None
    use_lora: Any = None


class SD3ModelConfig(ModelConfig):
    """SD3.5 uses only the shared model keys."""

    model_config = ConfigDict(extra="ignore")


class WanModelConfig(ModelConfig):
    """Wan-specific model keys consumed by wan_2_1 runtime/model loaders."""

    model_config = ConfigDict(extra="ignore")

    boundary_ratio: Any = None
    offload_mode: Literal["none", "model", "sequential"] = "none"
    reference_image: Any = None
    task_variant: Any = None
    trainable_transformers: Any = None


class CosmosPredict2ModelConfig(ModelConfig):
    """Cosmos Predict2 Video2World model keys."""

    model_config = ConfigDict(extra="ignore")

    reference_image: Any = None


class CosmosPredict25ModelConfig(ModelConfig):
    """Cosmos Predict2.5 model keys."""

    model_config = ConfigDict(extra="ignore")

    revision: Any = None
    skip_text_encoder: Any = None


class CosmosAnimaModelConfig(ModelConfig):
    """Cosmos Anima single-file artifact paths and scheduler key."""

    model_config = ConfigDict(extra="ignore")

    qwen_tokenizer_path: Any = None
    scheduler_shift: Any = None
    t5_tokenizer_path: Any = None
    text_encoder_file: Any = None
    text_encoder_path: Any = None
    transformer_file: Any = None
    transformer_path: Any = None
    vae_file: Any = None
    vae_path: Any = None


class JanusProModelConfig(ModelConfig):
    """Janus-Pro optional model wrapper keys."""

    model_config = ConfigDict(extra="ignore")

    trust_remote_code: Any = None
    vq_latent_channels: Any = None


class NextStep1ModelConfig(ModelConfig):
    """NextStep-1 tokenizer and frozen-module keys."""

    model_config = ConfigDict(extra="ignore")

    freeze_vae: Any = None
    vae_path: Any = None


class EchoModelConfig(ModelConfig):
    """JoyAI-Echo model keys consumed by the echo runtime/model loaders.

    ``path`` is the merged Echo safetensors checkpoint; ``gemma_path`` is the
    separate Gemma-3-12B text-encoder directory (both downloaded into the run's
    checkpoints dir — weights are not vendored).
    """

    model_config = ConfigDict(extra="ignore")

    gemma_path: Any = None
    task_variant: Any = None


_model_config_classes_by_family: dict[str, type[ModelConfig]] = {
    "sd3_5": SD3ModelConfig,
    "wan": WanModelConfig,
    "wan_2_1": WanModelConfig,
    "wan_2_1_i2v": WanModelConfig,
    "wan_i2v": WanModelConfig,
    "cosmos": CosmosPredict2ModelConfig,
    "cosmos-predict2": CosmosPredict2ModelConfig,
    "cosmos_predict2": CosmosPredict2ModelConfig,
    "cosmos-predict2.5": CosmosPredict25ModelConfig,
    "cosmos_predict2_5": CosmosPredict25ModelConfig,
    "anima": CosmosAnimaModelConfig,
    "cosmos_anima": CosmosAnimaModelConfig,
    "cosmos-predict2-anima": CosmosAnimaModelConfig,
    "janus": JanusProModelConfig,
    "janus_pro": JanusProModelConfig,
    "janus_r1": JanusProModelConfig,
    "janus_pro_r1": JanusProModelConfig,
    "nextstep": NextStep1ModelConfig,
    "nextstep_1": NextStep1ModelConfig,
    "echo": EchoModelConfig,
    "joyai_echo": EchoModelConfig,
}

_model_config_variant_classes: tuple[type[ModelConfig], ...] = tuple(
    dict.fromkeys(_model_config_classes_by_family.values())
)

_model_config_blocks_by_family: dict[type[ModelConfig], ConfigBlock] = {}


def _model_config_class_for_family(family: Any) -> type[ModelConfig]:
    return _model_config_classes_by_family.get(str(family or ""), ModelConfig)


def _model_config_block_for_unknown_keys(mapping: Mapping[str, Any]) -> ConfigBlock:
    cls = _model_config_class_for_family(mapping.get("family"))
    block = _model_config_blocks_by_family.get(cls)
    if block is None:
        block = ConfigBlock(cls)
        _model_config_blocks_by_family[cls] = block
    return block


def _validate_model_config_for_family(model: ModelConfig | None) -> None:
    if model is None:
        return
    payload = model.model_dump()
    cls = _model_config_class_for_family(payload.get("family"))
    try:
        cls.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(_prefix_model_error(_extract_error_message(exc))) from exc


def _prefix_model_error(message: str) -> str:
    if message.startswith("unknown "):
        rest = message[len("unknown "):]
        if rest.startswith("model."):
            return message
        return f"unknown model.{rest}"
    if message.startswith("config missing required field: "):
        rest = message[len("config missing required field: "):]
        return f"config missing required field: model.{rest}"
    return message


# ── Section key registries (values validated by their own layers) ────────────


class TrainerSection(ConfigBase):
    """Key registry for trainer.*; values validated by vrl.config.builders."""

    entrypoint: Any = None
    total_epochs: Any = None
    save_freq: Any = None
    output_dir: Any = None
    seed: Any = None
    profile: Any = None
    resume_from: Any = None
    resume_strict: Any = None
    debug: Annotated[Any, ConfigBlock(DebugConfig)] = None
    eval: Annotated[Any, ConfigBlock(EvalConfig)] = None
    precision_drift_guard: Annotated[Any, ConfigBlock(PrecisionDriftGuardConfig)] = None
    precision_correction: Annotated[Any, ConfigBlock(PrecisionCorrectionConfig)] = None
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
    gradient_checkpointing: Any = None  # off | full | selective (or bool: true=full, false=off)
    timestep_fraction: Any = None
    timestep_selection: Any = None  # strided | random (DanceGRPO)
    # offline DPO entrypoint (vrl/scripts/diffusion/wan_2_1/train_dpo.py)
    prediction_type: Any = None
    scale_lr: Any = None
    train_batch_size: Any = None
    use_adafactor: Any = None


class FSDPConfig(ConfigBase):
    """distributed.training.fsdp: the FSDP2 knobs ``build_strategy`` reads.

    Only the fields a reader consumes are declared (the same philosophy as
    ``TrainingSection``): ``vrl/trainers/strategy.py`` build_strategy +
    ``vrl/trainers/fsdp.py`` read ``mesh`` / ``precision_policy`` /
    ``reshard_after_forward``. The remaining FSDP2 knobs from
    SPRINT_multi_gpu_training.md §3 (activation_checkpointing, cpu_offload,
    state_dict, process-group backend/init) land here when their readers do —
    declaring an unread knob is a user-facing no-op footgun.
    """

    # 1D ZeRO-3 over the whole world; 2D HSDP is the multi-node follow-on.
    mesh: list[str] = Field(default_factory=lambda: ["dp_shard"])
    # actor -> MixedPrecisionPolicy(param=bf16, reduce=fp32); none -> full precision.
    # Named precision_policy (not mixed_precision) to avoid colliding with the
    # training-forward dtype `train_precision` (fp32/bf16/fp16); this is a
    # param/reduce policy, not a dtype.
    precision_policy: Literal["actor", "none"] = "actor"
    # True = re-gather params after forward (ZeRO-3, lowest memory).
    reshard_after_forward: bool = True


class DDPConfig(ConfigBase):
    """distributed.training.ddp: the one DDP knob ``build_strategy`` reads.

    DDP replicates the full module per rank (right when the model fits on one
    card, e.g. a 2B transformer + LoRA), so unlike ``fsdp`` there are no shard/mesh
    knobs. ``find_unused_parameters`` is the only DDP-specific lever a reader
    consumes; declare nothing the strategy doesn't read.
    """

    # DDP's reducer expects every requires_grad param to get a grad each backward;
    # set True only if a LoRA/grad-checkpoint forward conditionally skips a wrapped
    # branch (slower — adds an extra graph traversal).
    find_unused_parameters: bool = False


class TrainingSection(ConfigBase):
    """distributed.training: how the trainer process maps onto GPUs.

    Only the fields a reader actually consumes are declared: ``strategy`` (the
    single source of truth for the allowed backends — resource validation and the
    training context dispatch on it; the Literal is the only allow-list), the
    ``num_nodes``/``gpus_per_node`` topology the fsdp/ddp context cross-checks
    against ``WORLD_SIZE``, and the per-strategy knob block (``fsdp`` / ``ddp``)
    read by ``build_strategy``.
    """

    strategy: Literal["single_process", "fsdp", "ddp"] = "single_process"
    num_nodes: int = 1
    gpus_per_node: int = 1
    fsdp: FSDPConfig | None = None
    ddp: DDPConfig | None = None


class RolloutWorkerSection(ConfigBase):
    """distributed.rollout: per-worker runtime knobs.

    reader: vrl/generation/ray/config.py RayGenerationConfig.from_cfg (worker
    runtime knobs). Release scheduling and colocation are NOT declared here:
    colocation lives in distributed.resources.rollout.gpu_pool=trainer +
    memory_fraction (mirrors reward.gpu_pool), and release scheduling is derived
    from GPU topology. chunk_placement_strategy is a user-facing allow-list
    Literal: RayGenerationConfig is a plain dataclass whose annotations do not
    enforce, so this typed boundary is where a bad value is rejected (the runtime
    ChunkPlacementPolicy guard stays the wire-boundary check). sync_trainable_state
    is a plain on/off: True keeps rollout workers resynced to the trained policy
    (the syncer flattens whatever is trainable — lora or full-param), False
    disables the weight syncer.
    """

    cpus_per_worker: float = 1.0
    max_inflight_chunks_per_worker: int = 1
    chunk_placement_strategy: Literal["round_robin", "dynamic"] = "round_robin"
    sync_trainable_state: bool = True
    # Opt-in single-worker pipelined rollout. Read by RayGenerationConfig.from_cfg;
    # only takes effect with exactly one rollout worker and more than one planned
    # chunk. Best paired with the largest samples_per_chunk that keeps 2 chunks
    # resident.
    pipelined: bool = False


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
                # memory_fraction is a public input key (resident-colocation GPU cap)
                # parsed in vrl/ray/resources.py _parse_rollout_pool into the flat
                # rollout_gpu_memory_fraction; it is not a stored field on
                # RolloutResourceConfig, so name it here alongside the derived keys.
                "rollout": ConfigBlock(
                    (
                        *(f.name for f in dataclass_fields(RolloutResourceConfig)),
                        "memory_fraction",
                    ),
                ),
                # gpu_pool is the field; share_with_rollout is the legacy compat
                # key mapped to it at parse time (vrl/ray/resources.py
                # _parse_reward_gpu_pool), so both are accepted here.
                "reward": ConfigBlock(
                    (
                        *(f.name for f in dataclass_fields(RewardResourceConfig)),
                        "share_with_rollout",
                    ),
                ),
            },
        ),
    ] = None
    # reader: vrl/generation/ray/config.py RayGenerationConfig.from_cfg (worker
    # runtime knobs). chunk_placement_strategy / sync_trainable_state Literals reject
    # bad values here at parse time. Colocation lives in resources.rollout.gpu_pool.
    rollout: RolloutWorkerSection | None = None
    # reader: vrl/ray/resources.py reward runtime block. Ray reward actor-pool
    # knobs were removed; reward release is derived from topology and heavy
    # in-process rewards use reward.kwargs.<name>.sleep_offload.
    reward: Annotated[
        Any,
        ConfigBlock(("cpus_per_worker",)),
    ] = None
    # readers: vrl/trainers/distributed.py resolve_training_context (rank/device)
    # + vrl/ray/resources.py strategy-aware trainer GPU validation
    training: TrainingSection | None = None


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
    model: Annotated[
        ModelConfig | None,
        ConfigBlock(
            ModelConfig,
            select=_model_config_block_for_unknown_keys,
            variants=_model_config_variant_classes,
        ),
    ] = None
    sampling: SamplingConfig | None = None
    # per-component production gates; contract checks live in
    # vrl/config/validation.py validate_production_* (raw-cfg checks)
    production: Annotated[dict[str, Any] | None, OPEN] = None
    trainer: TrainerSection | None = None
    actor: ActorSection | None = None
    distributed: DistributedSection | None = None
    # reader: vrl/config/precision.py (scalar form skips the block walk).
    # `rollout` is the experimental fp8/fp4-rollout split (rollout != train).
    precision: Annotated[Any, ConfigBlock(("train", "rollout", "math", "frozen", "rollout_recipe"))] = None
    # reader: vrl/scripts/diffusion/cosmos/train.py
    cosmos: Annotated[Any, ConfigBlock(("reference_mode",))] = None

    @model_validator(mode="after")
    def _cross_field_validate(self) -> RootConfig:
        _validate_model_config_for_family(self.model)

        algo = self.algorithm
        if algo is None:
            return self

        kind = algo.kind
        rollout = self.rollout

        # grpo / diffusion_nft: SDE type must be sde or cps
        # grpo / diffusion_nft require an sde block; sde.type membership is now
        # enforced by the SdeConfig Literal (for every kind, not just these two —
        # the runtime layout guard remains the wire-boundary check).
        if kind in {
            "grpo",
            "dance_grpo",
            "flow_dppo",
            "grpo_guard",
            "diffusion_nft",
        } and (rollout is None or rollout.sde is None):
            raise ValueError("config missing required field: rollout.sde.type")

        # token_grpo: nextstep_1 family requires rollout.noise_level
        if kind == "token_grpo":
            model_family = self.model.family if self.model else None
            if model_family == "nextstep_1" and (
                rollout is None or rollout.noise_level is None
            ):
                raise ValueError("config missing required field: rollout.noise_level")

        # token_grpo_multisegment: janus_pro only; final_image_policy from rollout
        if kind == "token_grpo_multisegment":
            model_family = (self.model.family or "") if self.model else ""
            if model_family != "janus_pro":
                raise ValueError(
                    "token_grpo_multisegment currently requires model.family=janus_pro"
                )
            # Single source: final_image_policy lives on rollout only. Validate it
            # here as a legality check (the collector reads rollout.final_image_policy).
            policy = (rollout.final_image_policy or "") if rollout else ""
            if policy not in {"always_generate", "use_selfcheck"}:
                raise ValueError(
                    "rollout.final_image_policy must be 'always_generate' or 'use_selfcheck'"
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
