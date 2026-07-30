"""Pydantic typed boundary for merged training configs.

OmegaConf handles YAML defaults, interpolation, and CLI overrides.
Pydantic validates the fully-resolved, merged container after OmegaConf finishes.
Unknown YAML keys load fine and are reported loudly by the single whole-tree
walker in vrl.config.unknown_keys — a typo, a dead key, and a removed legacy
key all get the same treatment: one warning naming the dotted path.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from dataclasses import fields as dataclass_fields
from typing import Annotated, Any, ClassVar, Literal, TypeVar, get_args, get_type_hints

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue
from pydantic import (
    ConfigDict,
    Field,
    SerializeAsAny,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from vrl.config.algorithm import algorithm_config_class, resolve_kl_reward_coef
from vrl.config.base import ConfigBase
from vrl.config.data import DataLoaderName, resolve_data_loader
from vrl.config.model_schema import ModelSection
from vrl.config.precision import PrecisionConfig
from vrl.config.reward_inference import parse_reward_inference_config
from vrl.config.sampling_schema import SamplingSection
from vrl.config.unknown_keys import OPEN, ConfigBlock
from vrl.families.names import normalize_model_family
from vrl.families.registry import FAMILY_REGISTRY, get_model_family_entry
from vrl.generation.execution.types import ChunkPlacementStrategy
from vrl.ray.resources import (
    RewardResourceConfig,
    RoleResourceConfig,
    RolloutResourceConfig,
)
from vrl.trainers.data.prompt_sampler import PromptSamplingStrategy
from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig
from vrl.trajectory.storage import TrajectoryStoragePolicy
from vrl.utils.config import import_from_path
from vrl.utils.profiling import TorchProfilerConfig

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
            if isinstance(sub, dict):
                parse_reward_inference_config(
                    sub.get("inference"),
                    context=f"reward.kwargs.{name}.inference",
                )

        # Zero keeps a scorer observation-only: it is still computed and logged
        # but contributes nothing to the optimization reward.
        for name, weight_raw in self.components.items():
            try:
                weight = float(weight_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"reward.components.{name} must be numeric, got {weight_raw!r}",
                ) from exc
            if weight < 0:
                raise ValueError(f"reward.components.{name} must be >= 0, got {weight}")
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

    # The only algorithm hyper-parameter the cross-field validator reads.
    # Every other key is derived from, and validated against, the runtime
    # dataclass selected by ``kind``.
    sft_weight: Any = None
    # Collector-owned diffusion reward-shaping coefficient. Token trajectories
    # do not carry the per-step KL tensor needed to consume a positive value.
    kl_reward_coef: float | None = None

    @field_validator("kl_reward_coef", mode="before")
    @classmethod
    def _validate_kl_reward_coef(cls, value: object | None) -> float | None:
        if value is None:
            return None
        return resolve_kl_reward_coef(value)


@functools.cache
def _algorithm_config_block(cls: type[Any]) -> ConfigBlock:
    # kind selects the dataclass; kl_reward_coef is collector-owned and its
    # trajectory compatibility is validated at the root boundary below.
    known = {field.name for field in dataclass_fields(cls)} | {"kind", "kl_reward_coef"}
    return ConfigBlock(known)


def _algorithm_config_block_for_unknown_keys(mapping: Mapping[str, Any]) -> ConfigBlock:
    kind = str(mapping.get("kind") or "")
    try:
        cls = algorithm_config_class(kind)
    except ValueError:
        # Unknown-key reporting runs before schema validation. Let the Literal
        # produce the authoritative invalid-kind error instead of failing here.
        return ConfigBlock(AlgorithmConfig)
    return _algorithm_config_block(cls)


_algorithm_config_variant_blocks: tuple[ConfigBlock, ...] = tuple(
    _algorithm_config_block(cls)
    for cls in dict.fromkeys(
        algorithm_config_class(kind)
        for kind in get_args(AlgorithmConfig.model_fields["kind"].annotation)
    )
)


# ── Data section ──────────────────────────────────────────────────────────────


class DataConfig(ConfigBase):
    loader: DataLoaderName | None = None
    manifest: str | None = None
    eval_manifest: str | None = None
    # readers: _validate_data + loader tooling
    preprocessing: Annotated[
        dict[str, Any] | None,
        ConfigBlock(
            (
                "resolution",
                "random_crop",
                "horizontal_flip",
                "format",
                "image_field",
                "caption_field",
                "conditioning",
                "reference_image",
            )
        ),
    ] = None
    sampler: Annotated[
        dict[str, Any] | None,
        ConfigBlock(("type", "shuffle", "drop_last", "dataloader_num_workers")),
    ] = None
    dataset_name: str | None = None
    split: str | None = None
    cache_dir: str | None = None
    # Precomputed clean-latents shard for the GRPO diffusion-loss regularizer
    # (algorithm.sft_weight > 0): {target_video -> VAE latents} written by
    # vrl/scripts/denoise/encode_targets.py. reader: run_online_recipe.
    sft_latents: str | None = None
    max_train_samples: Any = None
    task_type: str | None = None
    # Key registry: consumed by data/eval tooling, not validated here.
    allow_absolute_artifact_paths: Any = None
    artifact_data_root: Any = None
    source_report: Any = None

    @model_validator(mode="after")
    def _validate_data(self) -> DataConfig:
        self.loader = resolve_data_loader(
            self.loader,
            (self.preprocessing or {}).get("format"),
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
            for field in ("format", "image_field", "caption_field", "conditioning"):
                if field not in self.preprocessing:
                    raise ValueError(f"config missing required field: data.preprocessing.{field}")
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
                    raise ValueError(f"config missing required field: data.preprocessing.{field}")
            sampler = self.sampler or {}
            for field in ("shuffle", "drop_last", "dataloader_num_workers"):
                if field not in sampler:
                    raise ValueError(f"config missing required field: data.sampler.{field}")

        return self

    def _validate_sampler_type(self) -> None:
        """Shared sampler.type check for the prompt-manifest loaders."""
        sampler = self.sampler or {}
        sampler_type = str(sampler.get("type", "")) if "type" in sampler else ""
        if not sampler_type:
            raise ValueError("config missing required field: data.sampler.type")
        try:
            PromptSamplingStrategy(sampler_type)
        except ValueError as exc:
            expected = " / ".join(strategy.value for strategy in PromptSamplingStrategy)
            raise ValueError(
                f"unknown data.sampler.type={sampler_type!r}; expected {expected}",
            ) from exc


# ── Supporting sections for cross-field validation ────────────────────────────


class SdeConfig(ConfigBase):
    """Typed rollout.sde block. ``type`` is the user-facing allow-list, replacing
    hand-written membership checks previously duplicated in the
    schema cross-validator, layout, and flow_matching. The layout request-boundary
    guard stays for over-the-wire request dicts; ``window_*`` stay permissive.

    ``type`` names the replay/sampling transition distribution (flow_grpo, ddim,
    or cps); it is orthogonal to ``denoise_mode`` (native/sde), which owns the
    word ``sde``."""

    type: Literal["flow_grpo", "ddim", "cps"]
    window_size: Any = None
    window_range: Any = None


class RolloutConfig(ConfigBase):
    # readers: vrl/math/denoise/flow_matching.py window + RootConfig check
    sde: SdeConfig | None = Field(
        default=None,
        json_schema_extra={"runtime_owner": "generation_request"},
    )
    noise_level: float | None = Field(
        default=None,
        json_schema_extra={"runtime_owner": "generation_request"},
    )
    # janus_pro R1 only; the sole source for final_image_policy. Validated for
    # legality in RootConfig._cross_field_validate (which requires it for that kind).
    final_image_policy: Literal["always_generate", "use_selfcheck"] | None = Field(
        default=None,
        json_schema_extra={"runtime_owner": "generation_request"},
    )
    n_samples_per_prompt: int | None = None
    prompts_per_batch: int | None = None
    # reader: vrl/generation/bindings/full_sequence_denoise/layout.py
    # _parse_denoise_mode (request boundary).
    # Allowed set is the type; the layout guard stays for over-the-wire request dicts.
    denoise_mode: Literal["native", "sde"] | None = Field(
        default=None,
        json_schema_extra={"runtime_owner": "generation_request"},
    )
    # reader: vrl/generation/bindings/full_sequence_denoise/layout.py — opt-in to storing
    # each denoise step's rollout proposal mean for trust-region replay.
    return_prev_sample_mean: Any = Field(
        default=None,
        json_schema_extra={"runtime_owner": "generation_request"},
    )
    # reader: vrl/generation/bindings/full_sequence_denoise/layout.py — opt-in to caching
    # the frozen reference (LoRA-disabled) noise_pred at collect, so KL replay never
    # reruns the ref forward. Lossless: replay applies the same sde_step_with_logprob.
    cache_ref_noise_pred: Any = Field(
        default=None,
        json_schema_extra={"runtime_owner": "generation_request"},
    )
    # reader: generation planner (chunk_placement.py) + diffusion layout. int =
    # fixed chunk size; "auto" = the Ray runtime's startup chunk-size probe
    # resolves it before the first request (SPRINT_chunk_size_probe; Ray-only,
    # the planner rejects "auto" on other runtimes); null = samples_per_prompt.
    samples_per_chunk: int | Literal["auto"] | None = Field(
        default=None,
        json_schema_extra={"runtime_owner": "generation_request"},
    )
    torch_profiler: Annotated[Any, ConfigBlock(TorchProfilerConfig)] = None
    trajectory_storage: Annotated[Any, ConfigBlock(TrajectoryStoragePolicy)] = Field(
        default=None,
        json_schema_extra={"runtime_owner": "generation_request"},
    )

    @field_validator("samples_per_chunk", mode="before")
    @classmethod
    def _validate_samples_per_chunk(cls, value: Any) -> Any:
        """Keep fixed generation chunks positive; the runtime owns ``auto``."""

        if value is None or value == "auto":
            return value
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                "rollout.samples_per_chunk must be a positive integer, 'auto', or null",
            )
        return value


def generation_request_rollout_fields() -> frozenset[str]:
    """Derive rollout keys allowed to cross the generation request boundary."""

    return frozenset(
        name
        for name, model_field in RolloutConfig.model_fields.items()
        if (model_field.json_schema_extra or {}).get("runtime_owner") == "generation_request"
    )


@functools.cache
def _model_section_class_from_path(path: str) -> type[ModelSection]:
    section_cls = import_from_path(path)
    if not isinstance(section_cls, type) or not issubclass(section_cls, ModelSection):
        raise TypeError(
            f"model section path {path!r} must resolve to a ModelSection subclass",
        )
    return section_cls


def _model_section_class_for_family(family: Any) -> type[ModelSection]:
    if family is None or not str(family).strip():
        raise ValueError("config missing required field: model.family")
    entry = get_model_family_entry(str(family))
    return _model_section_class_from_path(entry.model_section_cls)


_model_section_variant_classes: tuple[type[ModelSection], ...] = tuple(
    dict.fromkeys(
        _model_section_class_from_path(entry.model_section_cls)
        for entry in FAMILY_REGISTRY.values()
    )
)


@functools.cache
def _model_section_block(cls: type[ModelSection]) -> ConfigBlock:
    return ConfigBlock(cls)


def _model_section_block_for_unknown_keys(mapping: Mapping[str, Any]) -> ConfigBlock:
    try:
        section_cls = _model_section_class_for_family(mapping.get("family"))
    except ValueError:
        # Unknown-key reporting precedes structural validation. Keep the walker
        # useful while RootConfig emits the authoritative missing/unknown-family
        # error instead of silently selecting a permissive fallback.
        section_cls = ModelSection
    return _model_section_block(section_cls)


_SectionT = TypeVar("_SectionT", bound=ConfigBase)


def _revalidate_section(
    section_cls: type[_SectionT],
    payload: Any,
    *,
    section: str,
) -> _SectionT:
    """Validate a bare section payload and re-anchor errors to its YAML path.

    The selected family class validates the bare ``model``/``sampling`` payload;
    on failure its error is re-prefixed so callers still receive the public
    ``<section>.<field>`` location instead of the bare field name.
    """

    try:
        return section_cls.model_validate(payload)
    except ValidationError as exc:
        message = _extract_error_message(exc)
        if message.startswith("unknown ") and not message.startswith(f"unknown {section}."):
            message = f"unknown {section}.{message[len('unknown ') :]}"
        elif message.startswith("config missing required field: "):
            rest = message[len("config missing required field: ") :]
            message = f"config missing required field: {section}.{rest}"
        raise ValueError(message) from exc


def _parse_model_section(value: Any) -> ModelSection | None:
    if value is None:
        return None
    if isinstance(value, ModelSection):
        family = value.family
        section_cls = _model_section_class_for_family(family)
        if isinstance(value, section_cls):
            parsed = value
            payload = None
        else:
            payload = value.model_dump(exclude_unset=True)
    elif isinstance(value, Mapping):
        payload = dict(value)
        section_cls = _model_section_class_for_family(payload.get("family"))
    else:
        raise ValueError("model must be a mapping")

    if payload is not None:
        parsed = _revalidate_section(section_cls, payload, section="model")

    entry = get_model_family_entry(str(parsed.family))
    entry.validate_model_runtime_sections(
        executor_config=parsed.executor,
        memory_config=parsed.memory,
    )
    return parsed


@functools.cache
def _sampling_section_class_from_path(path: str) -> type[SamplingSection]:
    section_cls = import_from_path(path)
    if not isinstance(section_cls, type) or not issubclass(section_cls, SamplingSection):
        raise TypeError(
            f"sampling section path {path!r} must resolve to a SamplingSection subclass",
        )
    return section_cls


def sampling_section_class_for_family(family: Any) -> type[SamplingSection]:
    if family is None or not str(family).strip():
        raise ValueError("sampling requires model.family")
    entry = get_model_family_entry(str(family))
    return _sampling_section_class_from_path(entry.sampling_section_cls)


_sampling_section_variant_classes: tuple[type[SamplingSection], ...] = tuple(
    dict.fromkeys(
        _sampling_section_class_from_path(entry.sampling_section_cls)
        for entry in FAMILY_REGISTRY.values()
    )
)


def _sampling_section_known_fields() -> tuple[str, ...]:
    """Derive the unknown-key inventory from every registered concrete schema."""

    return tuple(
        sorted(
            {
                name
                for section_cls in _sampling_section_variant_classes
                for name in section_cls.model_fields
            },
        ),
    )


def _parse_sampling_section(
    value: Any,
    *,
    model: ModelSection | None,
) -> SamplingSection | None:
    if value is None:
        return None
    section_cls = sampling_section_class_for_family(
        None if model is None else model.family,
    )
    if isinstance(value, SamplingSection):
        if isinstance(value, section_cls):
            return value
        payload = value.model_dump(exclude_unset=True)
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise ValueError("sampling must be a mapping")

    return _revalidate_section(section_cls, payload, section="sampling")


# ── Section key registries (values validated by their own layers) ────────────


@functools.cache
def _online_runtime_section_shape(
    section: Literal["actor", "trainer"],
    owners: tuple[type[Any], ...] = (TrainerConfig, OnlineBatchPlan),
) -> tuple[frozenset[str], dict[str, ConfigBlock]]:
    """Derive one public section shape from online runtime YAML metadata."""

    known: set[str] = set()
    children: dict[str, ConfigBlock] = {}

    for owner in owners:
        hints = get_type_hints(owner)
        for runtime_field in dataclass_fields(owner):
            home = runtime_field.metadata.get("yaml")
            if home is None:
                raise AssertionError(
                    f"{owner.__name__}.{runtime_field.name} does not declare "
                    "field metadata {'yaml': ...}",
                )
            if home == "bridged":
                continue
            if not isinstance(home, str):
                raise AssertionError(
                    f"{owner.__name__}.{runtime_field.name} declares non-string "
                    f"yaml home {home!r}",
                )
            top, _, nested_path = home.partition(".")
            if top != section:
                continue
            if runtime_field.name in known:
                raise AssertionError(
                    f"{section}.{runtime_field.name} has multiple online runtime owners",
                )
            known.add(runtime_field.name)
            if nested_path:
                if nested_path != runtime_field.name:
                    raise AssertionError(
                        f"{owner.__name__}.{runtime_field.name} declares yaml home "
                        f"{home!r}; nested runtime fields must use "
                        f"{section}.{runtime_field.name}",
                    )
                children[runtime_field.name] = ConfigBlock(
                    hints[runtime_field.name],
                )

    return frozenset(known), children


class _OnlineRuntimeSection(ConfigBase):
    """Retain derived runtime fields while dropping truly unknown extras."""

    model_config = ConfigDict(extra="allow")
    _yaml_section: ClassVar[Literal["actor", "trainer"]]

    @model_validator(mode="before")
    @classmethod
    def _drop_unknown_extras(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        runtime_fields, _ = _online_runtime_section_shape(cls._yaml_section)
        allowed = set(cls.model_fields) | runtime_fields
        return {key: inner for key, inner in value.items() if str(key) in allowed}


class TrainerSection(_OnlineRuntimeSection):
    """Non-online trainer keys plus the offline-entrypoint transition boundary."""

    _yaml_section = "trainer"

    entrypoint: Any = None
    total_epochs: Any = None
    save_freq: Any = None
    seed: Any = None
    resume_from: Any = None
    resume_strict: Any = None
    # offline DPO entrypoint (vrl/scripts/families/wan_2_1/train_dpo.py)
    checkpointing_steps: Any = None
    log_interval: Any = None
    max_train_steps: Any = None


class ActorSection(_OnlineRuntimeSection):
    """Non-online actor keys plus the offline-entrypoint transition boundary."""

    _yaml_section = "actor"

    # Public size input from which OnlineBatchPlan derives its canonical
    # gradient-accumulation count; it is intentionally not stored on that plan.
    microbatch_size: Any = None
    gradient_checkpointing: Any = None  # off | full | selective (or bool: true=full, false=off)
    # offline DPO entrypoint (vrl/scripts/families/wan_2_1/train_dpo.py)
    prediction_type: Any = None
    scale_lr: Any = None
    train_batch_size: Any = None
    use_adafactor: Any = None


def _online_runtime_section_block(
    section: Literal["actor", "trainer"],
    public_section: type[ConfigBase],
) -> ConfigBlock:
    """Combine explicit boundary keys with runtime-owned online keys."""

    explicit = ConfigBlock(public_section)
    runtime_fields, runtime_children = _online_runtime_section_shape(section)
    duplicates = explicit.known & runtime_fields
    if duplicates:
        names = ", ".join(sorted(duplicates))
        raise AssertionError(
            f"{public_section.__name__} duplicates online runtime field(s): {names}",
        )
    return ConfigBlock(
        explicit.known | runtime_fields,
        children={**explicit.children, **runtime_children},
    )


# Entrypoint-specific schema boundary. These are the only shared actor/trainer
# keys consumed by train_wan_2_1_dpo (plus trainer.entrypoint, consumed by the
# public dispatcher). Keeping the sets here makes inherited online-only knobs
# fail loudly before model/data construction.
_OFFLINE_DPO_ACTOR_FIELDS = frozenset(
    {
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "max_norm",
        "optim",
        "prediction_type",
        "scale_lr",
        "train_batch_size",
        "use_adafactor",
    },
)
_OFFLINE_DPO_TRAINER_FIELDS = frozenset(
    {
        "checkpointing_steps",
        "entrypoint",
        "log_interval",
        "max_train_steps",
        "output_dir",
        "resume_from",
        "resume_strict",
    },
)


class FSDPConfig(ConfigBase):
    """distributed.training.fsdp: the FSDP2 knobs ``build_strategy`` reads.

    Only the fields a reader consumes are declared (the same philosophy as
    ``TrainingSection``): ``vrl/trainers/strategy.py`` build_strategy +
    ``vrl/trainers/fsdp.py`` read ``mesh`` / ``precision_policy`` /
    ``reshard_after_forward`` / ``cpu_offload``. The remaining FSDP2 knobs from
    SPRINT_multi_gpu_training.md §3 (activation_checkpointing, state_dict,
    process-group backend/init) land here when their readers do —
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
    # Keep parameter/gradient shards on CPU between forwards. This is slower but
    # lets timestep-routed multi-root models materialize only the active expert.
    cpu_offload: bool = False


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

    @model_validator(mode="after")
    def _resolve_strategy_defaults(self) -> TrainingSection:
        """Resolve defaults here so runtime constructors cannot redeclare them.

        The unselected block stays absent: it has no runtime consumer and should
        not become duplicated resolved state merely because another strategy was
        chosen.
        """
        if self.fsdp is not None and self.strategy != "fsdp":
            raise ValueError(
                "distributed.training.fsdp requires distributed.training.strategy=fsdp",
            )
        if self.ddp is not None and self.strategy != "ddp":
            raise ValueError(
                "distributed.training.ddp requires distributed.training.strategy=ddp",
            )
        if self.strategy == "fsdp" and self.fsdp is None:
            self.fsdp = FSDPConfig()
        elif self.strategy == "ddp" and self.ddp is None:
            self.ddp = DDPConfig()
        return self


class RolloutWorkerSection(ConfigBase):
    """distributed.rollout: per-worker runtime knobs.

    reader: vrl/generation/ray/config.py RayGenerationConfig.from_cfg (worker
    runtime knobs). Release scheduling and colocation are NOT declared here:
    colocation lives in distributed.resources.rollout.gpu_pool=trainer (mirrors
    reward.gpu_pool), and release scheduling is derived from GPU topology.
    chunk_placement_strategy is a user-facing allow-list
    Literal: RayGenerationConfig is a plain dataclass whose annotations do not
    enforce, so this typed boundary is where a bad value is rejected (the runtime
    ChunkPlacementPolicy guard stays the wire-boundary check). sync_trainable_state
    is a plain on/off: True keeps rollout workers resynced to the trained policy
    (the syncer flattens whatever is trainable — lora or full-param), False
    disables the weight syncer.
    """

    cpus_per_worker: float = 1.0
    max_inflight_chunks_per_worker: int = 1
    # Background liveness probing of rollout workers. interval <= 0 disables it;
    # a worker that stops answering kills the owned actors so active or subsequent
    # foreground work fails closed. A failed verdict then enters the supervisor's
    # bounded restart policy.
    health_check_interval_s: float = 30.0
    health_check_timeout_s: float = 30.0
    health_check_first_wait_s: float = 0.0
    # Opaque control-plane calls expose no useful progress. Bound startup,
    # metadata, capability, and weight acknowledgements independently.
    worker_rpc_timeout_s: float = 600.0
    # Generation has a separate stall budget: a completed chunk is real progress,
    # and the pipelined path reports the same progress from its worker. Thirty
    # minutes covers the measured 733-second Cosmos chunk without turning a stuck
    # metadata or weight call into a two-hour outage.
    generation_stall_timeout_s: float = 1800.0
    chunk_placement_strategy: ChunkPlacementStrategy = "round_robin"
    sync_trainable_state: bool = True
    # Opt-in single-worker pipelined rollout. Config resolution rejects multiple
    # workers; requests with fewer than two chunks use the standard per-chunk path,
    # and a pipeline OOM falls back to that path's split-and-retry behavior.
    pipelined: bool = False

    @model_validator(mode="after")
    def _validate_health_check(self) -> RolloutWorkerSection:
        if not math.isfinite(self.health_check_interval_s):
            raise ValueError(
                "distributed.rollout.health_check_interval_s must be finite",
            )
        if self.health_check_interval_s > 0 and (
            not math.isfinite(self.health_check_timeout_s) or self.health_check_timeout_s <= 0
        ):
            raise ValueError(
                "distributed.rollout.health_check_timeout_s must be finite and > 0 "
                "when health checking is enabled",
            )
        if not math.isfinite(self.health_check_first_wait_s) or self.health_check_first_wait_s < 0:
            raise ValueError(
                "distributed.rollout.health_check_first_wait_s must be finite and >= 0",
            )
        if not math.isfinite(self.worker_rpc_timeout_s) or self.worker_rpc_timeout_s <= 0:
            raise ValueError(
                "distributed.rollout.worker_rpc_timeout_s must be finite and > 0",
            )
        if (
            not math.isfinite(self.generation_stall_timeout_s)
            or self.generation_stall_timeout_s <= 0
        ):
            raise ValueError(
                "distributed.rollout.generation_stall_timeout_s must be finite and > 0",
            )
        return self


class DistributedSection(ConfigBase):
    """Key registry for distributed.*; values validated by vrl.ray.resources."""

    # reader: vrl/ray/resources.py resolve_distributed_resources
    resources: Annotated[
        Any,
        ConfigBlock(
            ("visible_devices", "trainer", "rollout", "reward", "allow_overlap", "cross_node"),
            {
                "trainer": ConfigBlock(RoleResourceConfig),
                "rollout": ConfigBlock(RolloutResourceConfig),
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
    # knobs were removed; reward release and CuMem parking are derived from
    # topology rather than selected by an independent YAML lifecycle key.
    reward: Annotated[
        Any,
        ConfigBlock(("cpus_per_worker",)),
    ] = None
    # readers: vrl/trainers/distributed.py resolve_training_context (rank/device)
    # + vrl/ray/resources.py strategy-aware trainer GPU validation
    training: TrainingSection | None = None


# ── Root config ───────────────────────────────────────────────────────────────


class KlingVideoRewardProductionConfig(ConfigBase):
    """Production enablement for the Kling VideoReward contract gate."""

    enabled: bool = False


class ProductionSection(ConfigBase):
    """Closed registry of production gates with runtime consumers."""

    kling_video_reward: KlingVideoRewardProductionConfig = Field(
        default_factory=KlingVideoRewardProductionConfig,
    )


class RootConfig(ConfigBase):
    """Top-level typed boundary for all training config sections.

    Sections with leaf validation are modelled; trainer/actor/distributed are
    key registries (values validated by builders / ray.resources); precision
    is parsed by vrl.config.precision; cosmos by the cosmos entrypoint.
    Anything else warns via ConfigBase.
    """

    algorithm: Annotated[
        AlgorithmConfig | None,
        ConfigBlock(
            AlgorithmConfig,
            select=_algorithm_config_block_for_unknown_keys,
            variants=_algorithm_config_variant_blocks,
        ),
    ] = None
    data: DataConfig | None = None
    reward: RewardConfig | None = None
    rollout: RolloutConfig | None = None
    model: Annotated[
        SerializeAsAny[ModelSection] | None,
        ConfigBlock(
            ModelSection,
            select=_model_section_block_for_unknown_keys,
            variants=_model_section_variant_classes,
        ),
    ] = None
    sampling: Annotated[
        SerializeAsAny[SamplingSection] | None,
        ConfigBlock(
            _sampling_section_known_fields(),
            variants=_sampling_section_variant_classes,
        ),
    ] = None
    # Per-component production gates; contract checks live in
    # vrl/config/validation.py validate_production_* (raw-cfg checks)
    production: ProductionSection | None = None
    trainer: Annotated[
        TrainerSection | None,
        _online_runtime_section_block("trainer", TrainerSection),
    ] = None
    actor: Annotated[
        ActorSection | None,
        _online_runtime_section_block("actor", ActorSection),
    ] = None
    distributed: DistributedSection | None = None
    # reader: vrl/config/precision.py. Keep Any here so the resolver owns precise
    # migration and availability errors; ConfigBlock derives every known nested
    # key from PrecisionConfig rather than duplicating a hand-maintained tuple.
    precision: Annotated[Any, ConfigBlock(PrecisionConfig)] = None

    @field_validator("model", mode="before")
    @classmethod
    def _select_model_section(cls, value: Any) -> ModelSection | None:
        return _parse_model_section(value)

    @field_validator("sampling", mode="before")
    @classmethod
    def _select_sampling_section(
        cls,
        value: Any,
        info: ValidationInfo,
    ) -> SamplingSection | None:
        model = info.data.get("model")
        return _parse_sampling_section(
            value,
            model=model if isinstance(model, ModelSection) else None,
        )

    @model_validator(mode="after")
    def _cross_field_validate(self) -> RootConfig:
        algo = self.algorithm
        if algo is None:
            return self

        kind = algo.kind
        kl_reward_coef = resolve_kl_reward_coef(algo.kl_reward_coef)
        if kl_reward_coef > 0.0 and kind in {
            "token_grpo",
            "token_grpo_multisegment",
            "diffusion_dpo",
        }:
            raise ValueError(
                "algorithm.kl_reward_coef > 0 requires a diffusion rollout "
                "trajectory with collected per-step KL; "
                f"algorithm.kind={kind!r} does not provide one",
            )
        rollout = self.rollout
        model_family = normalize_model_family(
            (self.model.family or "") if self.model else "",
        )

        if kind == "diffusion_dpo":
            self._validate_offline_dpo_surface()

        # The SFT term belongs to continuous diffusion GRPO and offline
        # Diffusion-DPO. Validate the numeric domain and algorithm ownership
        # here so an inherited token-GRPO field cannot silently become a no-op.
        raw_sft_weight = getattr(algo, "sft_weight", None)
        if raw_sft_weight is not None:
            try:
                sft_weight = float(raw_sft_weight)
            except (TypeError, ValueError) as exc:
                raise ValueError("algorithm.sft_weight must be a finite number >= 0") from exc
            if not math.isfinite(sft_weight) or sft_weight < 0:
                raise ValueError("algorithm.sft_weight must be a finite number >= 0")
            if sft_weight > 0:
                if kind in {"grpo", "dance_grpo"}:
                    if self.data is None or not self.data.sft_latents:
                        raise ValueError(
                            "algorithm.sft_weight > 0 requires data.sft_latents "
                            "(the precomputed clean-latents shard; see "
                            "vrl/scripts/denoise/encode_targets.py)",
                        )
                elif kind != "diffusion_dpo":
                    raise ValueError(
                        "algorithm.sft_weight > 0 is supported only for diffusion "
                        "grpo/dance_grpo or diffusion_dpo",
                    )

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
        if (
            kind == "token_grpo"
            and model_family == "nextstep_1"
            and (rollout is None or rollout.noise_level is None)
        ):
            raise ValueError("config missing required field: rollout.noise_level")

        if model_family == "janus_pro_r1" and kind != "token_grpo_multisegment":
            raise ValueError(
                "model.family=janus_pro_r1 requires algorithm.kind=token_grpo_multisegment",
            )

        # token_grpo_multisegment: explicit Janus R1 protocol; final_image_policy
        # remains owned by rollout.
        if kind == "token_grpo_multisegment":
            if model_family != "janus_pro_r1":
                raise ValueError(
                    "token_grpo_multisegment currently requires model.family=janus_pro_r1",
                )
            # Single source: final_image_policy lives on rollout only. Validate it
            # here as a legality check (the collector reads rollout.final_image_policy).
            policy = (rollout.final_image_policy or "") if rollout else ""
            if policy not in {"always_generate", "use_selfcheck"}:
                raise ValueError(
                    "rollout.final_image_policy must be 'always_generate' or 'use_selfcheck'"
                )

        return self

    def _validate_offline_dpo_surface(self) -> None:
        for section_name, section, allowed in (
            ("actor", self.actor, _OFFLINE_DPO_ACTOR_FIELDS),
            ("trainer", self.trainer, _OFFLINE_DPO_TRAINER_FIELDS),
        ):
            if section is None:
                continue
            unsupported = sorted(section.model_fields_set - allowed)
            if unsupported:
                fields = ", ".join(f"{section_name}.{name}" for name in unsupported)
                raise ValueError(
                    f"diffusion_dpo does not consume config field(s): {fields}",
                )

        if self.rollout is not None and self.rollout.model_fields_set:
            fields = ", ".join(f"rollout.{name}" for name in sorted(self.rollout.model_fields_set))
            raise ValueError(
                f"diffusion_dpo does not consume config field(s): {fields}",
            )
        if self.reward is not None:
            raise ValueError("diffusion_dpo does not consume the reward config section")


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
    if error_type == "extra_forbidden":
        return f"unknown {loc}"
    # Literal enum mismatch — reformat to "unknown {loc}={input!r}; expected ..."
    if error_type == "literal_error":
        input_val = first.get("input", "")
        expected = msg.replace("Input should be", "expected")
        return f"unknown {loc}={input_val!r}; {expected}"
    # ValueError raised inside a validator — strip Pydantic's "Value error, " prefix
    if msg.startswith("Value error, "):
        return msg[len("Value error, ") :]
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
    "ModelSection",
    "RewardConfig",
    "RolloutConfig",
    "RootConfig",
    "generation_request_rollout_fields",
    "parse_config",
    "sampling_section_class_for_family",
]
