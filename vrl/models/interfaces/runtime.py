"""Model build inputs and runtime bundle (CONTRACT.md, SPRINT_model_refactor.md §5.1, §5.3.E).

These two dataclasses are the only sanctioned interface between training scripts
and family-adjacent builders. Scripts must not import family model classes
directly; builders consume a ``ModelBuild`` and return a ``RuntimeBundle``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast, get_args

from vrl.config.precision import QuantizationPolicy, RolePrecision
from vrl.models.interfaces.generation_memory import GenerationMemoryPolicy
from vrl.models.interfaces.replay import RuntimeModel

# Single source of truth for the model_config compile block that the
# ``ModelBuild.torch_compile`` property below consumes.
TORCH_COMPILE_MODEL_KEY = "torch_compile"
# Which build roles ``model.torch_compile`` applies to. ``rollout`` exists
# because the trainer's constraints (gradient checkpointing, FSDP2) forbid
# compiling the replay policy while the rollout policy — a plain inference
# module — remains compile-clean; ``replay`` is the mirror for sequence-parallel
# rollouts whose worker mutates the module tree after tracing.
TorchCompileScope = Literal["all", "rollout", "replay"]
PipelineOffloadMode = Literal["none", "model", "sequential"]

# Internal module attribute carrying the non-derivable part of checkpoint
# ownership. The complete set is derived from this plus trainable parameters.
_CHECKPOINT_OWNED_STATE_NAMES_ATTR = "_vrl_checkpoint_owned_state_names"


def _module_state_names(module: Any) -> frozenset[str]:
    """Return every parameter/buffer FQN exposed by a PyTorch-like module."""

    named_parameters = getattr(module, "named_parameters", None)
    named_buffers = getattr(module, "named_buffers", None)
    if not callable(named_parameters) or not callable(named_buffers):
        raise TypeError("checkpoint-owned state requires a PyTorch-like module")
    return frozenset(
        [
            *(name for name, _ in named_parameters(remove_duplicate=False)),
            *(name for name, _ in named_buffers(remove_duplicate=False)),
        ],
    )


def register_checkpoint_owned_state(module: Any, names: Iterable[str]) -> None:
    """Register frozen mutable parameter/buffer names needed for exact resume.

    Ordinary optimized parameters are derived from ``requires_grad`` and must
    not be registered. This stores only the exceptional state whose mutability
    cannot be inferred, such as DiffusionNFT's frozen ``previous`` adapter.
    """

    if isinstance(names, (str, bytes)):
        raise TypeError("checkpoint-owned state names must be an iterable of strings")
    registered = frozenset(names)
    if not registered or any(not isinstance(name, str) or not name for name in registered):
        raise ValueError("checkpoint-owned state names must be non-empty strings")
    unknown = sorted(registered - _module_state_names(module))
    if unknown:
        raise ValueError(f"checkpoint-owned state names do not exist in module: {unknown}")
    trainable = {
        name
        for name, parameter in module.named_parameters(remove_duplicate=False)
        if parameter.requires_grad
    }
    redundant = sorted(registered & trainable)
    if redundant:
        raise ValueError(
            "trainable parameters are already checkpoint-owned and must not be "
            f"registered explicitly: {redundant}",
        )
    current = getattr(module, _CHECKPOINT_OWNED_STATE_NAMES_ATTR, frozenset())
    if not isinstance(current, frozenset):
        raise TypeError("module checkpoint-owned state registry must be a frozenset")
    setattr(module, _CHECKPOINT_OWNED_STATE_NAMES_ATTR, current | registered)


def checkpoint_owned_state_names(module: Any) -> frozenset[str]:
    """Derive exact checkpoint state from trainable plus registered mutable names."""

    named_parameters = getattr(module, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("checkpoint-owned state requires a PyTorch-like module")
    trainable = frozenset(
        name
        for name, parameter in named_parameters(remove_duplicate=False)
        if parameter.requires_grad
    )
    registered = getattr(module, _CHECKPOINT_OWNED_STATE_NAMES_ATTR, frozenset())
    if not isinstance(registered, frozenset):
        raise TypeError("module checkpoint-owned state registry must be a frozenset")
    unknown = sorted(registered - _module_state_names(module))
    if unknown:
        raise ValueError(f"registered checkpoint-owned state no longer exists: {unknown}")
    return trainable | registered


@dataclass(frozen=True, slots=True)
class RolloutBuildOptions:
    """Generation-only precision and lifecycle inputs for a rollout model.

    ``prompt_encoder_dtype`` names the generation-only encoder precision policy
    the runtime actually consumes. ``pipeline_offload_mode`` owns optional
    Diffusers/Accelerate residency hooks that exist only on full generation
    pipelines. VAEs remain family-owned fp32 fidelity boundaries; putting their
    dtype under this option would advertise a knob they ignore.

    Replay builds carry ``rollout=None`` instead of hauling these fields through
    a path that must never quantize or load generation-only prompt encoders.
    """

    prompt_encoder_dtype: Any
    base_weight_sync: bool = True
    pipeline_offload_mode: PipelineOffloadMode = "none"

    def __post_init__(self) -> None:
        from vrl.models.dtypes import require_plain_dtype

        prompt_encoder_dtype = require_plain_dtype(
            self.prompt_encoder_dtype,
            what="rollout prompt encoder dtype",
        )
        object.__setattr__(self, "prompt_encoder_dtype", prompt_encoder_dtype)

        if not isinstance(self.base_weight_sync, bool):
            raise TypeError("rollout base_weight_sync must be a bool")
        require_pipeline_offload_mode(self.pipeline_offload_mode)


def require_pipeline_offload_mode(value: Any) -> PipelineOffloadMode:
    """Validate one ``rollout.pipeline_offload_mode`` value and return it.

    A free function, not a method: the Ray launch contract carries the rollout
    block as a serializable primitive mapping and must read this field BEFORE
    anyone reconstructs the typed ``RolloutBuildOptions`` from it. Both readers
    share this one definition so the wire path and the typed path cannot drift
    into accepting different vocabularies.
    """

    allowed = get_args(PipelineOffloadMode)
    if value not in allowed:
        raise ValueError(
            f"rollout pipeline_offload_mode must be one of {list(allowed)}, got {value!r}",
        )
    return cast("PipelineOffloadMode", value)


def require_torch_compile_scope(value: Any) -> TorchCompileScope:
    """Validate one ``model.torch_compile.scope`` value and return it.

    A free function, not a method: the config-load compile matrix
    (``vrl.config.validation.compile_conflicts``) must read this field before
    any ``ModelBuild`` exists, and the typed ``torch_compile`` property reads it
    per build. Both share this one definition so the two readers cannot drift
    into accepting different vocabularies (the ``require_pipeline_offload_mode``
    pattern above).
    """

    allowed = get_args(TorchCompileScope)
    if value not in allowed:
        raise ValueError(
            f"model.torch_compile.scope must be one of {list(allowed)}, got {value!r}",
        )
    return cast("TorchCompileScope", value)


def torch_compile_for_role(
    block: Any,
    role: Literal["rollout", "replay"],
) -> dict[str, Any] | None:
    """Resolve one raw ``model.torch_compile`` block for one build role.

    The single (block, role) -> compile decision. Consumers: the
    ``ModelBuild.torch_compile`` property (role from the build itself), the
    config-load compile matrix, and the two trainer-side runtime guards
    (FSDP2 in ``trainers/strategy.py``, grad-checkpointing in
    ``trainers/activation_checkpointing.py``) — all replay-role questions
    asked before a build exists. One definition so a scope added here cannot
    leave a guard reading the raw ``enable`` key and vetoing a role it does
    not bind.

    ``block`` is the typed ``TorchCompileSection`` (RootConfig) or the plain
    mapping ``ModelBuild.model_config`` carries across the process boundary;
    a mapping is validated into the section first so both read one way.
    """

    from vrl.config.model_schema import TorchCompileSection

    if block is None:
        return None
    if isinstance(block, Mapping):
        block = TorchCompileSection.model_validate(dict(block))
    if not block.enable:
        return None
    scope = require_torch_compile_scope(block.scope or "all")
    if scope not in ("all", role):
        return None
    return {"enable": True, "mode": block.mode or "default"}


@dataclass
class ModelBuild:
    """Model-construction slice of the whole RL config.

    Builders take this, not the whole RL cfg. Reward / algorithm / trainer /
    dataset / logging cadence are explicitly out of scope.

    ``model_config`` carries the model-owned remainder of ``cfg.model`` after
    registry identity, checkpoint path/revision, and executor settings are separated;
    ``sampling_config`` carries ``cfg.sampling``. The read properties expose
    common curated views so consumers read ``build.lora`` /
    ``build.num_steps`` directly instead of re-deriving from the raw block.
    The generation-memory policy is resolved into its own typed field before
    this build can cross Ray. Family-specific fields (e.g. anima checkpoint
    paths) are read straight from ``model_config``. The universal typed fields
    are runtime-injected or needed by every family, so they stay typed.
    """

    model_name_or_path: str
    # Immutable snapshot selector shared by every upstream model loader.
    revision: str | None
    device: Any
    # Resolved base transformer parameter dtype selected by the public role.
    parameter_dtype: Any
    # Canonical registry identity. Both driver replay and Ray rollout builds
    # require it; the worker restores it from the launch contract.
    family: str
    # The selected training or rollout role is the single precision source for
    # base compute, outer autocast, selective quantization, and process-local
    # FP32 matmul mode.
    # A primitive mapping is accepted only at the Ray wire boundary.
    precision: RolePrecision | Mapping[str, Any]
    model_config: dict[str, Any] | None = None
    sampling_config: dict[str, Any] | None = None
    # Generic denoise FSDP replay keeps the CPU-loaded trainable root on CPU
    # until fully_shard can move and shard it block by block. This is resolved
    # from the training strategy, not exposed as a user config knob.
    defer_trainable_device_move: bool = False
    # Resolved generation-only model memory policy. A primitive mapping is
    # accepted only at the Ray wire boundary and normalized immediately.
    generation_memory: GenerationMemoryPolicy | Mapping[str, Any] | None = None
    # Full-generation build inputs. ``None`` is the replay contract: replay owns
    # only differentiable policy modules and must not react to rollout FP8/frozen
    # component settings. A nested primitive mapping is accepted only for the Ray
    # wire boundary and normalized immediately in ``__post_init__``.
    rollout: RolloutBuildOptions | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        from vrl.models.dtypes import require_plain_dtype

        if not isinstance(self.family, str) or not self.family:
            raise ValueError("ModelBuild.family must be a non-empty string")
        if isinstance(self.precision, Mapping):
            precision_payload = dict(self.precision)
            quantization = precision_payload.get("quantization")
            if isinstance(quantization, Mapping):
                precision_payload["quantization"] = QuantizationPolicy(
                    **dict(quantization),
                )
            self.precision = RolePrecision(**precision_payload)
        elif not isinstance(self.precision, RolePrecision):
            raise TypeError(
                "ModelBuild.precision must be RolePrecision or a mapping",
            )
        self.parameter_dtype = require_plain_dtype(
            self.parameter_dtype,
            what="runtime parameter dtype",
            detail=(
                ". FP8 and NVFP4 are selective rollout quantization formats; "
                "neither is parameter storage."
            ),
        )
        if self.model_config is not None and "memory" in self.model_config:
            raise ValueError(
                "ModelBuild.model_config must not carry model.memory; "
                "resolve it into generation_memory",
            )
        if isinstance(self.rollout, Mapping):
            self.rollout = RolloutBuildOptions(**dict(self.rollout))
        elif self.rollout is not None and not isinstance(self.rollout, RolloutBuildOptions):
            raise TypeError(
                "ModelBuild.rollout must be RolloutBuildOptions, a mapping, or None",
            )
        if isinstance(self.generation_memory, Mapping):
            self.generation_memory = GenerationMemoryPolicy(
                **dict(self.generation_memory),
            )
        elif self.generation_memory is not None and not isinstance(
            self.generation_memory,
            GenerationMemoryPolicy,
        ):
            raise TypeError(
                "ModelBuild.generation_memory must be GenerationMemoryPolicy, a mapping, or None",
            )
        if not isinstance(self.defer_trainable_device_move, bool):
            raise TypeError("ModelBuild.defer_trainable_device_move must be a bool")
        if self.defer_trainable_device_move and self.rollout is not None:
            raise ValueError(
                "defer_trainable_device_move is replay-only and cannot be set "
                "with ModelBuild.rollout",
            )
        if self.generation_memory is not None and self.rollout is None:
            raise ValueError(
                "generation_memory is rollout-only and requires ModelBuild.rollout",
            )

    def require_rollout(self) -> RolloutBuildOptions:
        """Return rollout build inputs or fail at the role boundary."""

        if not isinstance(self.rollout, RolloutBuildOptions):
            raise ValueError(
                "rollout runtime construction requires ModelBuild.rollout; "
                "resolve the build through a rollout resolver",
            )
        return self.rollout

    def require_replay(self) -> None:
        """Reject generation-only state on a trainer replay build."""

        if self.rollout is not None:
            raise ValueError(
                "replay runtime construction requires ModelBuild.rollout=None; "
                "resolve the build through a replay resolver",
            )

    @property
    def use_lora(self) -> bool:
        """Whether the family should attach a LoRA adapter.

        ``False`` fallback when the block is absent (safe for fake test builds);
        every real experiment config sets ``model.use_lora`` explicitly.
        """
        return bool((self.model_config or {}).get("use_lora", False))

    @property
    def lora_path(self) -> str | None:
        """Resolved LoRA checkpoint path, or ``None`` when not loading one."""
        lora = (self.model_config or {}).get("lora") or {}
        return lora.get("path") or None

    @property
    def lora(self) -> dict[str, Any] | None:
        """Curated LoRA config (``rank``/``alpha``/``target_modules`` + extras).

        ``None`` when ``use_lora`` is off. Casts and the ``init_lora_weights`` /
        ``dropout`` / ``init`` extras are carried from the raw ``model.lora``
        block only when present, preserving per-family presence semantics.
        Token-family config dataclasses own their resolved LoRA defaults.
        """
        if not self.use_lora:
            return None
        lora = (self.model_config or {}).get("lora") or {}
        config: dict[str, Any] = {
            "rank": int(lora["rank"]),
            "alpha": int(lora["alpha"]),
            "target_modules": list(lora["target_modules"]),
        }
        for key in ("init_lora_weights", "dropout", "init"):
            if key in lora:
                config[key] = lora[key]
        return config

    @property
    def num_steps(self) -> int | None:
        """Diffusion scheduler step count from ``sampling.num_steps``."""
        num_steps = (self.sampling_config or {}).get("num_steps")
        return None if num_steps is None else int(num_steps)

    @property
    def torch_compile(self) -> dict[str, Any] | None:
        """``model.torch_compile`` block only when it covers this build's role.

        ``scope`` narrows compilation to one side of the rollout/replay split:
        under gradient checkpointing or FSDP2 the replay policy must stay
        eager while the rollout policy can still compile (and mirrored for
        sequence-parallel rollouts). The role is decided by the build itself —
        ``rollout`` present or ``None`` — so every consumer (the rollout
        ``CompilePass``, ``assemble_replay_bundle``, the loader's
        quantization conflict) reads one role-correct answer instead of
        re-deriving it.
        """
        return torch_compile_for_role(
            (self.model_config or {}).get(TORCH_COMPILE_MODEL_KEY),
            "rollout" if self.rollout is not None else "replay",
        )

    @property
    def revision_kwargs(self) -> dict[str, str]:
        """The immutable model snapshot argument for every upstream loader."""
        return {"revision": str(self.revision)} if self.revision else {}

    @property
    def pretrained_kwargs(self) -> dict[str, Any]:
        """Repository-wide options shared by full and component loaders."""
        kwargs: dict[str, Any] = self.revision_kwargs
        model_config = self.model_config or {}
        if "local_files_only" in model_config:
            kwargs["local_files_only"] = bool(model_config["local_files_only"])
        return kwargs

    def config_revision_kwargs(self, field: str) -> dict[str, str]:
        """An optional revision kwarg owned by one model-config repository.

        ``revision`` above pins the model repository itself; families that pull
        a tokenizer or encoder from a *separate* repository record that repo's
        snapshot under its own ``model_config`` key, named by ``field``.
        """
        revision = (self.model_config or {}).get(field)
        return {"revision": str(revision)} if revision else {}


@dataclass
class RuntimeBundle:
    """Sole output of a family builder. Consumed by scripts / collectors / trainers.

    ``raw_handle`` carries the raw family object (e.g. the diffusers pipeline
    or AR upstream wrapper) and must be treated as builder-internal — scripts
    and trainers must not reach into it.

    ``model`` is the family-provided general inference object. Shared trainer
    runtime code only relies on the narrow ``vrl.models.interfaces.RuntimeModel``
    contract: ``replay_forward``, ``disable_adapter``, and
    ``load_trainable_state``.
    Generation-only methods remain family/runtime implementation details.

    ``trainable_modules`` is the training-checkpoint root contract. Every module
    registered here must expose PyTorch-compatible ``state_dict`` and
    ``load_state_dict`` methods. Within each root, checkpointing derives exact
    ownership from trainable parameters plus explicitly registered frozen
    mutable state; family builders must include every root needed for resume.

    ``precision`` is the selected training or rollout role, including whether
    diffusion forwards apply its dtype through outer autocast. Assembly stamps
    the complete role on ``model`` for replay algorithms that call the
    transformer outside the family ``forward_step`` wrapper.

    ``loads_full_generation_modules`` is true when the bundle owns
    generation-only modules such as prompt encoders, VAE/VQ decoders, or a full
    pipeline object. Rollout workers own that full generation state for
    sampling/decoding; trainers own only the modules needed to replay recorded
    trajectory actions. Consumed by the colocated-RAM guard in
    ``RayGenerationConfig.validate_driver_state`` to size host memory.

    ``adapter_roots`` maps checkpoint root name to the module holding that
    root's PEFT adapter, which is not always the root itself (token families
    register the wrapper but attach LoRA to ``language_model`` inside it).
    Snapshotted after the family attaches LoRA so ``build_adapter_exports`` can
    publish artifacts without reaching past the narrow ``RuntimeModel``
    contract. Empty for a family that owns no exportable adapter.
    """

    model: RuntimeModel
    trainable_modules: dict[str, Any]
    scheduler: Any
    raw_handle: Any
    precision: RolePrecision
    loads_full_generation_modules: bool
    adapter_roots: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.precision, RolePrecision):
            raise TypeError(
                "RuntimeBundle.precision must be RolePrecision",
            )
        # Replay consumers read the exact role policy from the model they
        # forward, so a caller cannot pair a model with a different policy.
        self.model.precision = self.precision
