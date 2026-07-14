"""Model build inputs and runtime bundle (CONTRACT.md, SPRINT_model_refactor.md §5.1, §5.3.E).

These two dataclasses are the only sanctioned interface between training scripts
and family-adjacent builders. Scripts must not import family model classes
directly; builders consume a ``ModelBuild`` and return a ``RuntimeBundle``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from vrl.models.interfaces.replay import RuntimeModel

# Single source of truth for valid ``model.memory`` subsection names, shared by
# the config schema's unknown-key lint and the generation memory policy's typo
# check. Today only ``vae_decode`` (sliced/tiled VAE decode, applied on the
# rollout/generation side). The trainer needs no section here: each family
# builds a minimal ReplayModel that never loads the generation-only modules, so
# there is nothing to offload. Adding a section means editing this tuple once;
# both consumers derive from it.
MODEL_MEMORY_SECTIONS: tuple[str, ...] = ("vae_decode",)

# Single source of truth for the model_config compile block that the
# ``ModelBuild.torch_compile`` property below consumes.
TORCH_COMPILE_MODEL_KEY = "torch_compile"

# The trainer and Ray rollout worker load different runtime surfaces: rollout
# workers own full generation state for sampling/decoding; trainers own only
# the modules needed to replay recorded trajectory actions. The one fact the
# runtime actually reads is whether a bundle owns full generation modules,
# consumed by the colocated-RAM guard (``validate_colocated_replay_memory``)
# to size host memory.
LOADS_FULL_GENERATION_MODULES_KEY = "loads_full_generation_modules"


def full_generation_bundle_metadata() -> dict[str, Any]:
    """Return metadata for a runtime bundle that owns full generation modules."""

    return {LOADS_FULL_GENERATION_MODULES_KEY: True}


def minimal_replay_bundle_metadata() -> dict[str, Any]:
    """Return metadata for a trainer bundle that owns only replay modules."""

    return {LOADS_FULL_GENERATION_MODULES_KEY: False}


def bundle_loads_full_generation_modules(bundle: Any) -> bool:
    """Return whether a runtime bundle declares full generation module ownership."""

    metadata = getattr(bundle, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        return False
    return bool(metadata.get(LOADS_FULL_GENERATION_MODULES_KEY, False))


@dataclass(frozen=True, slots=True)
class RolloutBuildOptions:
    """Generation-only precision and lifecycle inputs for a rollout model.

    ``autocast_dtype`` is deliberately distinct from ``quantization_format``. A plain
    fp32/bf16/fp16 rollout uses that dtype directly, while an FP8 rollout swaps
    selected GEMMs and still needs a real autocast dtype for attention, norms,
    residuals, and every linear that was not swapped. NVFP4 follows the same
    layered contract but conservatively swaps MLP linears only.

    ``prompt_encoder_dtype`` names the generation-only encoder precision policy
    the runtime actually consumes. VAEs remain family-owned fp32 fidelity boundaries;
    putting their dtype under this option would advertise a knob they ignore.

    Replay builds carry ``rollout=None`` instead of hauling these fields through
    a path that must never quantize or load generation-only prompt encoders.
    """

    autocast_dtype: Any
    prompt_encoder_dtype: Any
    quantization_format: str | None = None
    quantization_recipe: str | None = None
    base_weight_sync: bool = True

    def __post_init__(self) -> None:
        from vrl.models.dtypes import (
            dtype_to_precision_token,
            dtype_to_wire_name,
            resolve_torch_dtype,
        )

        autocast_dtype = resolve_torch_dtype(self.autocast_dtype)
        autocast_name = dtype_to_wire_name(autocast_dtype)
        try:
            dtype_to_precision_token(autocast_dtype)
        except ValueError as error:
            raise ValueError(
                "rollout autocast dtype must be fp16, bf16, or fp32; "
                f"got {autocast_name!r}. FP8 and NVFP4 are selective GEMM "
                "formats; neither is an outer torch.autocast dtype.",
            ) from error
        object.__setattr__(self, "autocast_dtype", autocast_dtype)
        prompt_encoder_dtype = resolve_torch_dtype(self.prompt_encoder_dtype)
        prompt_encoder_name = dtype_to_wire_name(prompt_encoder_dtype)
        try:
            dtype_to_precision_token(prompt_encoder_dtype)
        except ValueError as error:
            raise ValueError(
                "rollout prompt encoder dtype must be fp16, bf16, or fp32; "
                f"got {prompt_encoder_name!r}",
            ) from error
        object.__setattr__(self, "prompt_encoder_dtype", prompt_encoder_dtype)

        recipe = (
            str(self.quantization_recipe).lower().strip()
            if self.quantization_recipe is not None
            else None
        )
        if recipe is not None and self.quantization_format is None:
            raise ValueError(
                "rollout quantization_recipe requires quantization_format",
            )
        quantization_format = None
        if self.quantization_format is not None:
            from vrl.config.precision import QuantizationPolicy

            quantization = QuantizationPolicy(
                format=str(self.quantization_format),
                recipe=recipe,
            )
            quantization_format = quantization.format
            recipe = quantization.recipe
        if not isinstance(self.base_weight_sync, bool):
            raise TypeError("rollout base_weight_sync must be a bool")
        object.__setattr__(self, "quantization_format", quantization_format)
        object.__setattr__(self, "quantization_recipe", recipe)


@dataclass
class ModelBuild:
    """Model-construction slice of the whole RL config.

    Builders take this, not the whole RL cfg. Reward / algorithm / trainer /
    dataset / logging cadence are explicitly out of scope.

    ``model_config`` carries the model-owned remainder of ``cfg.model`` after
    registry identity, checkpoint path, and executor settings are separated;
    ``sampling_config`` carries ``cfg.sampling``. The read properties expose
    common curated views so
    consumers read ``build.memory`` / ``build.lora`` / ``build.num_steps`` directly
    instead of re-deriving from the raw block. They centralize the lora /
    scheduler / memory / compile transforms in one place, so no read-time logic
    is duplicated per family. Family-specific fields (e.g. anima checkpoint
    paths) are read straight from ``model_config``. The universal typed fields
    are runtime-injected or needed by every family, so they stay typed.
    """

    model_name_or_path: str
    device: Any
    # Resolved base transformer parameter dtype. A family build descriptor may pin
    # this independently of rollout autocast (SANA: fp16 parameters, bf16 forward).
    parameter_dtype: Any
    # Canonical registry identity. Both driver replay and Ray rollout builds
    # require it; the worker restores it from the launch contract.
    family: str
    model_config: dict[str, Any] | None = None
    sampling_config: dict[str, Any] | None = None
    # Full-generation build inputs. ``None`` is the replay contract: replay owns
    # only differentiable policy modules and must not react to rollout FP8/frozen
    # component settings. A nested primitive mapping is accepted only for the Ray
    # wire boundary and normalized immediately in ``__post_init__``.
    rollout: RolloutBuildOptions | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        from vrl.models.dtypes import (
            dtype_to_precision_token,
            dtype_to_wire_name,
            resolve_torch_dtype,
        )

        if not isinstance(self.family, str) or not self.family:
            raise ValueError("ModelBuild.family must be a non-empty string")
        self.parameter_dtype = resolve_torch_dtype(self.parameter_dtype)
        parameter_name = dtype_to_wire_name(self.parameter_dtype)
        try:
            dtype_to_precision_token(self.parameter_dtype)
        except ValueError as error:
            raise ValueError(
                "runtime parameter dtype must be fp16, bf16, or fp32; "
                f"got {parameter_name!r}. FP8 and NVFP4 are selective rollout "
                "quantization formats; neither is parameter storage.",
            ) from error
        if isinstance(self.rollout, Mapping):
            self.rollout = RolloutBuildOptions(**dict(self.rollout))
        elif self.rollout is not None and not isinstance(self.rollout, RolloutBuildOptions):
            raise TypeError(
                "ModelBuild.rollout must be RolloutBuildOptions, a mapping, or None",
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
        block only when present, preserving per-family presence semantics. AR
        families layer their own defaults via
        ``vrl.models.ar.build.ar_model_config_base``.
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
        """``model.torch_compile`` block only when ``enable`` is truthy."""
        block = (self.model_config or {}).get(TORCH_COMPILE_MODEL_KEY) or {}
        if not block.get("enable"):
            return None
        return {"enable": True, "mode": block.get("mode", "default")}

    @property
    def memory(self) -> dict[str, Any] | None:
        """The whole ``model.memory`` block (consumer extracts its sub-block)."""
        return (self.model_config or {}).get("memory")


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

    ``trainable_modules`` is the training-checkpoint contract. Every module
    registered here must expose PyTorch-compatible ``state_dict`` and
    ``load_state_dict`` methods. Generic trainer checkpointing saves and
    restores only these modules, so family builders must include every
    trainable adapter/backbone needed for exact resume.

    ``metadata`` may include generic replay/runtime flags used by shared
    trainer infrastructure. Build these through this module's
    ``full_generation_bundle_metadata`` / ``minimal_replay_bundle_metadata``
    so family runtimes share one contract:

    - ``loads_full_generation_modules``: true when the bundle owns
      generation-only modules such as prompt encoders, VAE/VQ decoders, or a
      full pipeline object. Consumed by the colocated-RAM guard
      (``validate_colocated_replay_memory``).
    """

    model: RuntimeModel
    trainable_modules: dict[str, Any]
    scheduler: Any
    raw_handle: Any
    metadata: dict[str, Any] = field(default_factory=dict)
