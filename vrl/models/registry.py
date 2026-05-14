"""Model-owned rollout registration helpers.

Runtime modules call these helpers after defining their executor and runtime
builders. The central rollout registry explicitly imports those runtime modules
before reading this registry, so registration stays deterministic instead of
depending on incidental import order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from vrl.engine.core.capabilities import (
    FamilyCapability,
    ar_continuous_family_capability,
    diffusion_family_capability,
)

CollectorKind = Literal["diffusion", "ar_discrete", "ar_continuous", "ar_r1"]

DIFFUSION_MIGRATED_RETURN_ARTIFACTS = (
    "output",
    "trajectory",
)

SD3_MIGRATED_RETURN_ARTIFACTS = DIFFUSION_MIGRATED_RETURN_ARTIFACTS

JANUS_PRO_MIGRATED_RETURN_ARTIFACTS = (
    "output",
    "trajectory",
)

NEXTSTEP_MIGRATED_RETURN_ARTIFACTS = (
    "output",
    "trajectory",
)

JANUS_PRO_R1_MIGRATED_RETURN_ARTIFACTS = (
    "output",
    "trajectory",
)

DIFFUSION_COMMON_SAMPLING_FIELDS = (
    "num_steps",
    "guidance_scale",
    "height",
    "width",
    "cfg",
    "sample_batch_size",
    "sde_type",
    "sde_window_size",
    "sde_window_range",
    "same_latent",
    "max_sequence_length",
    "noise_level",
    "return_kl",
)

DIFFUSION_VIDEO_SAMPLING_FIELDS = (
    *DIFFUSION_COMMON_SAMPLING_FIELDS,
    "num_frames",
)


@dataclass(frozen=True, slots=True)
class GathererRegistration:
    """Driver-side chunk gatherer construction metadata."""

    import_path: str
    kwargs: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecutorKwargsRegistration:
    """Runtime executor kwargs that can be derived from a full rollout cfg."""

    include_sample_batch_size: bool = False
    include_reference_image: bool = False


@dataclass(frozen=True, slots=True)
class RolloutRegistration:
    """One rollout mode exposed by a model runtime."""

    family: str
    task: str
    collector_kind: CollectorKind
    collector_config_cls: str
    executor_cls: str
    runtime_builder: str
    runtime_spec_extractor: str
    gatherer: GathererRegistration
    capability: FamilyCapability
    aliases: tuple[str, ...] = ()
    request_prefix: str | None = None
    default_task_type: str | None = None
    error_prefix: str | None = None
    sampling_fields: tuple[str, ...] = ()
    return_artifacts: tuple[str, ...] = ("output", "trajectory")
    metadata_key: str | None = None
    executor_kwargs: ExecutorKwargsRegistration = field(
        default_factory=ExecutorKwargsRegistration,
    )


@dataclass(frozen=True, slots=True)
class ModelRegistration:
    """Model-level declaration of supported rollout modes."""

    name: str
    rollouts: tuple[RolloutRegistration, ...]
    aliases: tuple[str, ...] = ()


_MODEL_REGISTRY: dict[str, ModelRegistration] = {}


def register_model(
    name: str,
    *,
    rollouts: Sequence[RolloutRegistration],
    aliases: Sequence[str] = (),
) -> ModelRegistration:
    """Register one model and the rollout modes it exposes."""

    model_name = str(name)
    if not model_name:
        raise ValueError("model registration name must be non-empty")
    if model_name in _MODEL_REGISTRY:
        raise ValueError(f"model {model_name!r} is already registered")
    rollout_tuple = tuple(rollouts)
    if not rollout_tuple:
        raise ValueError(f"model {model_name!r} must expose at least one rollout")
    registration = ModelRegistration(
        name=model_name,
        aliases=tuple(str(alias) for alias in aliases),
        rollouts=rollout_tuple,
    )
    _MODEL_REGISTRY[model_name] = registration
    return registration


def register_diffusion_model(
    name: str,
    *,
    task: str,
    executor_cls: str | type,
    runtime_builder: str | Callable[..., Any],
    runtime_spec_extractor: str | Callable[..., Any],
    collector_config_cls: str | type,
    aliases: Sequence[str] = (),
    request_prefix: str | None = None,
    default_task_type: str | None = None,
    error_prefix: str | None = None,
    video: bool = False,
    extra_sampling_fields: Sequence[str] = (),
    gatherer_kwargs: dict[str, object] | None = None,
    include_reference_image: bool = False,
    model_name: str | None = None,
) -> ModelRegistration:
    """Register a diffusion model with the standard migrated rollout contract."""

    family = str(name)
    sampling_fields = (
        DIFFUSION_VIDEO_SAMPLING_FIELDS if video else DIFFUSION_COMMON_SAMPLING_FIELDS
    )
    rollout = RolloutRegistration(
        family=family,
        task=str(task),
        aliases=tuple(str(alias) for alias in aliases),
        collector_kind="diffusion",
        collector_config_cls=import_path(collector_config_cls),
        request_prefix=request_prefix or family,
        default_task_type=default_task_type,
        error_prefix=error_prefix,
        sampling_fields=(*sampling_fields, *(str(field) for field in extra_sampling_fields)),
        return_artifacts=(
            SD3_MIGRATED_RETURN_ARTIFACTS
            if family == "sd3_5"
            else DIFFUSION_MIGRATED_RETURN_ARTIFACTS
        ),
        executor_cls=import_path(executor_cls),
        runtime_builder=import_path(runtime_builder),
        runtime_spec_extractor=import_path(runtime_spec_extractor),
        gatherer=GathererRegistration(
            import_path="vrl.engine.execution.gather:DiffusionChunkGatherer",
            kwargs=dict(gatherer_kwargs or {"model_family": family}),
        ),
        capability=diffusion_family_capability(
            family,
            str(task),
            supports_reference_conditioning=include_reference_image,
        ),
        executor_kwargs=ExecutorKwargsRegistration(
            include_sample_batch_size=True,
            include_reference_image=include_reference_image,
        ),
    )
    return register_model(model_name or family, rollouts=(rollout,))


def register_ar_continuous_model(
    name: str,
    *,
    task: str,
    executor_cls: str | type,
    runtime_builder: str | Callable[..., Any],
    runtime_spec_extractor: str | Callable[..., Any],
    collector_config_cls: str | type,
    gatherer_import_path: str,
    aliases: Sequence[str] = (),
    request_prefix: str | None = None,
    sampling_fields: Sequence[str] = (),
    metadata_key: str | None = None,
) -> ModelRegistration:
    """Register a continuous-token AR model with one rollout mode."""

    family = str(name)
    rollout = RolloutRegistration(
        family=family,
        task=str(task),
        aliases=tuple(str(alias) for alias in aliases),
        collector_kind="ar_continuous",
        collector_config_cls=import_path(collector_config_cls),
        request_prefix=request_prefix or family,
        sampling_fields=tuple(str(field) for field in sampling_fields),
        return_artifacts=NEXTSTEP_MIGRATED_RETURN_ARTIFACTS,
        metadata_key=metadata_key,
        executor_cls=import_path(executor_cls),
        runtime_builder=import_path(runtime_builder),
        runtime_spec_extractor=import_path(runtime_spec_extractor),
        gatherer=GathererRegistration(import_path=gatherer_import_path),
        capability=ar_continuous_family_capability(family, str(task)),
    )
    return register_model(family, rollouts=(rollout,))


def registered_models() -> tuple[ModelRegistration, ...]:
    """Return model registrations in deterministic insertion order."""

    return tuple(_MODEL_REGISTRY.values())


def import_path(value: str | type | Callable[..., Any]) -> str:
    """Return ``module:qualname`` for a class/function or pass strings through."""

    if isinstance(value, str):
        if ":" not in value:
            raise ValueError(f"import path must use 'module:attr' syntax: {value!r}")
        return value
    module = getattr(value, "__module__", "")
    qualname = getattr(value, "__qualname__", "")
    if not module or not qualname or "<locals>" in qualname:
        raise ValueError(f"cannot derive stable import path for {value!r}")
    return f"{module}:{qualname}"


__all__ = [
    "DIFFUSION_COMMON_SAMPLING_FIELDS",
    "DIFFUSION_MIGRATED_RETURN_ARTIFACTS",
    "DIFFUSION_VIDEO_SAMPLING_FIELDS",
    "JANUS_PRO_MIGRATED_RETURN_ARTIFACTS",
    "JANUS_PRO_R1_MIGRATED_RETURN_ARTIFACTS",
    "NEXTSTEP_MIGRATED_RETURN_ARTIFACTS",
    "SD3_MIGRATED_RETURN_ARTIFACTS",
    "CollectorKind",
    "ExecutorKwargsRegistration",
    "GathererRegistration",
    "ModelRegistration",
    "RolloutRegistration",
    "import_path",
    "register_ar_continuous_model",
    "register_diffusion_model",
    "register_model",
    "registered_models",
]
