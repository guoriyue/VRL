"""Canonical rollout family registry derived from model registrations.

Model runtime modules own the one-line registration of supported rollout modes.
This module explicitly imports those built-in runtimes, then adapts their
model-level declarations into the rollout-facing metadata consumed by
collectors, runtime launchers, and distributed backends.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Literal

from vrl.engine.core.capabilities import FamilyCapability
from vrl.models.registry import (
    DIFFUSION_COMMON_SAMPLING_FIELDS,
    DIFFUSION_MIGRATED_RETURN_ARTIFACTS,
    DIFFUSION_VIDEO_SAMPLING_FIELDS,
    JANUS_PRO_MIGRATED_RETURN_ARTIFACTS,
    JANUS_PRO_R1_MIGRATED_RETURN_ARTIFACTS,
    NEXTSTEP_MIGRATED_RETURN_ARTIFACTS,
    SD3_MIGRATED_RETURN_ARTIFACTS,
    RolloutRegistration,
    registered_models,
)

CollectorKind = Literal["diffusion", "ar_discrete", "ar_continuous", "ar_r1"]


@dataclass(frozen=True, slots=True)
class CollectorMetadata:
    """Collector-facing family metadata shared by collector builders."""

    kind: CollectorKind
    config_cls: str
    request_prefix: str | None = None
    default_task_type: str | None = None
    error_prefix: str | None = None
    sampling_fields: tuple[str, ...] = ()
    return_artifacts: tuple[str, ...] = ()
    metadata_key: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutorKwargsMetadata:
    """Runtime executor kwargs that can be derived from a full rollout cfg."""

    include_sample_batch_size: bool = False
    include_reference_image: bool = False


@dataclass(frozen=True, slots=True)
class GathererMetadata:
    """Driver-side chunk gatherer construction metadata."""

    import_path: str
    kwargs: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RolloutFamilyEntry:
    """Declarative binding for one canonical rollout family."""

    family: str
    task: str
    collector: CollectorMetadata
    executor_cls: str
    runtime_builder: str
    runtime_spec_extractor: str
    gatherer: GathererMetadata
    capability: FamilyCapability
    executor_kwargs: ExecutorKwargsMetadata = field(
        default_factory=ExecutorKwargsMetadata,
    )
    aliases: tuple[str, ...] = ()


_BUILTIN_MODEL_RUNTIME_MODULES = (
    "vrl.models.families.sd3_5.runtime",
    "vrl.models.families.wan_2_1.runtime",
    "vrl.models.families.cosmos.predict2.runtime",
    "vrl.models.families.cosmos.predict2_5.runtime",
    "vrl.models.families.janus_pro.runtime",
    "vrl.models.families.nextstep_1.runtime",
)


def _load_builtin_model_registrations() -> None:
    for module_name in _BUILTIN_MODEL_RUNTIME_MODULES:
        importlib.import_module(module_name)


def _rollout_entry_from_model_registration(
    registration: RolloutRegistration,
) -> RolloutFamilyEntry:
    return RolloutFamilyEntry(
        family=registration.family,
        task=registration.task,
        aliases=registration.aliases,
        collector=CollectorMetadata(
            kind=registration.collector_kind,
            config_cls=registration.collector_config_cls,
            request_prefix=registration.request_prefix,
            default_task_type=registration.default_task_type,
            error_prefix=registration.error_prefix,
            sampling_fields=registration.sampling_fields,
            return_artifacts=registration.return_artifacts,
            metadata_key=registration.metadata_key,
        ),
        executor_cls=registration.executor_cls,
        runtime_builder=registration.runtime_builder,
        runtime_spec_extractor=registration.runtime_spec_extractor,
        gatherer=GathererMetadata(
            import_path=registration.gatherer.import_path,
            kwargs=dict(registration.gatherer.kwargs),
        ),
        capability=registration.capability,
        executor_kwargs=ExecutorKwargsMetadata(
            include_sample_batch_size=(
                registration.executor_kwargs.include_sample_batch_size
            ),
            include_reference_image=registration.executor_kwargs.include_reference_image,
        ),
    )


def _build_family_registry() -> dict[str, RolloutFamilyEntry]:
    _load_builtin_model_registrations()
    entries: dict[str, RolloutFamilyEntry] = {}
    for model_registration in registered_models():
        for rollout_registration in model_registration.rollouts:
            entry = _rollout_entry_from_model_registration(rollout_registration)
            if entry.family in entries:
                raise ValueError(f"duplicate rollout family registration: {entry.family!r}")
            entries[entry.family] = entry
    return entries


FAMILY_REGISTRY: dict[str, RolloutFamilyEntry] = _build_family_registry()

_FAMILY_ALIASES: dict[str, str] = {
    alias: family
    for family, entry in FAMILY_REGISTRY.items()
    for alias in (family, *entry.aliases)
}


def normalize_rollout_family(family: str) -> str:
    """Return the canonical registry key for a rollout family or alias."""

    text = str(family)
    return _FAMILY_ALIASES.get(text, text)


def get_rollout_family_entry(family: str) -> RolloutFamilyEntry:
    """Return the canonical rollout family entry for ``family``."""

    normalized = normalize_rollout_family(family)
    try:
        return FAMILY_REGISTRY[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unsupported rollout family: {family!r}; registered={sorted(FAMILY_REGISTRY)}",
        ) from exc


def registered_rollout_families() -> tuple[str, ...]:
    """Return canonical rollout family keys."""

    return tuple(FAMILY_REGISTRY)


__all__ = [
    "DIFFUSION_COMMON_SAMPLING_FIELDS",
    "DIFFUSION_MIGRATED_RETURN_ARTIFACTS",
    "DIFFUSION_VIDEO_SAMPLING_FIELDS",
    "FAMILY_REGISTRY",
    "JANUS_PRO_MIGRATED_RETURN_ARTIFACTS",
    "JANUS_PRO_R1_MIGRATED_RETURN_ARTIFACTS",
    "NEXTSTEP_MIGRATED_RETURN_ARTIFACTS",
    "SD3_MIGRATED_RETURN_ARTIFACTS",
    "CollectorKind",
    "CollectorMetadata",
    "ExecutorKwargsMetadata",
    "FamilyCapability",
    "GathererMetadata",
    "RolloutFamilyEntry",
    "get_rollout_family_entry",
    "normalize_rollout_family",
    "registered_rollout_families",
]
