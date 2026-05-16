"""Explicit rollout collector registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vrl.engine import RolloutBackend
from vrl.rollouts.collector.core import RolloutCollector
from vrl.rollouts.collector.requests import (
    RolloutEngineRequestBuilder,
    RolloutRequestBuilder,
)
from vrl.rollouts.collector.rewards import RewardScorer
from vrl.rollouts.family_registry import (
    CollectorKind,
    FAMILY_REGISTRY,
    normalize_rollout_family,
)

LAST_COLLECT_PHASES: dict[str, float] = {}


@dataclass(frozen=True, slots=True)
class CollectorRegistryEntry:
    """Declarative binding from explicit family name to collector components."""

    family: str
    task: str
    kind: CollectorKind
    request_prefix: str | None = None
    default_task_type: str | None = None
    return_artifacts: tuple[str, ...] = ()
    metadata_key: str | None = None


COLLECTOR_REGISTRY: dict[str, CollectorRegistryEntry] = {
    family: CollectorRegistryEntry(
        family=entry.family,
        task=entry.task,
        kind=entry.collector.kind,
        request_prefix=entry.collector.request_prefix,
        default_task_type=entry.collector.default_task_type,
        return_artifacts=entry.collector.return_artifacts,
        metadata_key=entry.collector.metadata_key,
    )
    for family, entry in FAMILY_REGISTRY.items()
}


def build_rollout_collector(
    family: str,
    *,
    model: Any | None,
    reward_fn: Any | None,
    config: Any | None = None,
    runtime: RolloutBackend | None = None,
    reference_image: Any = None,
) -> RolloutCollector:
    """Build a rollout collector from an explicit family registry key."""

    registry_key = normalize_rollout_family(family)
    entry = _entry_for(registry_key)
    settings = _resolve_settings(entry, config)
    request_builder = _build_request_builder(entry, settings)
    del reference_image

    return RolloutCollector(
        model=model,
        config=settings,
        family=entry.family,
        task=entry.task,
        request_builder=request_builder,
        reward_scorer=RewardScorer(reward_fn),
        default_group_size=_default_group_size(entry, settings),
        runtime=runtime,
        phase_sink=LAST_COLLECT_PHASES,
    )


def _entry_for(family: str) -> CollectorRegistryEntry:
    try:
        return COLLECTOR_REGISTRY[family]
    except KeyError as exc:
        raise NotImplementedError(
            f"no rollout collector registered for family={family!r}; "
            f"registered={sorted(COLLECTOR_REGISTRY)}",
        ) from exc


def _resolve_settings(entry: CollectorRegistryEntry, config: Any | None) -> Any:
    if config is None:
        raise ValueError(
            f"{entry.family} collector requires resolved rollout settings; "
            "build them from YAML before constructing the collector",
        )
    return config


def _build_request_builder(
    entry: CollectorRegistryEntry,
    config: Any,
) -> RolloutRequestBuilder:
    if entry.request_prefix is None:
        raise ValueError(f"{entry.family} collector registry entry is incomplete")
    return RolloutEngineRequestBuilder(
        family=entry.family,
        task=entry.task,
        request_prefix=entry.request_prefix,
        config=config,
        return_artifacts=entry.return_artifacts,
        default_task_type=entry.default_task_type,
        metadata_key=entry.metadata_key,
    )


def _default_group_size(entry: CollectorRegistryEntry, config: Any) -> int:
    if entry.kind == "diffusion":
        return 1
    return int(_require_config_value(entry, config, "n_samples_per_prompt"))


def _require_config_value(
    entry: CollectorRegistryEntry,
    config: Any,
    name: str,
) -> Any:
    require = getattr(config, "require", None)
    if callable(require):
        return require(name)
    try:
        return getattr(config, name)
    except AttributeError as exc:
        raise ValueError(
            f"{entry.family} collector config missing required field {name!r}",
        ) from exc


__all__ = [
    "COLLECTOR_REGISTRY",
    "LAST_COLLECT_PHASES",
    "CollectorRegistryEntry",
    "build_rollout_collector",
]
