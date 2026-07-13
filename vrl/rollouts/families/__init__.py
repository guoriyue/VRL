"""Rollout family registry and external-name normalization."""

from vrl.rollouts.families.registry import (
    FAMILY_REGISTRY,
    CollectorKind,
    CollectorMetadata,
    GathererMetadata,
    RolloutFamilyEntry,
    get_rollout_family_entry,
    normalize_rollout_family,
    register_rollout_family,
    registered_rollout_families,
    resolve_entry_model_build,
)

__all__ = [
    "FAMILY_REGISTRY",
    "CollectorKind",
    "CollectorMetadata",
    "GathererMetadata",
    "RolloutFamilyEntry",
    "get_rollout_family_entry",
    "normalize_rollout_family",
    "register_rollout_family",
    "registered_rollout_families",
    "resolve_entry_model_build",
]
