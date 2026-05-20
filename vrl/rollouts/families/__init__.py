"""Rollout family registry and launch input wiring."""

from vrl.rollouts.families.registry import (
    FAMILY_REGISTRY,
    CollectorKind,
    CollectorMetadata,
    ExecutorKwargsMetadata,
    GathererMetadata,
    RayGenerationLaunchInputs,
    RolloutFamilyEntry,
    build_ray_generation_inputs_for_family,
    get_rollout_family_entry,
    normalize_rollout_family,
    register_rollout_family,
    registered_rollout_families,
)

__all__ = [
    "FAMILY_REGISTRY",
    "CollectorKind",
    "CollectorMetadata",
    "ExecutorKwargsMetadata",
    "GathererMetadata",
    "RayGenerationLaunchInputs",
    "RolloutFamilyEntry",
    "build_ray_generation_inputs_for_family",
    "get_rollout_family_entry",
    "normalize_rollout_family",
    "register_rollout_family",
    "registered_rollout_families",
]
