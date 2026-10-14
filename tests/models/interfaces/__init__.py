"""Shared registry-derived fixtures for model interface contract tests."""

from __future__ import annotations

from vrl.models.families.registry import FAMILY_REGISTRY, DenoiseFamilyBuild
from vrl.utils.config import import_from_path


def registered_runtime_model_classes() -> dict[str, type]:
    """Resolve every registered family's runtime model class."""

    return {
        family: import_from_path(entry.family_build.model_cls)
        for family, entry in FAMILY_REGISTRY.items()
    }


def registered_replay_model_classes() -> dict[str, type]:
    """Resolve replay model classes only for families with a replay recipe."""

    resolved: dict[str, type] = {}
    for family, entry in FAMILY_REGISTRY.items():
        if not entry.supports_policy_replay:
            continue
        build = entry.family_build
        assert isinstance(build, DenoiseFamilyBuild)
        resolved[family] = import_from_path(build.replay_cls)
    return resolved


__all__ = [
    "registered_replay_model_classes",
    "registered_runtime_model_classes",
]
