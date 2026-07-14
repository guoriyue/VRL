"""Shared registry-derived fixtures for model interface contract tests."""

from __future__ import annotations

from vrl.families.registry import FAMILY_REGISTRY, ARFamilyBuild, DiffusionFamilyBuild
from vrl.utils.config import import_from_path

# Custom replay construction is the only place the registry cannot derive the
# concrete replay class from the generic descriptor.
_CUSTOM_REPLAY_MODEL_CLASSES = {
    "cosmos-predict2-anima": "vrl.models.diffusion.cosmos.anima.model:AnimaReplayModel",
    "cosmos3": "vrl.models.diffusion.cosmos.cosmos3.model:Cosmos3ReplayModel",
    "echo": "vrl.models.diffusion.echo.model:EchoReplayModel",
}


def registered_family_model_classes() -> dict[str, tuple[type, type]]:
    """Resolve every registered family's runtime and replay model classes."""

    resolved: dict[str, tuple[type, type]] = {}
    for family, entry in FAMILY_REGISTRY.items():
        build = entry.family_build
        if isinstance(build, ARFamilyBuild):
            runtime_path = build.model_cls
            replay_path = build.replay_cls
        else:
            assert isinstance(build, DiffusionFamilyBuild)
            runtime_path = build.model_cls
            replay_path = build.replay_cls or _CUSTOM_REPLAY_MODEL_CLASSES.get(family)
            if replay_path is None:
                raise AssertionError(
                    f"custom replay family {family!r} lacks a contract-test model class",
                )
        resolved[family] = (
            import_from_path(runtime_path),
            import_from_path(replay_path),
        )
    return resolved


__all__ = ["registered_family_model_classes"]
