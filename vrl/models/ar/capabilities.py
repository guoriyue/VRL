"""Shared capability templates for autoregressive model families."""

from __future__ import annotations

from vrl.generation.capabilities import FamilyCapability, TrajectoryKind


def ar_discrete_family_capability(
    family: str,
    task: str,
    *,
    trajectory_kind: TrajectoryKind = "ar_discrete",
) -> FamilyCapability:
    """Capability template for discrete-token AR image generation."""

    return FamilyCapability(
        family=family,
        task=task,
        trajectory_kind=trajectory_kind,
    )


def ar_continuous_family_capability(
    family: str,
    task: str,
) -> FamilyCapability:
    """Capability template for continuous-token AR image generation."""

    return FamilyCapability(
        family=family,
        task=task,
        trajectory_kind="ar_continuous",
    )


__all__ = [
    "ar_continuous_family_capability",
    "ar_discrete_family_capability",
]
