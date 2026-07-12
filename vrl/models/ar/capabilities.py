"""Shared capability templates for autoregressive model families."""

from __future__ import annotations

from vrl.generation.capabilities import FamilyCapability, TrajectoryKind


def ar_discrete_family_capability(
    family: str,
    task: str,
    *,
    trajectory_kind: TrajectoryKind = "ar_discrete",
) -> FamilyCapability:
    """Capability template for discrete-token AR image generation.

    Production AR models inherit ``ARModelBase`` and register their CUDA modules,
    so ``nn.Module.to`` is the complete CPU fallback. Worker startup checks the
    concrete model and post-sleep physical memory before a phase handoff commits.
    """

    return FamilyCapability(
        family=family,
        task=task,
        trajectory_kind=trajectory_kind,
    )


def ar_continuous_family_capability(
    family: str,
    task: str,
) -> FamilyCapability:
    """Capability template for continuous-token AR image generation.

    Continuous-token families share the same registered ``ARModelBase`` parking
    contract and runtime residual validation as discrete-token families.
    """

    return FamilyCapability(
        family=family,
        task=task,
        trajectory_kind="ar_continuous",
    )


__all__ = [
    "ar_continuous_family_capability",
    "ar_discrete_family_capability",
]
