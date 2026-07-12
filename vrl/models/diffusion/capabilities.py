"""Shared capability template for diffusion model families."""

from __future__ import annotations

from vrl.generation.capabilities import FamilyCapability


def diffusion_family_capability(
    family: str,
    task: str,
) -> FamilyCapability:
    """Capability template for diffusion timestep rollouts.

    Every production diffusion model inherits ``DiffusionModelBase``: registered
    state follows ``nn.Module.to`` and unregistered pipeline components follow its
    ``move_frozen_components`` contract. Worker startup checks those methods again,
    and the post-sleep physical snapshot is the behavioral backstop.
    """

    return FamilyCapability(
        family=family,
        task=task,
        trajectory_kind="diffusion",
    )


__all__ = ["diffusion_family_capability"]
