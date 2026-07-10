"""Shared capability template for diffusion model families."""

from __future__ import annotations

from vrl.generation.capabilities import FamilyCapability


def diffusion_family_capability(
    family: str,
    task: str,
    *,
    supports_reference_conditioning: bool = False,
) -> FamilyCapability:
    """Capability template for diffusion timestep rollouts."""

    return FamilyCapability(
        family=family,
        task=task,
        trajectory_kind="diffusion",
        supports_reference_conditioning=supports_reference_conditioning,
        supports_torch_compile=True,
    )


__all__ = ["diffusion_family_capability"]
