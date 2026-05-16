"""Shared capability template for diffusion model families."""

from __future__ import annotations

from vrl.engine.core.capabilities import (
    AxisCapability,
    ExecutionUnitCapability,
    FamilyCapability,
)


def diffusion_family_capability(
    family: str,
    task: str,
    *,
    trainable_segment: str = "denoise",
    supports_reference_conditioning: bool = False,
) -> FamilyCapability:
    """Capability template for diffusion timestep rollouts."""

    return FamilyCapability(
        family=family,
        task=task,
        trajectory_kind="diffusion",
        expected_axes=(
            AxisCapability("sample", "sample", batchable=True, chunkable=True),
            AxisCapability("timestep", "denoise_step", batchable=True, chunkable=False),
        ),
        execution_units=(
            ExecutionUnitCapability(
                "denoise_step",
                segment=trainable_segment,
                axis="timestep",
                cache_read=True,
                cache_write=True,
            ),
            ExecutionUnitCapability("vq_decode", segment=trainable_segment),
            ExecutionUnitCapability("reward_artifact", profiler_name="collector.reward_score"),
        ),
        trainable_segments=(trainable_segment,),
        reward_views=("image",),
        supports_stepwise=True,
        supports_cfg=True,
        supports_batched_decode=True,
        supports_reference_conditioning=supports_reference_conditioning,
        supports_resident_rollout_state=True,
        cache_kinds=("prompt_embed_cache", "latent_cache"),
    )


__all__ = ["diffusion_family_capability"]
