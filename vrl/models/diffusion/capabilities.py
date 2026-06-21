"""Shared capability template for diffusion model families."""

from __future__ import annotations

from vrl.generation.capabilities import (
    AxisCapability,
    ExecutionStageCapability,
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
            AxisCapability("denoise", "denoise_step", batchable=True, chunkable=False),
        ),
        execution_stages=(
            ExecutionStageCapability(
                "prompt_encode",
                profiler_name="generation.prompt_encode",
            ),
            ExecutionStageCapability(
                "prepare_sampling",
                profiler_name="generation.prepare_sampling",
            ),
            ExecutionStageCapability(
                "denoise_step",
                segment=trainable_segment,
                axis="denoise",
                profiler_name="generation.denoise_step",
            ),
            ExecutionStageCapability(
                "decode_latents",
                segment=trainable_segment,
                profiler_name="generation.decode_latents",
            ),
            ExecutionStageCapability(
                "reward_artifact",
                profiler_name="collector.reward_score",
            ),
        ),
        supports_reference_conditioning=supports_reference_conditioning,
        supports_torch_compile=True,
    )


__all__ = ["diffusion_family_capability"]
