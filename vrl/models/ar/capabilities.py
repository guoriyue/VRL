"""Shared capability templates for autoregressive model families."""

from __future__ import annotations

from vrl.generation.capabilities import (
    AxisCapability,
    ExecutionStageCapability,
    FamilyCapability,
    TrajectoryKind,
)


def ar_discrete_family_capability(
    family: str,
    task: str,
    *,
    trajectory_kind: TrajectoryKind = "ar_discrete",
    trainable_segments: tuple[str, ...] = ("image_tokens",),
) -> FamilyCapability:
    """Capability template for discrete-token AR image generation."""

    if not trainable_segments:
        raise ValueError("trainable_segments must be non-empty")
    return FamilyCapability(
        family=family,
        task=task,
        trajectory_kind=trajectory_kind,
        expected_axes=(
            AxisCapability("sample", "sample", batchable=True, chunkable=True),
            AxisCapability("token", "discrete_token", batchable=True, chunkable=False),
        ),
        execution_stages=(
            ExecutionStageCapability(
                "prefill",
                segment=trainable_segments[0],
                cache_write=True,
            ),
            ExecutionStageCapability(
                "decode_step",
                segment=trainable_segments[0],
                axis="token",
                cache_read=True,
                cache_write=True,
            ),
            ExecutionStageCapability("vq_decode", segment=trainable_segments[-1]),
            ExecutionStageCapability(
                "reward_artifact",
                profiler_name="collector.reward_score",
            ),
        ),
        trainable_segments=trainable_segments,
        reward_views=("image",),
        cache_kinds=("kv_cache", "prompt_embed_cache", "token_buffer"),
    )


def ar_continuous_family_capability(
    family: str,
    task: str,
    *,
    trainable_segment: str = "image_tokens",
) -> FamilyCapability:
    """Capability template for continuous-token AR image generation."""

    return FamilyCapability(
        family=family,
        task=task,
        trajectory_kind="ar_continuous",
        expected_axes=(
            AxisCapability("sample", "sample", batchable=True, chunkable=True),
            AxisCapability("token", "continuous_token", batchable=True, chunkable=False),
        ),
        execution_stages=(
            ExecutionStageCapability(
                "prefill",
                segment=trainable_segment,
                cache_write=True,
            ),
            ExecutionStageCapability(
                "decode_step",
                segment=trainable_segment,
                axis="token",
                cache_read=True,
                cache_write=True,
            ),
            ExecutionStageCapability("vq_decode", segment=trainable_segment),
            ExecutionStageCapability(
                "reward_artifact",
                profiler_name="collector.reward_score",
            ),
        ),
        trainable_segments=(trainable_segment,),
        reward_views=("image",),
        cache_kinds=("prompt_embed_cache", "token_buffer"),
    )


__all__ = [
    "ar_continuous_family_capability",
    "ar_discrete_family_capability",
]
