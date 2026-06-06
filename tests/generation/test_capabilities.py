"""Tests for generation capability serialization."""

from __future__ import annotations

from vrl.generation.capabilities import (
    AxisCapability,
    ExecutionStageCapability,
    FamilyCapability,
)


def test_execution_stage_capability_round_trip_preserves_default_profiler_name() -> None:
    """Checks execution stage capability round trip preserves default profiler name."""
    stage = ExecutionStageCapability(
        name="denoise_step",
        segment="denoise",
        axis="timestep",
        metadata={"phase": "sample"},
    )

    data = stage.to_dict()
    restored = ExecutionStageCapability.from_value(data)

    assert data["profiler_name"] is None
    assert restored == stage
    assert restored.profiler_name is None
    assert restored.profiler_label == "engine.denoise_step"


def test_family_capability_round_trip_preserves_stage_profiler_names() -> None:
    """Checks family capability round trip preserves stage profiler names."""
    capability = FamilyCapability(
        family="unit",
        task="t2v",
        trajectory_kind="diffusion",
        expected_axes=(
            AxisCapability(
                name="sample",
                kind="sample",
                batchable=True,
                chunkable=True,
            ),
        ),
        execution_stages=(
            ExecutionStageCapability(name="prepare"),
            ExecutionStageCapability(
                name="decode",
                profiler_name="generation.decode_latents",
            ),
        ),
        trainable_segments=("denoise",),
        reward_views=("video",),
        cache_kinds=("kv",),
        metadata={"source": "unit"},
    )

    restored = FamilyCapability.from_value(capability.to_dict())

    assert restored == capability
    assert restored.execution_stages[0].profiler_name is None
    assert restored.profiler_labels == ("engine.prepare", "generation.decode_latents")
