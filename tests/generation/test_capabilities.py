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
        axis="denoise",
    )

    data = stage.to_dict()
    restored = ExecutionStageCapability.from_value(data)

    assert data["profiler_name"] is None
    assert restored == stage
    assert restored.profiler_name is None
    # profiler_label falls back to f"engine.{name}" when profiler_name is unset;
    # derive the expected from stage.name so a prefix-template rename does not
    # break the fallback-behavior contract.
    assert restored.profiler_label == f"engine.{stage.name}"


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
        metadata={"source": "unit"},
    )

    restored = FamilyCapability.from_value(capability.to_dict())

    assert restored == capability
    assert restored.execution_stages[0].profiler_name is None
    prepare_stage, decode_stage = capability.execution_stages
    # First stage has no profiler_name -> engine.<name> fallback (derived);
    # second stage's label is the explicitly-set profiler_name input (echoed).
    assert restored.profiler_labels == (
        f"engine.{prepare_stage.name}",
        decode_stage.profiler_name,
    )
