"""The diffusion generation stage topology (SGLang-Omni-shape stage graph)."""

from __future__ import annotations

from vrl.generation.diffusion.pipeline import (
    DECODE,
    DENOISE,
    PREPARE,
    PROMPT_ENCODE,
    build_diffusion_pipeline_topology,
)


def _entry(topology) -> str:
    value = topology.resolved_entry_stage
    return value() if callable(value) else value


def _terminals(topology) -> tuple[str, ...]:
    value = topology.terminal_stages
    return tuple(value() if callable(value) else value)


def test_default_topology_is_the_linear_generation_graph() -> None:
    """Default topology is encode->prepare->denoise->decode (decode terminal)."""
    topology = build_diffusion_pipeline_topology()

    assert [stage.name for stage in topology.stages] == [
        PROMPT_ENCODE,
        PREPARE,
        DENOISE,
        DECODE,
    ]
    assert _entry(topology) == PROMPT_ENCODE
    assert _terminals(topology) == (DECODE,)
    assert topology.stage_for(DENOISE).next_stages == (DECODE,)
    assert topology.stage_for(DECODE).terminal is True


def test_default_topology_pins_no_gpu_so_single_card_is_unchanged() -> None:
    """No placement by default: every stage inherits the rollout GPU (gpu_ids empty)."""
    topology = build_diffusion_pipeline_topology()

    assert all(stage.gpu_ids == () for stage in topology.stages)


def test_per_stage_placement_pins_stages_to_distinct_gpus() -> None:
    """The multi-GPU lever: two stages land on different GPU ordinals so their
    tensor-core compute can run in parallel (1/max(stage), not the sum)."""
    topology = build_diffusion_pipeline_topology(
        gpu_ids={DENOISE: (0, 1), DECODE: (2,)},
    )

    assert topology.stage_for(DENOISE).gpu_ids == (0, 1)
    assert topology.stage_for(DECODE).gpu_ids == (2,)
    # Unplaced stages still inherit the rollout GPU.
    assert topology.stage_for(PREPARE).gpu_ids == ()
    # Placed stages are disjoint — the placement that enables the overlap.
    assert set(topology.stage_for(DECODE).gpu_ids).isdisjoint(
        topology.stage_for(DENOISE).gpu_ids,
    )
