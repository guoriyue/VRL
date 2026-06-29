"""The diffusion rollout stage topology (SGLang-Omni-shape stage graph)."""

from __future__ import annotations

from vrl.generation.diffusion.pipeline import (
    DECODE,
    DENOISE,
    PREPARE,
    PROMPT_ENCODE,
    REWARD,
    build_diffusion_pipeline_topology,
)


def _entry(topology) -> str:
    value = topology.resolved_entry_stage
    return value() if callable(value) else value


def _terminals(topology) -> tuple[str, ...]:
    value = topology.terminal_stages
    return tuple(value() if callable(value) else value)


def test_default_topology_is_the_linear_rollout_graph() -> None:
    """Default (single-GPU) topology is encode->prepare->denoise->decode->reward."""
    topology = build_diffusion_pipeline_topology()

    assert [stage.name for stage in topology.stages] == [
        PROMPT_ENCODE,
        PREPARE,
        DENOISE,
        DECODE,
        REWARD,
    ]
    assert _entry(topology) == PROMPT_ENCODE
    assert _terminals(topology) == (REWARD,)
    assert topology.stage_for(DENOISE).next_stages == (DECODE,)
    assert topology.stage_for(DECODE).next_stages == (REWARD,)
    assert topology.stage_for(REWARD).terminal is True


def test_default_topology_pins_no_gpu_so_single_card_is_unchanged() -> None:
    """No placement by default: every stage inherits the rollout GPU (gpu_ids empty)."""
    topology = build_diffusion_pipeline_topology()

    assert all(stage.gpu_ids == () for stage in topology.stages)


def test_per_stage_placement_pins_denoise_and_reward_to_distinct_gpus() -> None:
    """The multi-GPU lever: denoise and reward land on different GPU ordinals so
    their tensor-core compute can run in parallel (1/max(stage), not the sum)."""
    topology = build_diffusion_pipeline_topology(
        gpu_ids={DENOISE: (0, 1), REWARD: (2,)},
    )

    assert topology.stage_for(DENOISE).gpu_ids == (0, 1)
    assert topology.stage_for(REWARD).gpu_ids == (2,)
    # Unplaced stages still inherit the rollout GPU.
    assert topology.stage_for(PREPARE).gpu_ids == ()
    # Reward is disjoint from denoise — the placement that enables the overlap.
    assert set(topology.stage_for(REWARD).gpu_ids).isdisjoint(
        topology.stage_for(DENOISE).gpu_ids,
    )
