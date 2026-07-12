"""Pre-launch rollout schedule topology validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vrl.rollouts.orchestration import validate_rollout_schedule_topology


def _resources(
    *,
    colocated: bool,
    reward_handoff: bool = False,
    trainer_reward_handoff: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        colocated=colocated,
        lifecycle=SimpleNamespace(
            handoff=SimpleNamespace(
                release_rollout_before_reward=reward_handoff,
                release_trainer_before_reward=trainer_reward_handoff,
            ),
        ),
    )


def test_strict_shared_gpu_is_allowed() -> None:
    validate_rollout_schedule_topology(
        SimpleNamespace(schedule_mode="strict_on_policy"),
        _resources(colocated=True),
    )


def test_continuous_shared_gpu_is_rejected() -> None:
    with pytest.raises(ValueError, match="disjoint trainer and rollout GPUs"):
        validate_rollout_schedule_topology(
            SimpleNamespace(schedule_mode="continuous"),
            _resources(colocated=True),
        )


def test_continuous_reward_handoff_is_rejected() -> None:
    with pytest.raises(ValueError, match="reward scoring"):
        validate_rollout_schedule_topology(
            SimpleNamespace(schedule_mode="continuous"),
            _resources(colocated=False, reward_handoff=True),
        )


def test_continuous_trainer_reward_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="reward scoring on the trainer GPU"):
        validate_rollout_schedule_topology(
            SimpleNamespace(schedule_mode="continuous"),
            _resources(colocated=False, trainer_reward_handoff=True),
        )


def test_continuous_disjoint_topology_is_allowed() -> None:
    validate_rollout_schedule_topology(
        SimpleNamespace(schedule_mode="continuous"),
        _resources(colocated=False),
    )
