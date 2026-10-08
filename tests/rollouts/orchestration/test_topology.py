"""Pre-launch rollout schedule topology validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vrl.rollouts.orchestration import validate_rollout_schedule_topology


def _resources(
    *,
    colocated: bool,
    trainer_rollout_handoff: bool = False,
    reward_handoff: bool = False,
    trainer_reward_handoff: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        colocated=colocated,
        lifecycle=SimpleNamespace(
            park_trainer_for_rollout=trainer_rollout_handoff,
            park_rollout_for_reward=reward_handoff,
            park_trainer_for_reward=trainer_reward_handoff,
        ),
    )


@pytest.mark.parametrize(
    ("schedule_mode", "colocated"),
    [("strict_on_policy", True), ("continuous", False)],
)
def test_supported_topologies_pass_validation(schedule_mode: str, colocated: bool) -> None:
    """Strict on a shared GPU and continuous on disjoint GPUs are the two
    supported pairings; validation is silent for both."""

    validate_rollout_schedule_topology(
        SimpleNamespace(schedule_mode=schedule_mode),
        _resources(colocated=colocated),
    )


def test_continuous_shared_gpu_is_rejected() -> None:
    with pytest.raises(ValueError, match="disjoint trainer and rollout GPUs"):
        validate_rollout_schedule_topology(
            SimpleNamespace(schedule_mode="continuous"),
            _resources(colocated=True),
        )


def test_continuous_trainer_parked_for_generation_is_rejected() -> None:
    """An explicit trainer offload on disjoint GPUs still cannot overlap backward."""

    with pytest.raises(ValueError, match="without trainer parking for generation"):
        validate_rollout_schedule_topology(
            SimpleNamespace(schedule_mode="continuous"),
            _resources(colocated=False, trainer_rollout_handoff=True),
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

    validate_rollout_schedule_topology(
        SimpleNamespace(schedule_mode="continuous"),
        _resources(colocated=False),
    )


def test_continuous_rejects_a_reward_sharing_the_trainer_gpu() -> None:
    """A GPU reward with no reservation of its own follows the trainer's card in
    the real resolved plan, so scoring would need the trainer to yield its GPU;
    continuous scheduling overlaps backward with scoring and cannot."""

    from omegaconf import OmegaConf

    from vrl.config.schema import parse_config
    from vrl.ray.resources import ResolvedDistributedResources
    from vrl.trainers.core.types import RolloutOrchestrationConfig

    resources = ResolvedDistributedResources.from_root(
        parse_config(
            OmegaConf.create(
                {
                    "distributed": {
                        "resources": {
                            "visible_devices": [0, 1],
                            "trainer": {"devices": [0]},
                            "rollout": {"devices": [1]},
                        },
                    },
                    "reward": {"components": {"aesthetic": 1.0}, "kwargs": {"aesthetic": {}}},
                },
            ),
        ),
    )
    assert resources.reward_follows_trainer
    assert resources.lifecycle.park_trainer_for_reward

    with pytest.raises(ValueError, match="reward scoring on the trainer GPU"):
        validate_rollout_schedule_topology(
            RolloutOrchestrationConfig(schedule_mode="continuous"),
            resources,
        )
