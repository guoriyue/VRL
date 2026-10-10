"""Pre-launch rollout schedule validation on the resolved resource plan."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.config.schema import parse_config
from vrl.ray.resources import ResolvedDistributedResources


def _resources(
    *,
    visible: list[int],
    trainer: list[int],
    rollout: list[int],
    reward: list[int] | None = None,
) -> ResolvedDistributedResources:
    """A real plan over one GPU reward (``aesthetic``), with or without its own card."""

    resources: dict[str, object] = {
        "visible_devices": visible,
        "trainer": {"devices": trainer},
        "rollout": {"devices": rollout},
    }
    if reward is not None:
        resources["reward"] = {"devices": reward}
    return ResolvedDistributedResources.from_root(
        parse_config(
            OmegaConf.create(
                {
                    "distributed": {"resources": resources},
                    "reward": {"components": {"aesthetic": 1.0}, "kwargs": {"aesthetic": {}}},
                },
            ),
        ),
    )


def test_strict_on_a_shared_gpu_and_continuous_on_disjoint_gpus_pass() -> None:
    """The two supported pairings are silent."""

    _resources(visible=[0], trainer=[0], rollout=[0]).validate_schedule_mode("strict_on_policy")
    _resources(visible=[0, 1, 2], trainer=[0], rollout=[1], reward=[2]).validate_schedule_mode(
        "continuous"
    )


def test_continuous_shared_gpu_is_rejected() -> None:
    with pytest.raises(ValueError, match="disjoint trainer and rollout GPUs"):
        _resources(visible=[0], trainer=[0], rollout=[0]).validate_schedule_mode("continuous")


def test_continuous_rejects_a_reward_sharing_the_rollout_gpu() -> None:
    """Scoring on the rollout card would hand it over mid-iteration."""

    with pytest.raises(ValueError, match="hand the rollout GPU to reward scoring"):
        _resources(visible=[0, 1], trainer=[0], rollout=[1], reward=[1]).validate_schedule_mode(
            "continuous"
        )


def test_continuous_rejects_a_reward_sharing_the_trainer_gpu() -> None:
    """A GPU reward with no reservation of its own follows the trainer's card, so
    scoring would need the trainer to yield its GPU; continuous scheduling
    overlaps backward with scoring and cannot."""

    resources = _resources(visible=[0, 1], trainer=[0], rollout=[1])
    assert resources.reward_follows_trainer
    assert resources.lifecycle.park_trainer_for_reward

    with pytest.raises(ValueError, match="reward scoring on the trainer GPU"):
        resources.validate_schedule_mode("continuous")
