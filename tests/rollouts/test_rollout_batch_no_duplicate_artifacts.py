"""Regression tests for keeping reward artifacts out of trainer state."""

from __future__ import annotations

import torch

from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.collector.artifacts import (
    RewardArtifactPolicy,
    release_reward_artifact_if_needed,
    reward_artifact_bytes,
)


def test_release_policy_removes_videos_without_mutating_trainer_context() -> None:
    artifact = torch.ones(2, 3, 4, 4)
    context = {"reward_metadata": {"source": "unit"}}
    batch = RolloutBatch(
        observations=torch.zeros(2, 1),
        actions=torch.zeros(2, 1),
        rewards=torch.zeros(2),
        dones=torch.ones(2, dtype=torch.bool),
        group_ids=torch.tensor([0, 0]),
        context=context,
        videos=artifact,
    )

    release_reward_artifact_if_needed(
        batch,
        RewardArtifactPolicy(keep_after_reward=False),
    )

    assert batch.videos is None
    assert batch.context == context
    assert reward_artifact_bytes(batch.context) == 0
