"""CPU unit tests for the Future Reward functions (SPRINT_future_reward).

Reward discrimination quality (exact clip vs blur/static/shuffle hacks) is judged
by a human reviewing rollouts, not an automated probe.
"""

from __future__ import annotations

import pytest

from vrl.rewards.functions.motion_dynamics import MotionDynamicsReward
from vrl.rewards.functions.target_dino_similarity import TargetDinoSimilarityReward


@pytest.mark.parametrize(
    "reward_cls",
    [MotionDynamicsReward, TargetDinoSimilarityReward],
)
def test_future_reward_accepts_declared_worker_config(reward_cls: type) -> None:
    reward = reward_cls(device="cpu", worker_config={"num_frames": 2})
    assert reward.scorer is not None


@pytest.mark.parametrize(
    "reward_cls",
    [MotionDynamicsReward, TargetDinoSimilarityReward],
)
def test_future_reward_rejects_unknown_kwargs(reward_cls: type) -> None:
    with pytest.raises(TypeError, match="unknown_knob"):
        reward_cls(device="cpu", unknown_knob=True)
