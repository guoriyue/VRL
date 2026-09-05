"""CPU unit tests for the Future Reward functions (SPRINT_future_reward).

Reward discrimination quality (exact clip vs blur/static/shuffle hacks) is judged
by a human reviewing rollouts, not an automated probe.
"""

from __future__ import annotations

import pytest

from vrl.rewards.functions.motion_dynamics import MotionDynamicsReward


def test_future_reward_rejects_unknown_kwargs() -> None:
    with pytest.raises(TypeError, match="unknown_knob"):
        MotionDynamicsReward(device="cpu", unknown_knob=True)
