"""Reward model implementation ownership and structural contract shape."""

from __future__ import annotations

import inspect

import pytest

from vrl.rewards.models.base import RewardModel
from vrl.rewards.models.cosmos3_reasoner import Cosmos3ReasonerRewardModel
from vrl.rewards.models.kling_video_reward import KlingVideoRewardModel
from vrl.rewards.models.phymotion import PhyMotionModel
from vrl.rewards.models.unified_reward_video import UnifiedRewardVideoModel
from vrl.rewards.models.videocon_physics import VideoConPhysicsModel
from vrl.rewards.models.videoscore2 import VideoScore2Model

_REWARD_MODEL_CLASSES = (
    VideoScore2Model,
    VideoConPhysicsModel,
    KlingVideoRewardModel,
    PhyMotionModel,
    Cosmos3ReasonerRewardModel,
    UnifiedRewardVideoModel,
)


@pytest.mark.parametrize("model_class", _REWARD_MODEL_CLASSES)
def test_concrete_reward_model_keeps_protocol_out_of_its_mro(model_class: type) -> None:
    assert RewardModel not in model_class.__mro__


@pytest.mark.parametrize("model_class", _REWARD_MODEL_CLASSES)
def test_concrete_reward_model_owns_the_structural_call_shape(model_class: type) -> None:
    call = model_class.__dict__["__call__"]
    parameters = inspect.signature(call).parameters

    assert parameters["artifact"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert "request" not in parameters
