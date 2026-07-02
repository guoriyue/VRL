"""Rewards score in-process; the removed pool transport fails loud."""

from __future__ import annotations

import pytest

from vrl.rewards.functions.pickscore import PickScoreReward
from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.runtime import LocalRewardRuntime, make_reward_runtime

_FACTORY = "vrl.rewards.models.pickscore:pickscore_reward_model"


def _request() -> RewardInferenceRequest:
    return RewardInferenceRequest(
        request_id="req",
        artifacts=(
            RewardInferenceArtifact(
                artifact_id="a0", path="/tmp/a0.mp4", media_type="video", prompt="p",
            ),
        ),
        reward_name="fake",
        score_key="overall",
    )


def test_make_reward_runtime_inline() -> None:
    """Checks make reward runtime inline -> in-process transport."""
    runtime = make_reward_runtime(
        "inline", model_factory=_FACTORY, worker_config={"device": "cpu"},
    )
    assert isinstance(runtime, LocalRewardRuntime)


def test_make_reward_runtime_rejects_removed_pool() -> None:
    """Checks the removed pool transport fails loud with the migration hint."""
    with pytest.raises(ValueError, match="sleep_offload"):
        make_reward_runtime("pool", model_factory=_FACTORY)


def test_make_reward_runtime_rejects_unknown() -> None:
    """Checks make reward runtime rejects unknown."""
    with pytest.raises(ValueError, match="execution"):
        make_reward_runtime("bogus", model_factory=_FACTORY)


def test_reward_class_uses_inline_transport() -> None:
    """Checks reward class builds the in-process transport."""
    reward = PickScoreReward(device="cpu")
    assert isinstance(reward.runtime, LocalRewardRuntime)


class _MovableModel:
    """Fake reward model recording .to() moves (sleep_offload contract)."""

    def __init__(self) -> None:
        self.device = "cuda:7"
        self.moves: list[str] = []

    def to(self, device: str) -> None:
        self.moves.append(device)
        self.device = device

    def __call__(self, *, artifact, request):
        return {"overall": 1.0}


@pytest.mark.asyncio
async def test_sleep_offload_parks_between_scores() -> None:
    """sleep_offload wakes the model for scoring and parks it back on CPU.

    First score: the factory-built (or injected) model is already on its GPU,
    so no wake move happens; after scoring it parks to cpu. Second score: wake
    restores the captured device, then parks again.
    """
    model = _MovableModel()
    runtime = LocalRewardRuntime({"sleep_offload": True}, model=model)

    await runtime.score_batch(_request())
    assert model.moves == ["cpu"]
    assert model.device == "cpu"

    await runtime.score_batch(_request())
    assert model.moves == ["cpu", "cuda:7", "cpu"]


@pytest.mark.asyncio
async def test_sleep_offload_requires_movable_model() -> None:
    """A sleep_offload model without .to() fails loud, not silently resident."""

    class _Immovable:
        def __call__(self, *, artifact, request):
            return {"overall": 1.0}

    runtime = LocalRewardRuntime({"sleep_offload": True}, model=_Immovable())
    with pytest.raises(TypeError, match="sleep_offload"):
        await runtime.score_batch(_request())
