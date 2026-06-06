"""Model-backed RewardFunction over the local transport."""

from __future__ import annotations

import pytest
import torch

from vrl.rewards.base import RewardFunction
from vrl.rewards.models.base import TorchRewardModel
from vrl.rewards.runtime import LocalRewardRuntime
from vrl.rewards.types import RewardRollout, RewardTrajectory


class _FakeTorchReward(TorchRewardModel):
    """Toy torch reward model: score = mean of the in-memory media tensor."""

    def _load(self) -> None:
        self.loaded_marker = True

    def score_media(self, *, media, prompt, request):
        return {"fake": float(media.float().mean().item())}


def _rollout(output: torch.Tensor, *, policy_version: int = 2) -> RewardRollout:
    return RewardRollout(
        request=None,
        trajectory=RewardTrajectory(prompt="p", output=output),
        metadata={"policy_version": policy_version},
    )


def _reward_function_local() -> RewardFunction:
    return RewardFunction(
        reward_name="fake",
        score_key="fake",
        runtime=LocalRewardRuntime(model=_FakeTorchReward({"device": "cpu"})),
        artifact_builder=lambda rollouts: RewardFunction.build_inmemory_artifacts(
            rollouts, media_type="image",
        ),
    )


@pytest.mark.asyncio
async def test_reward_function_local_scores_in_process_no_disk() -> None:
    """Checks reward function local scores in process no disk."""
    reward = _reward_function_local()
    out = await reward.score_batch(
        [
            _rollout(torch.full((1, 3, 2, 2), 0.5)),
            _rollout(torch.ones(1, 3, 2, 2)),
        ],
    )

    assert out == pytest.approx([0.5, 1.0])
    assert [r.worker_id for r in reward.last_results] == ["local", "local"]
    # in-memory artifacts carry no materialized path
    await reward.shutdown()


@pytest.mark.asyncio
async def test_reward_function_local_single_score() -> None:
    """Checks reward function local single score."""
    reward = _reward_function_local()
    assert await reward.score(_rollout(torch.zeros(1, 3, 2, 2))) == pytest.approx(0.0)


def test_pickscore_reward_model_constructs_lazily() -> None:
    """Checks pickscore reward model constructs lazily."""
    from vrl.rewards.models.pickscore import pickscore_reward_model

    model = pickscore_reward_model(
        {"device": "cpu", "model_name": "x", "processor_name": "y"},
    )
    assert model.model_name == "x"
    assert model.processor_name == "y"
    assert model._loaded is False  # no heavy load at construction
