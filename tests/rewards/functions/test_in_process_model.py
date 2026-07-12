"""Tests for model-backed RewardFunction over the in-process transport."""

from __future__ import annotations

import pytest
import torch

from vrl.rewards.base import (
    DiskArtifactRewardFunction,
    RewardCleanupError,
    RewardFunction,
)
from vrl.rewards.models.base import TorchRewardModel
from vrl.rewards.runtime import InProcessRewardRuntime
from vrl.rewards.types import RewardRollout


class _FakeTorchReward(TorchRewardModel):
    """Toy torch reward model: score = mean of the in-memory media tensor."""

    def _load(self) -> None:
        self.loaded_marker = True

    def score_media(self, *, media, prompt, request):
        return {"fake": float(media.float().mean().item())}


def _rollout(
    output: torch.Tensor,
    *,
    sample_id: str = "sample-0",
    policy_version: int = 2,
) -> RewardRollout:
    return RewardRollout(
        prompt="p",
        output=output,
        source_request_id="request-0",
        sample_id=sample_id,
        group_id="group-0",
        trajectory_id=f"trajectory-{sample_id}",
        policy_version=policy_version,
    )


def _reward_function_in_process() -> RewardFunction:
    return RewardFunction(
        reward_name="fake",
        score_key="fake",
        runtime=InProcessRewardRuntime(model=_FakeTorchReward({"device": "cpu"})),
        artifact_builder=lambda rollouts: RewardFunction.build_inmemory_artifacts(
            rollouts,
            media_type="image",
        ),
    )


@pytest.mark.asyncio
async def test_reward_function_in_process_scores_without_disk() -> None:
    """Checks reward function scoring in-process without disk artifacts."""
    reward = _reward_function_in_process()
    report = await reward.score_batch_report(
        [
            _rollout(torch.full((1, 3, 2, 2), 0.5)),
            _rollout(torch.ones(1, 3, 2, 2), sample_id="sample-1"),
        ],
    )

    assert report.scores == pytest.approx([0.5, 1.0])
    assert [r.worker_id for r in report.results] == ["local", "local"]
    # in-memory artifacts carry no materialized path
    await reward.shutdown()


@pytest.mark.asyncio
async def test_reward_function_in_process_single_score() -> None:
    """Checks a single in-process reward score."""
    reward = _reward_function_in_process()
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


def test_reward_artifact_transport_is_registry_visible() -> None:
    assert RewardFunction.artifact_transport == "in_memory"
    assert DiskArtifactRewardFunction.artifact_transport == "disk"


@pytest.mark.asyncio
async def test_reward_reports_operation_and_artifact_cleanup_failures() -> None:
    class _FailingRuntime:
        async def score_batch(self, request):
            raise RuntimeError("score failed")

        async def shutdown(self) -> None:
            return None

    def fail_cleanup(artifacts) -> None:
        assert artifacts
        raise OSError("cleanup failed")

    reward = RewardFunction(
        reward_name="fake",
        score_key="fake",
        runtime=_FailingRuntime(),
        artifact_builder=RewardFunction.build_inmemory_artifacts,
        artifact_finalizer=fail_cleanup,
    )

    with pytest.raises(RewardCleanupError) as error:
        await reward.score(_rollout(torch.zeros(1, 3, 2, 2)))

    assert [str(item) for item in error.value.errors] == [
        "score failed",
        "cleanup failed",
    ]


@pytest.mark.asyncio
async def test_reward_retains_artifacts_when_remote_state_is_ambiguous() -> None:
    class _AmbiguousRuntime:
        async def score_batch(self, request):
            error = RuntimeError("remote state unknown")
            error.retain_reward_artifacts = True
            raise error

        async def shutdown(self) -> None:
            return None

    finalized = False
    retained = False

    def finalize(artifacts) -> None:
        nonlocal finalized
        finalized = True

    def retain(artifacts) -> None:
        nonlocal retained
        assert artifacts
        retained = True

    reward = RewardFunction(
        reward_name="fake",
        score_key="fake",
        runtime=_AmbiguousRuntime(),
        artifact_builder=RewardFunction.build_inmemory_artifacts,
        artifact_finalizer=finalize,
        artifact_retainer=retain,
    )

    with pytest.raises(RuntimeError, match="remote state unknown"):
        await reward.score(_rollout(torch.zeros(1, 3, 2, 2)))

    assert retained is True
    assert finalized is False
