"""Tests for reward artifact lifetime policy."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from vrl.generation import GenerationOutput, GenerationRequest, GenerationSampleRow
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.collector.artifacts import (
    RewardArtifactPolicy,
    release_reward_artifact_if_needed,
    reward_artifact_bytes,
    reward_artifact_policy_from_cfg,
)
from vrl.rollouts.collector.config import RolloutConfig, build_rollout_config_from_cfg
from vrl.rollouts.collector.core import RolloutCollector
from vrl.rollouts.collector.requests import CollectorRequest
from vrl.rollouts.collector.rewards import RewardScoringInput
from vrl.trajectory import build_ar_continuous_trajectory, build_ar_discrete_trajectory


class _RequestBuilder:
    def build(
        self,
        prompts: list[str],
        group_size: int,
        kwargs: dict[str, Any],
    ) -> CollectorRequest:
        request = GenerationRequest(
            request_id="artifact-request",
            family="unit",
            task="ar_t2i",
            prompts=prompts,
            samples_per_prompt=group_size,
            return_artifacts={"output"},
        )
        return CollectorRequest(request=request, metadata={})


class _Runtime:
    def should_release_memory_before_reward(self) -> bool:
        return False

    async def generate(self, request: GenerationRequest) -> GenerationOutput:
        batch_size = len(request.prompts) * request.samples_per_prompt
        sample_rows = _sample_rows(request)
        output = torch.ones(batch_size, 3, 4, 4)
        trajectory = build_ar_discrete_trajectory(
            request=request,
            sample_rows=sample_rows,
            token_ids=torch.ones(batch_size, 2, dtype=torch.long),
            token_log_probs=torch.zeros(batch_size, 2),
            token_mask=torch.ones(batch_size, 2),
            prompt_input_ids=torch.ones(batch_size, 3, dtype=torch.long),
            prompt_attention_mask=torch.ones(batch_size, 3, dtype=torch.long),
            uncond_input_ids=torch.zeros(batch_size, 3, dtype=torch.long),
            uncond_attention_mask=torch.ones(batch_size, 3, dtype=torch.long),
            context={"collector": "artifact-test"},
        )
        return GenerationOutput(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_rows=sample_rows,
            output=output,
            trajectory=trajectory,
        )


class _RewardScorer:
    def __init__(self) -> None:
        self.outputs_shape: tuple[int, ...] | None = None

    async def score(self, request: RewardScoringInput) -> torch.Tensor:
        self.outputs_shape = tuple(request.outputs.shape)
        return torch.zeros(request.batch_size, device=request.device)

    async def score_many(
        self,
        requests: list[RewardScoringInput],
    ) -> list[torch.Tensor]:
        return [await self.score(request) for request in requests]


@pytest.mark.asyncio
async def test_collector_releases_generation_output_after_reward_scoring() -> None:
    """Checks collector releases generation output after reward scoring."""
    scorer = _RewardScorer()
    collector = RolloutCollector(
        model=None,
        config=RolloutConfig(
            family="unit",
            values={"reward_artifact": {"keep_after_reward": False}},
        ),
        family="unit",
        task="ar_t2i",
        request_builder=_RequestBuilder(),
        reward_scorer=scorer,
        runtime=_Runtime(),
    )

    batch = await collector.collect(["p"], group_size=2)

    assert scorer.outputs_shape == (2, 3, 4, 4)
    assert batch.videos is None
    assert batch.trajectory is not None
    assert batch.trajectory.segments["image_tokens"].tensors["token_ids"].value is not None
    assert batch.context == {"collector": "artifact-test"}


def test_release_policy_drops_non_trainable_reward_view_tensors_only() -> None:
    """Checks release policy drops non trainable reward view tensors only."""
    trajectory = _continuous_trajectory()
    decoded = trajectory.segments["decoded"].tensors["images_for_reward"].value
    batch = RolloutBatch(
        observations=torch.zeros(2, 1, 2),
        actions=trajectory.segments["image_tokens"].tensors["tokens"].value,
        rewards=torch.zeros(2),
        dones=torch.ones(2, dtype=torch.bool),
        group_ids=torch.tensor([0, 0]),
        videos=decoded.unsqueeze(2),
        trajectory=trajectory,
    )

    release_reward_artifact_if_needed(batch, RewardArtifactPolicy(keep_after_reward=False))

    assert batch.videos is None
    assert trajectory.segments["decoded"].tensors["images_for_reward"].value is None
    assert trajectory.segments["decoded"].tensors["images_for_reward"].metadata == {
        "released_after_reward": True,
    }
    assert trajectory.segments["image_tokens"].tensors["tokens"].value is not None


def test_release_policy_drops_plain_videos_without_mutating_context() -> None:
    # Covers the no-trajectory path: a batch whose only reward artifact is the
    # decoded ``videos`` tensor. Release must free the tensor while leaving the
    # trainer ``context`` untouched.
    """Checks release policy drops plain videos without mutating context."""
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

    assert reward_artifact_bytes(batch.videos) == artifact.numel() * artifact.element_size()

    release_reward_artifact_if_needed(batch, RewardArtifactPolicy(keep_after_reward=False))

    assert batch.videos is None
    assert batch.context == {"reward_metadata": {"source": "unit"}}


def test_reward_artifact_policy_default_keeps_artifacts() -> None:
    """Checks reward artifact policy default keeps artifacts."""
    artifact = torch.ones(2, 3, 4, 4)
    batch = RolloutBatch(
        observations=torch.zeros(2, 1),
        actions=torch.zeros(2, 1),
        rewards=torch.zeros(2),
        dones=torch.ones(2, dtype=torch.bool),
        group_ids=torch.tensor([0, 0]),
        videos=artifact,
    )

    release_reward_artifact_if_needed(batch, reward_artifact_policy_from_cfg(None))

    assert batch.videos is artifact
    assert reward_artifact_bytes(artifact) == artifact.numel() * artifact.element_size()


def test_rollout_config_preserves_nested_storage_and_artifact_policy() -> None:
    """Checks rollout config preserves nested storage and artifact policy."""
    config = build_rollout_config_from_cfg(
        {
            "rollout": {
                "trajectory_storage": {"device": "cpu", "dtype": "float16"},
                "reward_artifact": {"keep_after_reward": False},
            },
        },
        family="unit",
    )

    assert config.get("trajectory_storage") == {"device": "cpu", "dtype": "float16"}
    assert config.get("reward_artifact") == {"keep_after_reward": False}


def _continuous_trajectory():
    request = GenerationRequest(
        request_id="continuous-request",
        family="nextstep_1",
        task="ar_t2i_continuous",
        prompts=["p"],
        samples_per_prompt=2,
    )
    sample_rows = _sample_rows(request)
    return build_ar_continuous_trajectory(
        request=request,
        sample_rows=sample_rows,
        tokens=torch.ones(2, 2, 3),
        saved_noise=torch.zeros(2, 2, 3),
        token_log_probs=torch.zeros(2, 2),
        token_mask=torch.ones(2, 2),
        prompt_input_ids=torch.ones(2, 3, dtype=torch.long),
        prompt_attention_mask=torch.ones(2, 3, dtype=torch.long),
        uncond_input_ids=torch.zeros(2, 3, dtype=torch.long),
        uncond_attention_mask=torch.ones(2, 3, dtype=torch.long),
        images_for_reward=torch.ones(2, 3, 4, 4),
        context={},
    )


def _sample_rows(request: GenerationRequest) -> list[GenerationSampleRow]:
    rows: list[GenerationSampleRow] = []
    for prompt_index, prompt in enumerate(request.prompts):
        for sample_index in range(request.samples_per_prompt):
            rows.append(
                GenerationSampleRow(
                    prompt_index=prompt_index,
                    sample_index=sample_index,
                    prompt=prompt,
                    prompt_id=f"p{prompt_index}",
                    group_id=f"g{prompt_index}",
                    sample_id=f"s{sample_index}",
                    trajectory_id=f"t{sample_index}",
                    seed=None,
                ),
            )
    return rows
