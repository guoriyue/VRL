"""Tests for rollout collector runtime orchestration."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from vrl.generation import GenerationOutput, GenerationRequest, GenerationSampleRow
from vrl.rollouts.collector.batch_builder import (
    RolloutBatchBuildContext,
    TrajectoryRolloutBatchBuilder,
)
from vrl.rollouts.collector.core import RolloutCollector
from vrl.rollouts.collector.requests import CollectorRequest
from vrl.rollouts.collector.rewards import RewardScoringInput
from vrl.trajectory import RewardView, build_ar_discrete_trajectory


class _RequestBuilder:
    def build(
        self,
        prompts: list[str],
        group_size: int,
        kwargs: dict[str, Any],
    ) -> CollectorRequest:
        request = GenerationRequest(
            request_id="unit-request",
            family="unit",
            task="collect",
            prompts=prompts,
            samples_per_prompt=group_size,
            sampling={"seed": kwargs.get("seed")},
            return_artifacts={"output"},
            metadata={"source": "collector-test"},
            policy_version=kwargs.get("policy_version"),
        )
        return CollectorRequest(
            request=request,
            metadata={"collector": "metadata"},
        )


class _Runtime:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationOutput:
        self.requests.append(request)
        batch_size = len(request.prompts) * request.samples_per_prompt
        sample_rows = _sample_rows(request)
        output = torch.ones(batch_size, 3, 2, 2)
        trajectory = build_ar_discrete_trajectory(
            request=request,
            sample_rows=sample_rows,
            token_ids=torch.arange(batch_size * 2, dtype=torch.long).reshape(batch_size, 2),
            token_log_probs=torch.zeros(batch_size, 2),
            token_mask=torch.ones(batch_size, 2),
            prompt_input_ids=torch.ones(batch_size, 3, dtype=torch.long),
            prompt_attention_mask=torch.ones(batch_size, 3, dtype=torch.long),
            uncond_input_ids=torch.zeros(batch_size, 3, dtype=torch.long),
            uncond_attention_mask=torch.ones(batch_size, 3, dtype=torch.long),
            context={"collector": "test"},
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
        self.calls: list[dict[str, Any]] = []

    async def score(
        self,
        request: RewardScoringInput,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "outputs": request.outputs,
                "prompts": request.prompts,
                "metadata": request.metadata,
                "device": request.device,
            },
        )
        return torch.arange(request.batch_size, dtype=torch.float32)


def _collector(
    *,
    runtime: _Runtime | None = None,
    reward_scorer: _RewardScorer | None = None,
) -> RolloutCollector:
    return RolloutCollector(
        model=None,
        config=object(),
        family="unit",
        task="collect",
        request_builder=_RequestBuilder(),
        reward_scorer=reward_scorer or _RewardScorer(),
        runtime=runtime,
    )


def test_collector_requires_runtime_before_collect() -> None:
    import asyncio

    collector = _collector()

    with pytest.raises(RuntimeError, match="runtime is not initialized"):
        asyncio.run(collector.collect(["p0"], group_size=1))


def test_collector_routes_request_through_runtime_reward_and_trajectory_batch() -> None:
    import asyncio

    runtime = _Runtime()
    reward_scorer = _RewardScorer()
    collector = _collector(
        runtime=runtime,
        reward_scorer=reward_scorer,
    )

    batch = asyncio.run(
        collector.collect(["p0", "p1"], group_size=2, seed=5, policy_version=7),
    )

    assert len(runtime.requests) == 1
    request = runtime.requests[0]
    assert request.prompts == ["p0", "p1"]
    assert request.samples_per_prompt == 2
    assert request.sampling == {"seed": 5}
    assert request.policy_version == 7
    assert reward_scorer.calls[0]["metadata"] == {"collector": "metadata"}
    assert batch.rewards.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert batch.context == {"collector": "test"}
    assert batch.trajectory is not None
    assert batch.training_view is not None


def test_reward_scoring_input_rejects_prompt_output_mismatch() -> None:
    with pytest.raises(ValueError, match="prompt/output batch mismatch"):
        RewardScoringInput(
            outputs=torch.ones(2, 3),
            prompts=["p0"],
            metadata={},
            device="cpu",
        )


def test_reward_view_selection_fails_fast_when_ambiguous() -> None:
    import asyncio

    request = GenerationRequest(
        request_id="unit-request",
        family="unit",
        task="collect",
        prompts=["p0"],
        samples_per_prompt=1,
    )
    output = asyncio.run(_Runtime().generate(request))
    assert output.trajectory is not None
    output.trajectory.reward_views["alternate"] = RewardView(
        name="alternate",
        modality="image",
        metadata={"output_ref": "GenerationOutput.output"},
    )

    with pytest.raises(RuntimeError, match="multiple reward views"):
        TrajectoryRolloutBatchBuilder(
            output,
            RolloutBatchBuildContext(metadata={}),
        ).reward_outputs()

    selected = TrajectoryRolloutBatchBuilder(
        output,
        RolloutBatchBuildContext(metadata={}, reward_view_name="image"),
    ).reward_outputs()

    assert selected.shape[0] == len(output.sample_rows)


def test_collector_forwards_reference_metadata_to_request() -> None:
    from vrl.rollouts.collector.requests import RolloutEngineRequestBuilder
    from vrl.rollouts.collector.settings import RolloutSettings

    builder = RolloutEngineRequestBuilder(
        family="cosmos",
        task="v2w",
        request_prefix="cosmos",
        config=RolloutSettings(family="cosmos", values={"num_steps": 1}),
        return_artifacts=("trajectory",),
        default_task_type="video2world",
    )

    collector_request = builder.build(
        ["prompt"],
        1,
        {"reference_image": "/tmp/reference.png"},
    )

    assert collector_request.request.metadata["reference_image"] == "/tmp/reference.png"
    assert collector_request.metadata["reference_image"] == "/tmp/reference.png"


def _sample_rows(request: GenerationRequest) -> list[GenerationSampleRow]:
    rows: list[GenerationSampleRow] = []
    for prompt_index, prompt in enumerate(request.prompts):
        for sample_index in range(request.samples_per_prompt):
            sample_id = f"p{prompt_index}_s{sample_index}"
            rows.append(
                GenerationSampleRow(
                    prompt_index=prompt_index,
                    sample_index=sample_index,
                    prompt=prompt,
                    prompt_id=f"p{prompt_index}",
                    group_id=f"g{prompt_index}",
                    sample_id=sample_id,
                    trajectory_id=f"t_{sample_id}",
                    seed=None,
                ),
            )
    return rows
