"""Tests for rollout collector runtime orchestration."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from vrl.generation import GenerationOutput, GenerationRequest, GenerationSampleRow
from vrl.ray.resources import ActorLeasePolicy, PhaseHandoffPolicy, RayLifecyclePlan
from vrl.rollouts.collector.batch_builder import (
    RolloutBatchBuildContext,
    TrajectoryRolloutBatchBuilder,
)
from vrl.rollouts.collector.core import RolloutCollector
from vrl.rollouts.collector.requests import CollectorRequest
from vrl.rollouts.collector.rewards import RewardScorer, RewardScoringInput
from vrl.rollouts.orchestration.prompt_collection import collect_prompt_batches
from vrl.trajectory import RewardView, build_ar_discrete_trajectory
from vrl.utils.stats import RolloutStats


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
        self.events: list[str] = []

    async def generate(self, request: GenerationRequest) -> GenerationOutput:
        self.requests.append(request)
        self.events.append("generate")
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

    async def release(self) -> None:
        self.events.append("release")


class _RewardScorer:
    def __init__(self, runtime: _Runtime | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.runtime = runtime

    async def score(
        self,
        request: RewardScoringInput,
    ) -> torch.Tensor:
        if self.runtime is not None:
            self.runtime.events.append("score")
        self.calls.append(
            {
                "outputs": request.outputs,
                "prompts": request.prompts,
                "metadata": request.metadata,
                "device": request.device,
            },
        )
        return torch.arange(request.batch_size, dtype=torch.float32)

    async def score_many(
        self,
        requests: list[RewardScoringInput],
    ) -> list[torch.Tensor]:
        return [await self.score(request) for request in requests]


def _collector(
    *,
    runtime: _Runtime | None = None,
    reward_scorer: _RewardScorer | None = None,
    lifecycle: RayLifecyclePlan | None = None,
) -> RolloutCollector:
    return RolloutCollector(
        model=None,
        config=object(),
        family="unit",
        task="collect",
        request_builder=_RequestBuilder(),
        reward_scorer=reward_scorer or _RewardScorer(),
        runtime=runtime,
        lifecycle=lifecycle,
    )


def test_collector_requires_runtime_before_collect() -> None:
    """Checks collector requires runtime before collect."""
    import asyncio

    collector = _collector()

    with pytest.raises(RuntimeError, match="runtime is not initialized"):
        asyncio.run(collector.collect(["p0"], group_size=1))


def test_collector_routes_request_through_runtime_reward_and_trajectory_batch() -> None:
    """Checks collector routes request through runtime reward and trajectory batch."""
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
    assert runtime.events == ["generate"]
    assert batch.rewards.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert batch.context == {"collector": "test"}
    assert batch.trajectory is not None
    assert batch.training_view is not None


def test_collector_releases_runtime_memory_before_reward_scoring() -> None:
    """Checks collector releases runtime memory before reward scoring."""
    import asyncio

    runtime = _Runtime()
    reward_scorer = _RewardScorer(runtime)
    # Shared reward GPU: the lifecycle plan (not the runtime) tells the collector
    # to drop rollout actors before the reward model scores.
    lifecycle = RayLifecyclePlan(
        rollout=ActorLeasePolicy(mode="on_demand"),
        reward=ActorLeasePolicy(mode="on_demand"),
        handoff=PhaseHandoffPolicy(
            release_rollout_before_train=True,
            release_rollout_before_reward=True,
            release_reward_after_score=True,
        ),
    )
    collector = _collector(
        runtime=runtime,
        reward_scorer=reward_scorer,
        lifecycle=lifecycle,
    )

    asyncio.run(collector.collect(["p0"], group_size=1))

    assert runtime.events == ["generate", "release", "score"]


def test_collector_does_not_release_runtime_memory_before_independent_reward() -> None:
    """Checks collector does not release runtime memory before independent reward."""
    import asyncio

    runtime = _Runtime()
    reward_scorer = _RewardScorer(runtime)
    # Dedicated reward GPU: the plan keeps both roles resident, so the collector
    # never releases before reward.
    lifecycle = RayLifecyclePlan(
        rollout=ActorLeasePolicy(mode="resident"),
        reward=ActorLeasePolicy(mode="resident"),
        handoff=PhaseHandoffPolicy(
            release_rollout_before_train=False,
            release_rollout_before_reward=False,
            release_reward_after_score=False,
        ),
    )
    collector = _collector(
        runtime=runtime,
        reward_scorer=reward_scorer,
        lifecycle=lifecycle,
    )

    asyncio.run(collector.collect(["p0"], group_size=1))

    assert runtime.events == ["generate", "score"]


def test_reward_scoring_input_rejects_prompt_output_mismatch() -> None:
    """Checks reward scoring input rejects prompt output mismatch."""
    with pytest.raises(ValueError, match="prompt/output batch mismatch"):
        RewardScoringInput(
            outputs=torch.ones(2, 3),
            prompts=["p0"],
            metadata={},
            device="cpu",
        )


def test_reward_scorer_score_many_uses_one_call_and_splits_per_group() -> None:
    """Checks score_many merges groups into one score_batch with per-group metadata."""
    import asyncio

    class _CountingReward:
        def __init__(self) -> None:
            self.score_batch_calls: list[list[Any]] = []

        async def score_batch(self, rollouts: list[Any]) -> list[float]:
            self.score_batch_calls.append(list(rollouts))
            return [float(i) for i in range(len(rollouts))]

    reward_fn = _CountingReward()
    scorer = RewardScorer(reward_fn)
    requests = [
        RewardScoringInput(
            outputs=torch.ones(2, 3),
            prompts=["g0-a", "g0-b"],
            metadata={"target_text": "group-0"},
            device="cpu",
        ),
        RewardScoringInput(
            outputs=torch.ones(3, 3),
            prompts=["g1-a", "g1-b", "g1-c"],
            metadata={"target_text": "group-1"},
            device="cpu",
        ),
    ]

    rewards = asyncio.run(scorer.score_many(requests))

    # One reward call for both groups — this is what keeps one actor
    # lifecycle per epoch for release_after_score rewards.
    assert len(reward_fn.score_batch_calls) == 1
    rollouts = reward_fn.score_batch_calls[0]
    assert [r.trajectory.prompt for r in rollouts] == [
        "g0-a", "g0-b", "g1-a", "g1-b", "g1-c",
    ]
    assert [r.metadata["target_text"] for r in rollouts] == [
        "group-0", "group-0", "group-1", "group-1", "group-1",
    ]
    # Scores split back by group size, in order.
    assert rewards[0].tolist() == [0.0, 1.0]
    assert rewards[1].tolist() == [2.0, 3.0, 4.0]

    assert asyncio.run(scorer.score_many([])) == []
    assert scorer.last_reward_timing_ms == {}


def test_collect_prompt_batches_folds_reward_timing_into_stats() -> None:
    """Checks reward runtime timing reaches RolloutStats."""
    import asyncio

    class _TimedReward:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []
            self.last_timing_ms: dict[str, float] = {}

        async def score_batch(self, rollouts: list[Any]) -> list[float]:
            self.batch_sizes.append(len(rollouts))
            self.last_timing_ms = {
                "latency_ms": 12.0,
                "queue_wait_ms": 3.0,
                "inference_ms": 9.0,
            }
            return [float(index + 1) for index in range(len(rollouts))]

    reward_fn = _TimedReward()
    collector = RolloutCollector(
        model=None,
        config=object(),
        family="unit",
        task="collect",
        request_builder=_RequestBuilder(),
        reward_scorer=RewardScorer(reward_fn),
        runtime=_Runtime(),
    )
    stats = RolloutStats()

    batches = asyncio.run(
        collect_prompt_batches(
            collector=collector,
            prompts=["p0"],
            group_size=2,
            runtime_debug=False,
            policy_version=None,
            stats=stats,
        ),
    )

    assert reward_fn.batch_sizes == [2]
    assert len(batches) == 1
    assert batches[0].rewards.tolist() == [1.0, 2.0]
    assert stats.reward_latency_ms == 12.0
    assert stats.reward_queue_wait_ms == 3.0
    assert stats.reward_inference_ms == 9.0
    assert stats.as_phase_dict()["reward.queue_wait_s"] == 0.003


def test_reward_view_selection_fails_fast_when_ambiguous() -> None:
    """Checks reward view selection fails fast when ambiguous."""
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
    """Checks collector forwards reference metadata to request."""
    from vrl.rollouts.collector.config import RolloutConfig
    from vrl.rollouts.collector.requests import GenerationRequestBuilder

    builder = GenerationRequestBuilder(
        family="cosmos",
        task="v2w",
        request_prefix="cosmos",
        config=RolloutConfig(family="cosmos", values={"num_steps": 1}),
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


def test_reward_outputs_reconstructs_uint8_wire_video_exactly() -> None:
    """Checks uint8 wire-packed video reconstructs to k/255 floats.

    The worker packs decoded video as uint8 before the wire (wire diet T1);
    reward_outputs must hand consumers [0, 1] floats that round-trip
    bit-exactly through every downstream to_uint8 quantization, keeping
    reward scores identical to the fp32-over-wire path.
    """
    import asyncio

    from vrl.utils.media import to_uint8

    request = GenerationRequest(
        request_id="unit-request",
        family="unit",
        task="collect",
        prompts=["p0"],
        samples_per_prompt=1,
    )
    output = asyncio.run(_Runtime().generate(request))
    packed = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 16, 16)
    output.output = packed

    reconstructed = TrajectoryRolloutBatchBuilder(
        output,
        RolloutBatchBuildContext(metadata={}, reward_view_name="image"),
    ).reward_outputs()

    assert reconstructed.dtype == torch.float32
    assert float(reconstructed.min()) >= 0.0
    assert float(reconstructed.max()) <= 1.0
    # The G3 guarantee: re-quantizing recovers every byte value exactly.
    assert torch.equal(to_uint8(reconstructed), packed)


def test_uint8_quantization_roundtrip_is_exact_for_all_byte_values() -> None:
    """Checks k/255 floats survive both downstream quantization formulas.

    Pins the mechanism behind reward-score equality: to_uint8 (reward
    models) and the *255-round mp4 path must both map k/255 back to k.
    """
    k = torch.arange(256, dtype=torch.float32)
    grid = k / 255.0

    from vrl.utils.media import to_uint8

    assert torch.equal(to_uint8(grid), k.to(torch.uint8))
    mp4_path = (grid * 255.0).round().clamp(0, 255).to(torch.uint8)
    assert torch.equal(mp4_path, k.to(torch.uint8))


def test_collect_phase_timings_are_per_call_not_shared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase timings live on each call's rollouts; no shared collector state."""
    import asyncio

    monkeypatch.setenv("VRL_PROFILE_COLLECT", "1")
    collector = _collector(runtime=_Runtime(), reward_scorer=_RewardScorer())

    async def _run() -> tuple[Any, Any]:
        first = await collector.collect_unscored(["p0"], group_size=1)
        second = await collector.collect_unscored(["p1"], group_size=1)
        await collector.score_rollouts([first, second])
        return first, second

    first, second = asyncio.run(_run())

    # Generation time is owned per collect_unscored call.
    assert "collect.engine_generate" in first.phases
    assert "collect.engine_generate" in second.phases
    # Call-level score/build timings land on the first group only, so summing
    # phases over groups never multiplies the same wall time.
    assert "collect.reward_score" in first.phases
    assert "collect.batch_build" in first.phases
    assert "collect.reward_score" not in second.phases
    assert "collect.batch_build" not in second.phases
    # The old shared mutable dict (clobbered by concurrent collects) is gone.
    assert not hasattr(collector, "last_collect_phases")
