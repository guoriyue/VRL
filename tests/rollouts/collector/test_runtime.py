"""Tests for rollout collector runtime orchestration."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import pytest
import torch

from tests.rollouts.collector._helpers import collect_scored
from vrl.generation import (
    GenerationInput,
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
)
from vrl.ray.resources import ActorLeasePolicy, PhaseHandoffPolicy, RayLifecyclePlan
from vrl.rewards.base import RewardBatchReport, RewardCleanupError
from vrl.rewards.inference import RewardMemoryReleaseProof
from vrl.rollouts.collector.batch_builder import (
    RolloutBatchBuildContext,
    TrajectoryRolloutBatchBuilder,
)
from vrl.rollouts.collector.core import RolloutCollector
from vrl.rollouts.collector.requests import CollectorRequest
from vrl.rollouts.collector.rewards import (
    RewardScoreBatch,
    RewardScorer,
    RewardScoringInput,
)
from vrl.rollouts.orchestration.prompt_collection import collect_prompt_batches
from vrl.trajectory import (
    RewardView,
    build_ar_discrete_trajectory,
    build_chunk_autoregressive_denoise_trajectory,
)
from vrl.utils.stats import RolloutStats


class _RequestBuilder:
    def build(
        self,
        inputs: list[Any],
        group_size: int,
        *,
        metadata: dict[str, Any] | None = None,
        request_overrides: dict[str, Any] | None = None,
        seed: int | None = None,
        runtime_debug: bool = False,
        policy_version: int | None = None,
    ) -> CollectorRequest:
        request = GenerationRequest(
            request_id="unit-request",
            family="unit",
            task="collect",
            inputs=list(inputs),
            samples_per_prompt=group_size,
            sampling={"seed": seed},
            metadata={"source": "collector-test"},
            policy_version=policy_version,
        )
        return CollectorRequest(
            request=request,
            metadata={"collector": "metadata"},
        )


class _Runtime:
    def __init__(
        self,
        *,
        fail_offload: bool = False,
        shutdown_failures: int = 0,
    ) -> None:
        self.requests: list[GenerationRequest] = []
        self.events: list[str] = []
        self.fail_offload = fail_offload
        self.current_policy_version = 0
        self.requires_driver_model_offload = False
        self.shutdown_failures = shutdown_failures
        self.shutdown_calls = 0

    async def activate(self) -> None:
        return None

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
            sample_rows=sample_rows,
            output=output,
            trajectory=trajectory,
        )

    async def offload(self) -> None:
        self.events.append("offload")
        if self.fail_offload:
            raise RuntimeError("rollout offload failed")

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.shutdown_calls <= self.shutdown_failures:
            raise RuntimeError("runtime shutdown failed")

    def is_colocated(self) -> bool:
        return False


class _RewardScorer:
    def __init__(
        self,
        runtime: _Runtime | None = None,
        *,
        fail_park: bool = False,
        scoring_is_nonblocking: bool = False,
        external_accelerator_isolation_verified: bool = False,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.runtime = runtime
        self.fail_park = fail_park
        self.reward_fn = self
        self.scoring_is_nonblocking = scoring_is_nonblocking
        self.external_accelerator_isolation_verified = external_accelerator_isolation_verified
        self.shutdown_failures = 0
        self.shutdown_calls = 0
        self.last_memory_release_proofs: tuple[RewardMemoryReleaseProof, ...] = ()

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.shutdown_calls <= self.shutdown_failures:
            raise RuntimeError("reward shutdown failed")

    async def score(
        self,
        request: RewardScoringInput,
    ) -> torch.Tensor:
        if self.runtime is not None:
            self.runtime.events.append("score")
        self.calls.append(
            {
                "outputs": request.outputs,
                "prompts": [row.prompt for row in request.sample_rows],
                "metadata": request.metadata,
                "device": request.device,
            },
        )
        return torch.arange(request.batch_size, dtype=torch.float32)

    async def score_many(
        self,
        requests: list[RewardScoringInput],
        *,
        require_memory_release: bool = False,
    ) -> RewardScoreBatch:
        scores = [await self.score(request) for request in requests]
        self.last_memory_release_proofs = ()
        if require_memory_release:
            await self.park_memory(required=True)
        return RewardScoreBatch(scores=scores, components={}, timing_ms={})

    async def park_memory(
        self,
        *,
        required: bool,
    ) -> tuple[RewardMemoryReleaseProof, ...]:
        assert required is True
        if self.runtime is not None:
            self.runtime.events.append("reward_park")
        if self.fail_park:
            raise RuntimeError("reward park failed")
        self.last_memory_release_proofs = (
            RewardMemoryReleaseProof(request_id="collector-score", released=True),
        )
        return self.last_memory_release_proofs


def _collector(
    *,
    runtime: _Runtime | None = None,
    reward_scorer: _RewardScorer | None = None,
    lifecycle: RayLifecyclePlan | None = None,
) -> RolloutCollector:
    return RolloutCollector(
        config=object(),
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
        asyncio.run(collect_scored(collector, ["p0"], group_size=1))


def test_collector_rejects_incomplete_runtime_control_protocol() -> None:
    class _GenerateOnlyRuntime:
        async def generate(self, request: GenerationRequest) -> GenerationOutput:
            raise NotImplementedError

    with pytest.raises(TypeError, match="complete GenerationRuntime protocol"):
        _collector(runtime=_GenerateOnlyRuntime())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_collector_shutdown_retries_in_safe_runtime_then_reward_order() -> None:
    runtime = _Runtime(shutdown_failures=1)
    reward_scorer = _RewardScorer()
    reward_scorer.shutdown_failures = 1
    collector = _collector(runtime=runtime, reward_scorer=reward_scorer)

    with pytest.raises(RuntimeError, match="runtime shutdown failed"):
        await collector.shutdown()
    assert runtime.shutdown_calls == 1
    assert reward_scorer.shutdown_calls == 0
    assert collector._runtime is runtime
    assert collector._reward_shutdown_complete is False

    with pytest.raises(RuntimeError, match="reward shutdown failed"):
        await collector.shutdown()
    assert runtime.shutdown_calls == 2
    assert reward_scorer.shutdown_calls == 1
    assert collector._runtime is None
    assert collector._reward_shutdown_complete is False

    await collector.shutdown()
    await collector.shutdown()

    assert runtime.shutdown_calls == 2
    assert reward_scorer.shutdown_calls == 2
    assert collector._runtime is None
    assert collector._reward_shutdown_complete is True


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
        collect_scored(collector, ["p0", "p1"], group_size=2, seed=5, policy_version=7),
    )

    assert len(runtime.requests) == 1
    request = runtime.requests[0]
    assert request.prompts == ["p0", "p1"]
    assert request.samples_per_prompt == 2
    assert request.sampling == {"seed": 5}
    assert request.policy_version == 7
    assert reward_scorer.calls[0]["metadata"] == {"collector": "metadata"}
    assert reward_scorer.calls[0]["prompts"] == ["p0", "p0", "p1", "p1"]
    assert reward_scorer.calls[0]["outputs"].shape == (4, 3, 2, 2)
    assert runtime.events == ["generate"]
    assert batch.rewards.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert batch.context == {"collector": "test"}
    assert batch.trajectory is not None
    assert batch.training_view is not None
    assert not hasattr(batch, "dones")
    assert not hasattr(batch, "videos")
    assert not hasattr(batch, "prompts")


@pytest.mark.asyncio
async def test_profiled_collector_builds_cpu_batch_without_trainer_cuda_sync(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VRL_PROFILE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def reject_trainer_sync(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("collector touched the trainer CUDA device")

    monkeypatch.setattr(torch.cuda, "synchronize", reject_trainer_sync)
    collector = _collector(runtime=_Runtime(), reward_scorer=_RewardScorer())

    batch = await collect_scored(collector, ["p0"], group_size=1)

    assert batch.rewards.device.type == "cpu"
    assert batch.group_ids.device.type == "cpu"


def test_collector_offloads_runtime_memory_before_reward_scoring() -> None:
    """Checks collector offloads runtime memory before reward scoring."""
    import asyncio

    runtime = _Runtime()
    reward_scorer = _RewardScorer(runtime)
    # Shared reward GPU: the lifecycle plan (not the runtime) tells the collector
    # to park rollout GPU memory before the in-process reward model scores.
    lifecycle = RayLifecyclePlan(
        rollout=ActorLeasePolicy(mode="on_demand"),
        reward=ActorLeasePolicy(mode="on_demand"),
        handoff=PhaseHandoffPolicy(
            release_rollout_before_train=True,
            release_rollout_before_reward=True,
            release_trainer_before_reward=True,
            release_reward_after_score=True,
        ),
    )
    collector = _collector(
        runtime=runtime,
        reward_scorer=reward_scorer,
        lifecycle=lifecycle,
    )

    asyncio.run(collect_scored(collector, ["p0"], group_size=1))
    asyncio.run(collector.offload_runtime_memory())

    assert runtime.events == [
        "generate",
        "offload",
        "score",
        "reward_park",
        "offload",
        "reward_park",
    ]
    assert collector.requires_driver_model_offload_for_reward is True


def test_collector_does_not_offload_runtime_before_independent_reward() -> None:
    """Checks collector keeps rollout active for an independent reward."""
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
            release_trainer_before_reward=False,
            release_reward_after_score=False,
        ),
    )
    collector = _collector(
        runtime=runtime,
        reward_scorer=reward_scorer,
        lifecycle=lifecycle,
    )

    asyncio.run(collect_scored(collector, ["p0"], group_size=1))

    assert runtime.events == ["generate", "score"]
    assert collector.requires_driver_model_offload_for_reward is False


@pytest.mark.parametrize(
    (
        "rollout_handoff",
        "trainer_handoff",
        "scorer_supports_overlap",
        "expected",
    ),
    [
        (False, False, True, True),
        (False, False, False, False),
        (True, False, True, False),
        (False, True, True, False),
    ],
)
def test_collector_derives_reward_generation_overlap_from_topology_and_scorer(
    rollout_handoff: bool,
    trainer_handoff: bool,
    scorer_supports_overlap: bool,
    expected: bool,
) -> None:
    lifecycle = RayLifecyclePlan(
        rollout=ActorLeasePolicy(mode="resident"),
        reward=ActorLeasePolicy(mode="resident"),
        handoff=PhaseHandoffPolicy(
            release_rollout_before_train=False,
            release_rollout_before_reward=rollout_handoff,
            release_trainer_before_reward=trainer_handoff,
            release_reward_after_score=False,
        ),
    )
    collector = _collector(
        reward_scorer=_RewardScorer(
            scoring_is_nonblocking=scorer_supports_overlap,
            external_accelerator_isolation_verified=scorer_supports_overlap,
        ),
        lifecycle=lifecycle,
    )

    assert collector.supports_reward_generation_overlap is expected
    assert collector.supports_continuous_reward_execution is expected


def test_collector_keeps_continuous_admission_for_no_reward() -> None:
    collector = _collector(reward_scorer=RewardScorer(None))

    assert collector.supports_reward_generation_overlap is False
    assert collector.supports_continuous_reward_execution is True


def test_collector_separates_nonblocking_scoring_from_accelerator_isolation() -> None:
    scorer = _RewardScorer()
    scorer.scoring_is_nonblocking = True
    scorer.external_accelerator_isolation_verified = False
    collector = _collector(reward_scorer=scorer)

    assert collector.supports_reward_generation_overlap is False
    assert collector.supports_continuous_reward_execution is False

    scorer.external_accelerator_isolation_verified = True

    assert collector.supports_reward_generation_overlap is True
    assert collector.supports_continuous_reward_execution is True


def test_dedicated_local_reward_allows_concurrent_collects_without_streaming() -> None:
    scorer = _RewardScorer(external_accelerator_isolation_verified=True)
    collector = _collector(reward_scorer=scorer)

    assert collector.supports_reward_generation_overlap is False
    assert collector.supports_continuous_reward_execution is True


def test_collector_blocks_trainer_handoff_when_reward_release_proof_fails() -> None:
    """A failed reward park remains terminal even after rollout itself parks."""
    import asyncio

    class _FailingRewardScorer(_RewardScorer):
        async def score_many(
            self,
            requests: list[RewardScoringInput],
            *,
            require_memory_release: bool = False,
        ) -> RewardScoreBatch:
            await super().score_many(
                requests,
                require_memory_release=False,
            )
            assert require_memory_release is True
            raise RuntimeError("reward park failed")

    runtime = _Runtime()
    lifecycle = RayLifecyclePlan(
        rollout=ActorLeasePolicy(mode="on_demand"),
        reward=ActorLeasePolicy(mode="on_demand"),
        handoff=PhaseHandoffPolicy(
            release_rollout_before_train=True,
            release_rollout_before_reward=True,
            release_trainer_before_reward=True,
            release_reward_after_score=True,
        ),
    )
    collector = _collector(
        runtime=runtime,
        reward_scorer=_FailingRewardScorer(runtime, fail_park=True),
        lifecycle=lifecycle,
    )

    with pytest.raises(RuntimeError, match="reward park failed"):
        asyncio.run(collect_scored(collector, ["p0"], group_size=1))
    with pytest.raises(RuntimeError, match="reward park failed"):
        asyncio.run(collector.offload_runtime_memory())

    assert runtime.events == [
        "generate",
        "offload",
        "score",
        "offload",
        "reward_park",
    ]


def test_collector_phase_final_gate_retries_reward_parking() -> None:
    """The final rollout offload retries a transient reward park failure."""
    import asyncio

    class _FlakyRewardScorer(_RewardScorer):
        def __init__(self, runtime: _Runtime) -> None:
            super().__init__(runtime)
            self.park_attempts = 0

        async def park_memory(
            self,
            *,
            required: bool,
        ) -> tuple[RewardMemoryReleaseProof, ...]:
            self.park_attempts += 1
            if self.runtime is not None:
                self.runtime.events.append("reward_park")
            if self.park_attempts == 1:
                raise RuntimeError("transient reward park failure")
            return await super().park_memory(required=required)

    runtime = _Runtime()
    scorer = _FlakyRewardScorer(runtime)
    lifecycle = RayLifecyclePlan(
        rollout=ActorLeasePolicy(mode="on_demand"),
        reward=ActorLeasePolicy(mode="on_demand"),
        handoff=PhaseHandoffPolicy(
            release_rollout_before_train=True,
            release_rollout_before_reward=True,
            release_trainer_before_reward=True,
            release_reward_after_score=True,
        ),
    )
    collector = _collector(runtime=runtime, reward_scorer=scorer, lifecycle=lifecycle)

    with pytest.raises(RuntimeError, match="transient reward park failure"):
        asyncio.run(collect_scored(collector, ["p0"], group_size=1))

    asyncio.run(collector.offload_runtime_memory())

    assert scorer.park_attempts == 2
    assert scorer.last_memory_release_proofs[0].released is True


@pytest.mark.parametrize("reward_park_fails", [False, True])
def test_collector_attempts_reward_park_after_rollout_offload_failure(
    reward_park_fails: bool,
) -> None:
    """Rollout failure cannot skip reward cleanup; dual failures aggregate."""
    import asyncio

    runtime = _Runtime(fail_offload=True)
    scorer = _RewardScorer(runtime, fail_park=reward_park_fails)
    lifecycle = RayLifecyclePlan(
        rollout=ActorLeasePolicy(mode="on_demand"),
        reward=ActorLeasePolicy(mode="on_demand"),
        handoff=PhaseHandoffPolicy(
            release_rollout_before_train=True,
            release_rollout_before_reward=True,
            release_trainer_before_reward=True,
            release_reward_after_score=True,
        ),
    )
    collector = _collector(runtime=runtime, reward_scorer=scorer, lifecycle=lifecycle)
    collector._reward_phase_started = True

    if reward_park_fails:
        with pytest.raises(RewardCleanupError) as error:
            asyncio.run(collector.offload_runtime_memory())
        assert len(error.value.errors) == 2
    else:
        with pytest.raises(RuntimeError, match="rollout offload failed"):
            asyncio.run(collector.offload_runtime_memory())

    assert runtime.events == ["offload", "reward_park"]


def _reward_sample_rows(
    request_id: str,
    prompts: list[str],
    *,
    policy_version: int = 4,
) -> list[GenerationSampleRow]:
    return [
        GenerationSampleRow(
            prompt_index=index,
            sample_index=0,
            prompt=prompt,
            group_id=f"{request_id}:group:{index}",
            sample_id=f"{request_id}:sample:{index}",
            trajectory_id=f"{request_id}:trajectory:{index}",
            seed=None,
            metadata={"policy_version": policy_version},
        )
        for index, prompt in enumerate(prompts)
    ]


def test_reward_scoring_input_rejects_sample_row_output_mismatch() -> None:
    with pytest.raises(ValueError, match="sample-row/output batch mismatch"):
        RewardScoringInput(
            outputs=torch.ones(2, 3),
            source_request_id="request-0",
            sample_rows=_reward_sample_rows("request-0", ["p0"]),
            metadata={},
            device="cpu",
        )


def test_reward_scoring_input_derives_batch_size_and_prompt_rows() -> None:
    request = RewardScoringInput(
        outputs=torch.ones(2, 3),
        source_request_id="request-0",
        sample_rows=_reward_sample_rows("request-0", ["p0", "p1"]),
        metadata={},
        device="cpu",
    )

    assert request.batch_size == 2
    assert [row.prompt for row in request.sample_rows] == ["p0", "p1"]
    assert {field.name for field in fields(request)}.isdisjoint(
        {"prompts", "expected_count", "batch_size"},
    )


def test_reward_scoring_input_rejects_mismatched_row_request_id() -> None:
    rows = _reward_sample_rows("request-0", ["p0"])
    rows[0].metadata["request_id"] = "different-request"

    with pytest.raises(ValueError, match="source request/sample-row mismatch"):
        RewardScoringInput(
            outputs=torch.ones(1, 3),
            source_request_id="request-0",
            sample_rows=rows,
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
            source_request_id="request-0",
            sample_rows=_reward_sample_rows("request-0", ["g0-a", "g0-b"]),
            metadata={"target_text": "group-0"},
            device="cpu",
        ),
        RewardScoringInput(
            outputs=torch.ones(3, 3),
            source_request_id="request-1",
            sample_rows=_reward_sample_rows(
                "request-1",
                ["g1-a", "g1-b", "g1-c"],
                policy_version=5,
            ),
            metadata={"target_text": "group-1"},
            device="cpu",
        ),
    ]

    rewards = asyncio.run(scorer.score_many(requests)).scores

    # One reward call for both groups — this is what keeps one actor
    # lifecycle per epoch for release_after_score rewards.
    assert len(reward_fn.score_batch_calls) == 1
    rollouts = reward_fn.score_batch_calls[0]
    assert [r.prompt for r in rollouts] == [
        "g0-a",
        "g0-b",
        "g1-a",
        "g1-b",
        "g1-c",
    ]
    assert [r.metadata["target_text"] for r in rollouts] == [
        "group-0",
        "group-0",
        "group-1",
        "group-1",
        "group-1",
    ]
    assert [r.source_request_id for r in rollouts] == [
        "request-0",
        "request-0",
        "request-1",
        "request-1",
        "request-1",
    ]
    assert [r.sample_id for r in rollouts] == [
        "request-0:sample:0",
        "request-0:sample:1",
        "request-1:sample:0",
        "request-1:sample:1",
        "request-1:sample:2",
    ]
    assert [r.group_id for r in rollouts] == [
        "request-0:group:0",
        "request-0:group:1",
        "request-1:group:0",
        "request-1:group:1",
        "request-1:group:2",
    ]
    assert [r.trajectory_id for r in rollouts] == [
        "request-0:trajectory:0",
        "request-0:trajectory:1",
        "request-1:trajectory:0",
        "request-1:trajectory:1",
        "request-1:trajectory:2",
    ]
    assert [r.policy_version for r in rollouts] == [4, 4, 5, 5, 5]
    # Scores split back by group size, in order.
    assert rewards[0].tolist() == [0.0, 1.0]
    assert rewards[1].tolist() == [2.0, 3.0, 4.0]

    assert asyncio.run(scorer.score_many([])).scores == []


def test_collector_attaches_components_to_their_exact_rollout_groups() -> None:
    """Prefetched reward observations travel with the batch they describe."""
    import asyncio

    class _ComponentReward:
        async def score_batch_report(self, rollouts):
            scores = [float(index) for index in range(len(rollouts))]
            return RewardBatchReport(
                scores=scores,
                components={"observer": [value + 10.0 for value in scores]},
            )

    collector = _collector(
        runtime=_Runtime(),
        reward_scorer=RewardScorer(_ComponentReward()),
    )

    async def _collect_two_groups():
        pending = [
            await collector.collect_unscored(["p0"], group_size=2),
            await collector.collect_unscored(["p1"], group_size=3),
        ]
        return await collector.score_rollouts(pending)

    first, second = asyncio.run(_collect_two_groups())

    assert first.rewards.tolist() == [0.0, 1.0]
    assert first.extras["reward_components"]["observer"].tolist() == [10.0, 11.0]
    assert second.rewards.tolist() == [2.0, 3.0, 4.0]
    assert second.extras["reward_components"]["observer"].tolist() == [12.0, 13.0, 14.0]


def test_collect_prompt_batches_folds_reward_timing_into_stats() -> None:
    """Checks reward runtime timing reaches RolloutStats."""
    import asyncio

    class _TimedReward:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        async def score_batch_report(self, rollouts: list[Any]) -> RewardBatchReport:
            self.batch_sizes.append(len(rollouts))
            return RewardBatchReport(
                scores=[float(index + 1) for index in range(len(rollouts))],
                timing_ms={
                    "latency_ms": 12.0,
                    "queue_wait_ms": 3.0,
                    "inference_ms": 9.0,
                    "artifact_validation_ms": 2.0,
                },
            )

    reward_fn = _TimedReward()
    collector = RolloutCollector(
        config=object(),
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
    assert stats.as_phase_dict()["reward.latency_s"] == 0.012
    assert stats.reward_queue_wait_ms == 3.0
    assert stats.reward_inference_ms == 9.0
    assert stats.reward_extra_ms == {"artifact_validation_ms": 2.0}
    assert stats.as_phase_dict()["reward.queue_wait_s"] == 0.003
    assert stats.as_phase_dict()["reward.artifact_validation_s"] == 0.002


def test_reward_view_selection_fails_fast_when_ambiguous() -> None:
    """Checks reward view selection fails fast when ambiguous."""
    import asyncio

    request = GenerationRequest(
        request_id="unit-request",
        family="unit",
        task="collect",
        inputs=["p0"],
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


def test_chunk_denoise_kl_reward_sums_chunk_and_transition_axes() -> None:
    """Latent Gaussian trajectories use denoise packing and per-sample KL."""

    request = GenerationRequest(
        request_id="chunk-request",
        family="causvid",
        task="text_to_video",
        inputs=["p0"],
        samples_per_prompt=2,
    )
    sample_rows = _sample_rows(request)
    batch_size, chunk_count, transition_count = 2, 2, 3
    policy_shape = (batch_size, chunk_count, transition_count)
    trajectory = build_chunk_autoregressive_denoise_trajectory(
        request=request,
        sample_rows=sample_rows,
        observations=torch.zeros(*policy_shape, 1),
        actions=torch.ones(*policy_shape, 1),
        old_log_prob=torch.zeros(policy_shape),
        mask=torch.ones(policy_shape),
        timesteps=torch.zeros(policy_shape),
        finalized_chunk_latents=torch.zeros(batch_size, chunk_count, 1),
        replay_tensors={},
        context={},
        kl=torch.stack(
            (
                torch.ones(chunk_count, transition_count),
                torch.full((chunk_count, transition_count), 2.0),
            )
        ),
    )
    output = GenerationOutput(
        request_id=request.request_id,
        family=request.family,
        task=request.task,
        sample_rows=sample_rows,
        output=torch.zeros(batch_size, 3, 2, 2),
        trajectory=trajectory,
    )

    packed = TrajectoryRolloutBatchBuilder(
        output,
        RolloutBatchBuildContext(metadata={}, kl_reward_coef=0.25),
    ).build(torch.tensor([10.0, 20.0]))

    assert packed.observations.shape[:3] == policy_shape
    assert packed.actions.shape[:3] == policy_shape
    assert packed.rewards.tolist() == pytest.approx([8.5, 17.0])
    assert packed.training_view is not None
    assert packed.training_view.primary_segment == "denoise"


def test_nonlatent_gaussian_keeps_autoregressive_packing() -> None:
    """Gaussian token policies do not get mistaken for latent denoise policies."""

    import asyncio

    request = GenerationRequest(
        request_id="token-gaussian-request",
        family="unit",
        task="text_to_image",
        inputs=["p0"],
        samples_per_prompt=1,
    )
    output = asyncio.run(_Runtime().generate(request))
    assert output.trajectory is not None
    output.trajectory.segments["image_tokens"].distribution = "gaussian"

    packed = TrajectoryRolloutBatchBuilder(
        output,
        RolloutBatchBuildContext(metadata={}),
    ).build(torch.tensor([1.0]))

    assert packed.observations.shape == (1, 1, 3)
    assert packed.actions.shape == (1, 2)
    assert packed.training_view is not None
    assert packed.training_view.primary_segment == "image_tokens"


def test_collector_forwards_reference_metadata_to_request() -> None:
    """Checks collector forwards reference metadata to request."""
    from vrl.families.registry import get_model_family_entry
    from vrl.rollouts.collector.config import RolloutCollectorConfig
    from vrl.rollouts.collector.requests import GenerationRequestBuilder

    builder = GenerationRequestBuilder(
        entry=get_model_family_entry("cosmos-predict2"),
        config=RolloutCollectorConfig(request_sampling={"num_steps": 1}),
    )

    collector_request = builder.build(
        [GenerationInput(prompt="prompt", reference_image="/tmp/reference.png")],
        1,
    )

    assert collector_request.request.inputs[0].reference_image == "/tmp/reference.png"
    assert collector_request.metadata["reference_image"] == "/tmp/reference.png"


def test_collector_forwards_target_metadata_to_request() -> None:
    """Checks collector forwards target artifact metadata to rewards."""
    from vrl.families.registry import get_model_family_entry
    from vrl.rollouts.collector.config import RolloutCollectorConfig
    from vrl.rollouts.collector.requests import GenerationRequestBuilder

    builder = GenerationRequestBuilder(
        entry=get_model_family_entry("cosmos-predict2"),
        config=RolloutCollectorConfig(request_sampling={"num_steps": 1}),
    )

    targets = {"target_image": "/tmp/target.png", "target_video": "/tmp/target.mp4"}
    collector_request = builder.build(
        [
            GenerationInput(
                prompt="prompt",
                reference_image="/tmp/reference.png",
                metadata=dict(targets),
            ),
        ],
        1,
        metadata=dict(targets),
    )

    assert collector_request.request.inputs[0].metadata["target_image"] == "/tmp/target.png"
    assert collector_request.request.inputs[0].metadata["target_video"] == "/tmp/target.mp4"
    assert collector_request.metadata["target_image"] == "/tmp/target.png"
    assert collector_request.metadata["target_video"] == "/tmp/target.mp4"


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
                    group_id=f"g{prompt_index}",
                    sample_id=sample_id,
                    trajectory_id=f"t_{sample_id}",
                    seed=None,
                    metadata={
                        "request_id": request.request_id,
                        "policy_version": request.policy_version,
                    },
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
        inputs=["p0"],
        samples_per_prompt=1,
    )
    output = asyncio.run(_Runtime().generate(request))
    packed = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 16, 16)
    output.output = packed

    reconstructed = TrajectoryRolloutBatchBuilder(
        output,
        RolloutBatchBuildContext(metadata={}),
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

    monkeypatch.setenv("VRL_PROFILE", "1")
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
