"""Collector-facing reward runtime contract tests."""

from __future__ import annotations

import asyncio

import pytest

from vrl.rewards import RewardOutput, RewardRequest, RewardRuntime, RewardSample
from vrl.rewards.base import RewardBatchReport, RewardFunction
from vrl.rewards.inference import RewardMemoryReleaseProof
from vrl.rewards.runtime import RewardFunctionRuntime


def _sample(sample_id: str = "request-0:sample:0") -> RewardSample:
    return RewardSample(
        prompt="prompt",
        output=object(),
        source_request_id="request-0",
        sample_id=sample_id,
        group_id="request-0:prompt:0",
        trajectory_id=sample_id,
    )


def _request(*samples: RewardSample) -> RewardRequest:
    return RewardRequest(request_id="reward-0", samples=tuple(samples or (_sample(),)))


def test_reward_request_requires_unique_nonempty_samples() -> None:
    with pytest.raises(ValueError, match="samples must be non-empty"):
        RewardRequest(request_id="reward-0", samples=())

    sample = _sample()
    with pytest.raises(ValueError, match="sample_id values must be unique"):
        _request(sample, sample)


def test_reward_sample_request_and_output_require_string_ids() -> None:
    with pytest.raises(TypeError, match=r"RewardSample\.sample_id must be a str"):
        _sample(7)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"RewardRequest\.request_id must be a str"):
        RewardRequest(request_id=7, samples=(_sample(),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sample_ids must contain str"):
        RewardOutput(
            request_id="reward-0",
            sample_ids=(7,),  # type: ignore[arg-type]
            scores=(1.0,),
        )


def test_reward_output_validates_request_identity_and_order() -> None:
    request = _request(_sample("sample-0"), _sample("sample-1"))
    output = RewardOutput(
        request_id=request.request_id,
        sample_ids=("sample-0", "sample-1"),
        scores=(1, 2),
        components={"quality": [0.5, 0.75]},
        timing_ms={"latency_ms": 4},
    )

    output.validate(request)
    assert output.scores == (1.0, 2.0)
    assert output.components == {"quality": (0.5, 0.75)}
    assert output.timing_ms == {"latency_ms": 4.0}

    with pytest.raises(ValueError, match="different request"):
        RewardOutput(
            request_id="reward-other",
            sample_ids=output.sample_ids,
            scores=output.scores,
        ).validate(request)
    with pytest.raises(ValueError, match="sample order mismatch"):
        RewardOutput(
            request_id=request.request_id,
            sample_ids=tuple(reversed(output.sample_ids)),
            scores=output.scores,
        ).validate(request)


def test_reward_output_rejects_misaligned_scores_and_components() -> None:
    with pytest.raises(ValueError, match="sample_ids must be non-empty"):
        RewardOutput(request_id="reward-0", sample_ids=(), scores=())
    with pytest.raises(ValueError, match="score/sample mismatch"):
        RewardOutput(
            request_id="reward-0",
            sample_ids=("sample-0",),
            scores=(),
        )
    with pytest.raises(ValueError, match="component/sample mismatch"):
        RewardOutput(
            request_id="reward-0",
            sample_ids=("sample-0",),
            scores=(1.0,),
            components={"quality": ()},
        )


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_reward_output_rejects_non_finite_scores(score: float) -> None:
    with pytest.raises(ValueError, match="scores must contain only finite"):
        RewardOutput(
            request_id="reward-0",
            sample_ids=("sample-0",),
            scores=(score,),
        )
    with pytest.raises(ValueError, match=r"component .* finite"):
        RewardOutput(
            request_id="reward-0",
            sample_ids=("sample-0",),
            scores=(1.0,),
            components={"quality": (score,)},
        )


@pytest.mark.parametrize("timing", [float("nan"), float("inf"), -1.0])
def test_reward_output_rejects_invalid_timing(timing: float) -> None:
    with pytest.raises(ValueError, match=r"timing .* finite and non-negative"):
        RewardOutput(
            request_id="reward-0",
            sample_ids=("sample-0",),
            scores=(1.0,),
            timing_ms={"latency_ms": timing},
        )


@pytest.mark.asyncio
async def test_function_runtime_returns_one_correlated_batch_report() -> None:
    class _ReportingReward(RewardFunction):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[list[str]] = []

        async def score_batch_report(
            self,
            samples: list[RewardSample],
        ) -> RewardBatchReport:
            self.calls.append([sample.sample_id for sample in samples])
            return RewardBatchReport(
                scores=[1.25, 2.5],
                components={"quality": [0.2, 0.4]},
                timing_ms={"latency_ms": 7.0},
            )

    reward = _ReportingReward()
    runtime = RewardFunctionRuntime(reward)
    request = _request(_sample("sample-0"), _sample("sample-1"))

    output = await runtime.score(request)

    assert isinstance(runtime, RewardRuntime)
    assert reward.calls == [["sample-0", "sample-1"]]
    assert output == RewardOutput(
        request_id="reward-0",
        sample_ids=("sample-0", "sample-1"),
        scores=(1.25, 2.5),
        components={"quality": (0.2, 0.4)},
        timing_ms={"latency_ms": 7.0},
    )


@pytest.mark.asyncio
async def test_function_runtime_without_reward_returns_aligned_zeros() -> None:
    runtime = RewardFunctionRuntime(None)
    request = _request(_sample("sample-0"), _sample("sample-1"))

    output = await runtime.score(request)

    assert output.sample_ids == ("sample-0", "sample-1")
    assert output.scores == (0.0, 0.0)
    assert runtime.scoring_is_nonblocking is False
    assert runtime.external_accelerator_isolation_verified is True


@pytest.mark.asyncio
async def test_required_parking_is_atomic_with_scoring() -> None:
    class _BlockingReward(RewardFunction):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.events: list[str] = []

        async def score_batch_report(
            self,
            samples: list[RewardSample],
        ) -> RewardBatchReport:
            self.events.append("score_start")
            self.started.set()
            await self.release.wait()
            self.events.append("score_end")
            return RewardBatchReport(scores=[1.0] * len(samples))

        async def park_memory(self) -> tuple[RewardMemoryReleaseProof, ...]:
            self.events.append("park")
            return (
                RewardMemoryReleaseProof(
                    request_id="inference-0",
                    released=True,
                ),
            )

    reward = _BlockingReward()
    runtime = RewardFunctionRuntime(reward)
    score_task = asyncio.create_task(
        runtime.score(_request(), require_memory_release=True),
    )
    await reward.started.wait()
    second_park = asyncio.create_task(runtime.park_memory(required=False))
    await asyncio.sleep(0)

    assert reward.events == ["score_start"]
    reward.release.set()
    await score_task
    await second_park
    assert reward.events == ["score_start", "score_end", "park", "park"]


@pytest.mark.asyncio
async def test_required_parking_rejects_an_empty_proof() -> None:
    class _UnparkableReward(RewardFunction):
        async def score_batch_report(
            self,
            samples: list[RewardSample],
        ) -> RewardBatchReport:
            return RewardBatchReport(scores=[1.0] * len(samples))

    runtime = RewardFunctionRuntime(_UnparkableReward())

    with pytest.raises(RuntimeError, match="empty memory release proof"):
        await runtime.score(_request(), require_memory_release=True)


@pytest.mark.asyncio
async def test_required_parking_validates_proof_residual() -> None:
    class _IncompleteParkingReward(RewardFunction):
        async def score_batch_report(
            self,
            samples: list[RewardSample],
        ) -> RewardBatchReport:
            return RewardBatchReport(scores=[1.0] * len(samples))

        async def park_memory(self) -> tuple[RewardMemoryReleaseProof, ...]:
            return (
                RewardMemoryReleaseProof(
                    request_id="inference-0",
                    released=True,
                    residual_gpu_used_bytes=1,
                ),
            )

    runtime = RewardFunctionRuntime(_IncompleteParkingReward())

    with pytest.raises(RuntimeError, match="incomplete reward memory parking"):
        await runtime.score(_request(), require_memory_release=True)


@pytest.mark.asyncio
async def test_required_parking_rejects_proof_without_request_identity() -> None:
    class _UncorrelatedParkingReward(RewardFunction):
        async def score_batch_report(
            self,
            samples: list[RewardSample],
        ) -> RewardBatchReport:
            return RewardBatchReport(scores=[1.0] * len(samples))

        async def park_memory(self) -> tuple[RewardMemoryReleaseProof, ...]:
            return (RewardMemoryReleaseProof(request_id="", released=True),)

    runtime = RewardFunctionRuntime(_UncorrelatedParkingReward())

    with pytest.raises(ValueError, match="proof request_id must be non-empty"):
        await runtime.score(_request(), require_memory_release=True)


@pytest.mark.asyncio
async def test_function_runtime_forwards_preflight_capabilities_and_retries_shutdown() -> None:
    class _LifecycleReward(RewardFunction):
        def __init__(self) -> None:
            super().__init__()
            self.preflight_calls = 0
            self.shutdown_calls = 0

        @property
        def scoring_is_nonblocking(self) -> bool:
            return True

        @property
        def external_accelerator_isolation_verified(self) -> bool:
            return False

        async def preflight(self) -> None:
            self.preflight_calls += 1

        async def shutdown(self) -> None:
            self.shutdown_calls += 1
            if self.shutdown_calls == 1:
                raise RuntimeError("transient shutdown failure")

    reward = _LifecycleReward()
    runtime = RewardFunctionRuntime(reward)

    assert runtime.scoring_is_nonblocking is True
    assert runtime.external_accelerator_isolation_verified is False
    await runtime.preflight()
    assert reward.preflight_calls == 1
    with pytest.raises(RuntimeError, match="transient shutdown failure"):
        await runtime.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        await runtime.score(_request())
    await runtime.shutdown()
    await runtime.shutdown()
    assert reward.shutdown_calls == 2
    with pytest.raises(RuntimeError, match="shut down"):
        await runtime.score(_request())
