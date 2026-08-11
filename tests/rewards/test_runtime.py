"""Collector-facing reward runtime contract tests."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from vrl.rewards import RewardOutput, RewardRuntime, RewardSample
from vrl.rewards.base import RewardFunction
from vrl.rewards.runtime import RewardFunctionRuntime


def _sample(sample_id: str = "sample-0") -> RewardSample:
    return RewardSample(prompt="prompt", output=object(), sample_id=sample_id)


def test_reward_sample_requires_a_nonempty_string_diagnostic_id() -> None:
    with pytest.raises(TypeError, match=r"RewardSample\.sample_id must be a str"):
        _sample(7)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"RewardSample\.sample_id must be non-empty"):
        _sample("")


def test_reward_output_normalizes_and_validates_observations() -> None:
    output = RewardOutput(
        scores=(1, 2),
        components={"quality": [0.5, 0.75]},
        timing_ms={"latency_ms": 4},
    )

    assert output.scores == (1.0, 2.0)
    assert output.components == {"quality": (0.5, 0.75)}
    assert output.timing_ms == {"latency_ms": 4.0}


def test_reward_output_rejects_misaligned_components() -> None:
    with pytest.raises(ValueError, match="component/score mismatch"):
        RewardOutput(scores=(1.0,), components={"quality": ()})


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_reward_output_rejects_non_finite_scores(score: float) -> None:
    with pytest.raises(ValueError, match="scores must contain only finite"):
        RewardOutput(scores=(score,))
    with pytest.raises(ValueError, match=r"component .* finite"):
        RewardOutput(scores=(1.0,), components={"quality": (score,)})


@pytest.mark.parametrize("timing", [float("nan"), float("inf"), -1.0])
def test_reward_output_rejects_invalid_timing(timing: float) -> None:
    with pytest.raises(ValueError, match=r"timing .* finite and non-negative"):
        RewardOutput(scores=(1.0,), timing_ms={"latency_ms": timing})


@pytest.mark.asyncio
async def test_function_runtime_returns_the_function_output_directly() -> None:
    class _ReportingReward(RewardFunction):
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
            self.calls.append([sample.sample_id for sample in samples])
            return RewardOutput(
                scores=(1.25, 2.5),
                components={"quality": (0.2, 0.4)},
                timing_ms={"latency_ms": 7.0},
            )

    reward = _ReportingReward()
    runtime = RewardFunctionRuntime(reward)
    samples = (_sample("sample-0"), _sample("sample-1"))

    output = await runtime.score(samples)

    assert isinstance(runtime, RewardRuntime)
    assert reward.calls == [["sample-0", "sample-1"]]
    assert output == RewardOutput(
        scores=(1.25, 2.5),
        components={"quality": (0.2, 0.4)},
        timing_ms={"latency_ms": 7.0},
    )


@pytest.mark.asyncio
async def test_function_runtime_validates_sample_and_score_alignment() -> None:
    runtime = RewardFunctionRuntime(None)
    with pytest.raises(ValueError, match="at least one sample"):
        await runtime.score(())
    sample = _sample()
    with pytest.raises(ValueError, match="sample_id values must be unique"):
        await runtime.score((sample, sample))

    class _ShortReward(RewardFunction):
        async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
            return RewardOutput(scores=())

    with pytest.raises(ValueError, match="wrong number of scores"):
        await RewardFunctionRuntime(_ShortReward()).score((sample,))


@pytest.mark.asyncio
async def test_function_runtime_without_reward_returns_aligned_zeros() -> None:
    runtime = RewardFunctionRuntime(None)

    output = await runtime.score((_sample("sample-0"), _sample("sample-1")))

    assert output.scores == (0.0, 0.0)
    assert runtime.scoring_is_nonblocking is False
    assert runtime.external_accelerator_isolation_verified is True


@pytest.mark.asyncio
async def test_required_parking_is_atomic_with_scoring() -> None:
    class _BlockingReward(RewardFunction):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.events: list[str] = []

        async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
            self.events.append("score_start")
            self.started.set()
            await self.release.wait()
            self.events.append("score_end")
            return RewardOutput(scores=(1.0,) * len(samples))

        async def park_memory(self) -> bool:
            self.events.append("park")
            return True

    reward = _BlockingReward()
    runtime = RewardFunctionRuntime(reward)
    score_task = asyncio.create_task(
        runtime.score((_sample(),), require_memory_release=True),
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
async def test_required_parking_fails_when_no_owner_parked() -> None:
    class _UnparkableReward(RewardFunction):
        async def score(self, sample: RewardSample) -> float:
            return 1.0

    runtime = RewardFunctionRuntime(_UnparkableReward())

    with pytest.raises(RuntimeError, match="no active memory-parking owner"):
        await runtime.score((_sample(),), require_memory_release=True)


@pytest.mark.asyncio
async def test_function_runtime_forwards_capabilities_and_retries_shutdown() -> None:
    class _LifecycleReward(RewardFunction):
        def __init__(self) -> None:
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
        await runtime.score((_sample(),))
    await runtime.shutdown()
    await runtime.shutdown()
    assert reward.shutdown_calls == 2
