"""Collector-facing reward runtime contract tests."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from vrl.rewards import RewardOutput, RewardSample
from vrl.rewards.base import RewardFunction
from vrl.rewards.runtime import RewardFunctionRuntime
from vrl.utils.deadline import OperationTimeout


def _sample(sample_id: str = "sample-0") -> RewardSample:
    return RewardSample(prompt="prompt", output=object(), sample_id=sample_id)


@pytest.mark.asyncio
async def test_score_deadline_bounds_awaitable_scoring() -> None:
    """A scorer stuck at an await point raises the shared terminal timeout."""

    class _StuckReward(RewardFunction):
        async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
            await asyncio.sleep(3600)
            raise AssertionError("unreachable")

    runtime = RewardFunctionRuntime(_StuckReward(), score_timeout_s=0.05)

    with pytest.raises(OperationTimeout, match=r"reward\.score"):
        await runtime.score((_sample(),))


def test_reward_sample_requires_a_nonempty_diagnostic_id() -> None:
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


def test_reward_output_rejects_non_finite_scores() -> None:
    with pytest.raises(ValueError, match="scores must contain only finite"):
        RewardOutput(scores=(float("nan"),))
    with pytest.raises(ValueError, match=r"component .* finite"):
        RewardOutput(scores=(1.0,), components={"quality": (float("inf"),)})


def test_reward_output_rejects_invalid_timing() -> None:
    with pytest.raises(ValueError, match=r"timing .* finite and non-negative"):
        RewardOutput(scores=(1.0,), timing_ms={"latency_ms": -1.0})


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

    assert reward.calls == [["sample-0", "sample-1"]]
    assert output == RewardOutput(
        scores=(1.25, 2.5),
        components={"quality": (0.2, 0.4)},
        timing_ms={"latency_ms": 7.0},
    )


@pytest.mark.asyncio
async def test_function_runtime_rejects_empty_and_duplicate_samples() -> None:
    class _ZeroReward(RewardFunction):
        async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
            return RewardOutput(scores=(0.0,) * len(samples))

    runtime = RewardFunctionRuntime(_ZeroReward())
    with pytest.raises(ValueError, match="at least one sample"):
        await runtime.score(())
    sample = _sample()
    with pytest.raises(ValueError, match="sample_id values must be unique"):
        await runtime.score((sample, sample))


@pytest.mark.asyncio
async def test_parking_waits_for_scoring_to_release_the_function() -> None:
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
    score_task = asyncio.create_task(runtime.score((_sample(),)))
    await reward.started.wait()
    park = asyncio.create_task(runtime.park_memory())
    await asyncio.sleep(0)

    # The park cannot take the device from a function still scoring.
    assert reward.events == ["score_start"]
    reward.release.set()
    await score_task
    await park
    # A second park finds nothing held and does not reach the function.
    await runtime.park_memory()
    assert reward.events == ["score_start", "score_end", "park"]


@pytest.mark.asyncio
async def test_failed_activation_leaves_nothing_to_park() -> None:
    """The phase-final park after a failed activate is a no-op, so the activation
    error surfaces alone and the trainer's own cleanup can still run."""

    class _BrokenActivation(RewardFunction):
        def __init__(self) -> None:
            self.park_calls = 0

        async def activate(self) -> None:
            raise RuntimeError("model build failed")

        async def park_memory(self) -> bool:
            self.park_calls += 1
            return False

        async def score(self, sample: RewardSample) -> float:
            return 1.0

    reward = _BrokenActivation()
    runtime = RewardFunctionRuntime(reward)
    with pytest.raises(RuntimeError, match="model build failed"):
        await runtime.activate()

    await runtime.park_memory()
    assert reward.park_calls == 0


@pytest.mark.asyncio
async def test_required_parking_fails_when_no_owner_parked() -> None:
    class _UnparkableReward(RewardFunction):
        async def score(self, sample: RewardSample) -> float:
            return 1.0

    runtime = RewardFunctionRuntime(_UnparkableReward())
    await runtime.score((_sample(),))

    with pytest.raises(RuntimeError, match="no active memory-parking owner"):
        await runtime.park_memory()


@pytest.mark.asyncio
async def test_function_runtime_forwards_capabilities_and_retries_shutdown() -> None:
    class _LifecycleReward(RewardFunction):
        def __init__(self) -> None:
            self.preflight_calls = 0
            self.shutdown_calls = 0

        async def preflight(self) -> None:
            self.preflight_calls += 1

        async def shutdown(self) -> None:
            self.shutdown_calls += 1
            if self.shutdown_calls == 1:
                raise RuntimeError("transient shutdown failure")

    reward = _LifecycleReward()
    runtime = RewardFunctionRuntime(reward)

    await runtime.preflight()
    assert reward.preflight_calls == 1
    with pytest.raises(RuntimeError, match="transient shutdown failure"):
        await runtime.shutdown()
    with pytest.raises(RuntimeError, match="reward runtime is shutting down"):
        await runtime.score((_sample(),))
    await runtime.shutdown()
    await runtime.shutdown()
    assert reward.shutdown_calls == 2
