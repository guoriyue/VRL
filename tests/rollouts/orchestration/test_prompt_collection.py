"""Topology-safe deferred and streamed reward behavior of prompt collection.

Every collector is the real ``RolloutCollector`` over the tiny SANA in-process
runtime (``real_collector``). Whether scoring streams beside generation or
waits for every group is the lifecycle plan's reading of GPU placement: a
reward on a rollout or trainer GPU needs a handoff, an isolated one does not.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import pytest

from tests.rollouts.collector._helpers import CollectorBench, IndexReward, real_collector
from tests.rollouts.orchestration.continuous._helpers import _wait_until
from vrl.ray.resources import RoleLifecycle
from vrl.rewards import RewardOutput, RewardSample
from vrl.rollouts.collector.core import (
    PromptCollectionCleanupError,
)
from vrl.trainers.data.prompts import PromptExample
from vrl.utils.stats import recording

# The reward shares the rollout GPU: scoring needs a handoff, so the collector
# generates every group first and scores all of them in one call.
_SHARED_REWARD = RoleLifecycle(trainer=(0,), rollout=(1,), reward=(1,))


async def _collect(
    bench: CollectorBench,
    prompts: list[Any],
    *,
    group_size: int = 1,
) -> list[Any]:
    return await bench.collector.prepare_training_batches(
        prompts=prompts,
        group_size=group_size,
        runtime_debug=False,
        policy_version=None,
    )


def _flow(bench: CollectorBench) -> list[str]:
    """Generation and scoring order; parks and offloads are the plan's business."""

    return [event for event in bench.trace.events if event in ("generate", "score")]


def _generated_prompts(bench: CollectorBench) -> list[str]:
    return [",".join(request.prompts) for request in bench.trace.requests]


def _hook_generation(
    monkeypatch: pytest.MonkeyPatch,
    bench: CollectorBench,
    before: Callable[[str], Awaitable[None]],
) -> None:
    """Await ``before(prompt)`` on the driver loop ahead of each real generation.

    This is the dispatch await a Ray runtime has before a request reaches an
    engine; the real worker body still produces the output.
    """

    real = bench.runtime.generate

    async def generate(request: Any) -> Any:
        await before(",".join(request.prompts))
        return await real(request)

    monkeypatch.setattr(bench.runtime, "generate", generate)


@pytest.mark.asyncio
async def test_prompt_examples_generate_all_groups_before_one_scoring_call(
    monkeypatch, tmp_path
) -> None:
    """Checks all PromptExample groups generate before the single score call."""

    bench = real_collector(monkeypatch, tmp_path, lifecycle=_SHARED_REWARD)

    batches = await _collect(
        bench, [PromptExample(prompt=f"p{i}") for i in range(3)], group_size=2
    )

    assert _flow(bench) == ["generate", "generate", "generate", "score"]
    assert _generated_prompts(bench) == ["p0", "p1", "p2"]
    assert bench.reward.calls[0]["prompts"] == ["p0", "p0", "p1", "p1", "p2", "p2"]
    # One split batch per prompt group, remapped to the prompt index.
    assert len(batches) == 3
    for prompt_idx, batch in enumerate(batches):
        assert batch.group_ids.unique().tolist() == [prompt_idx]


@pytest.mark.asyncio
async def test_mixed_prompts_preserve_group_id_remap(monkeypatch, tmp_path) -> None:
    """Checks plain strings and PromptExamples keep their prompt indices."""

    bench = real_collector(monkeypatch, tmp_path, lifecycle=_SHARED_REWARD)

    batches = await _collect(bench, ["s0", PromptExample(prompt="e1"), "s2"])

    assert _flow(bench) == ["generate", "generate", "generate", "score"]
    assert _generated_prompts(bench) == ["s0", "e1", "s2"]
    assert [batch.group_ids.unique().tolist() for batch in batches] == [[0], [1], [2]]


@pytest.mark.asyncio
async def test_phase_times_accumulate_per_call(monkeypatch, tmp_path) -> None:
    """The caller's bound stats collect the generation, score and build phases."""

    bench = real_collector(monkeypatch, tmp_path, lifecycle=_SHARED_REWARD)

    with recording() as stats:
        await _collect(bench, [PromptExample(prompt="p0"), PromptExample(prompt="p1")])

    for phase in ("collect.engine_generate", "collect.reward_score", "collect.batch_build"):
        assert stats.phase_seconds[phase] > 0.0
    # Each group's generate timer starts at submission, so the prefetched
    # group's time overlaps the running one; the sum may exceed the wall.
    assert stats.phase_seconds["collect.wall"] > 0.0
    assert stats.phase_seconds["collect.generation_reward_overlap"] == 0.0
    assert stats.counters == {
        "collect.group_count": 2.0,
        "collect.sample_count": 2.0,
        "reward.call_count": 1.0,
    }


class _GatedReward(IndexReward):
    """Scoring that blocks on explicit gates so a test controls the overlap."""

    def __init__(self, *, fail_score: bool = False, fail_cleanup: bool = False) -> None:
        super().__init__()
        self.fail_score = fail_score
        self.fail_cleanup = fail_cleanup
        self.events: list[str] = []
        self.score_started = asyncio.Event()
        self.finish_score = asyncio.Event()
        self.score_cancelled = asyncio.Event()
        self.active = 0
        self.max_active = 0

    async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
        name = ";".join(dict.fromkeys(sample.prompt for sample in samples))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.events.append(f"score_start:{name}")
        self.score_started.set()
        try:
            if name == "p0":
                await self.finish_score.wait()
            if self.fail_score:
                raise RuntimeError("score failed")
            self.events.append(f"score_done:{name}")
            return await super().score_batch(samples)
        except asyncio.CancelledError as error:
            self.score_cancelled.set()
            if self.fail_cleanup:
                raise RuntimeError("score cleanup failed") from error
            raise
        finally:
            self.active -= 1


def _release_score_when_overlapped(
    reward: _GatedReward, *, fail_generation: bool = False
) -> Callable[[str], Awaitable[None]]:
    """p1's generation waits for p0's score to start, then releases it (or fails)."""

    async def before(prompt: str) -> None:
        if prompt != "p1":
            return
        await reward.score_started.wait()
        reward.events.append("generation_overlapped_score:p1")
        if fail_generation:
            # Fail while p0's score is still in flight: the collector must
            # cancel that score, not let it finish first.
            raise RuntimeError("generation failed")
        reward.finish_score.set()

    return before


@pytest.mark.asyncio
async def test_isolated_reward_overlaps_scoring_with_next_generation(
    monkeypatch, tmp_path
) -> None:
    reward = _GatedReward()
    bench = real_collector(monkeypatch, tmp_path, reward=reward)
    _hook_generation(monkeypatch, bench, _release_score_when_overlapped(reward))
    with recording() as stats:
        batches = await _collect(bench, [PromptExample(prompt="p0"), PromptExample(prompt="p1")])

    assert reward.events == [
        "score_start:p0",
        "generation_overlapped_score:p1",
        "score_done:p0",
        "score_start:p1",
        "score_done:p1",
    ]
    assert reward.max_active == 1
    assert reward.active == 0
    assert [batch.group_ids.item() for batch in batches] == [0, 1]
    assert stats.phase_seconds["collect.generation_reward_overlap"] > 0.0


@pytest.mark.asyncio
async def test_score_failure_is_drained_without_task_leak(monkeypatch, tmp_path) -> None:
    reward = _GatedReward(fail_score=True)
    bench = real_collector(monkeypatch, tmp_path, reward=reward)
    _hook_generation(monkeypatch, bench, _release_score_when_overlapped(reward))

    with pytest.raises(RuntimeError, match="score failed"):
        await _collect(bench, [PromptExample(prompt="p0"), PromptExample(prompt="p1")])

    assert reward.active == 0
    assert reward.max_active == 1
    assert reward.events.count("score_start:p0") == 1
    assert "score_start:p1" not in reward.events


@pytest.mark.asyncio
async def test_collection_cancellation_does_not_detach_score_task(monkeypatch, tmp_path) -> None:
    reward = _GatedReward()
    bench = real_collector(monkeypatch, tmp_path, reward=reward)
    collection = asyncio.create_task(_collect(bench, [PromptExample(prompt="p0")]))
    await reward.score_started.wait()

    collection.cancel()
    with pytest.raises(asyncio.CancelledError):
        await collection

    assert reward.score_cancelled.is_set()
    assert reward.active == 0
    assert reward.max_active == 1
    assert reward.events == ["score_start:p0"]


@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.asyncio
async def test_generation_failure_cancels_and_settles_inflight_score(
    monkeypatch, tmp_path, cleanup_fails: bool
) -> None:
    reward = _GatedReward(fail_cleanup=cleanup_fails)
    bench = real_collector(monkeypatch, tmp_path, reward=reward)
    _hook_generation(
        monkeypatch, bench, _release_score_when_overlapped(reward, fail_generation=True)
    )
    prompts = [PromptExample(prompt="p0"), PromptExample(prompt="p1")]

    if cleanup_fails:
        with pytest.raises(PromptCollectionCleanupError) as raised:
            await _collect(bench, prompts)
        assert str(raised.value.root_cause) == "generation failed"
        assert [str(error) for error in raised.value.cleanup_errors] == [
            "score cleanup failed",
        ]
    else:
        with pytest.raises(RuntimeError, match="generation failed"):
            await _collect(bench, prompts)

    assert reward.active == 0
    assert reward.score_cancelled.is_set()


def _gated_engine(
    monkeypatch: pytest.MonkeyPatch, bench: CollectorBench, names: Sequence[str]
) -> tuple[list[str], dict[str, asyncio.Event]]:
    """Record each generation's submission and completion; hold it at a per-prompt gate."""

    events: list[str] = []
    release = {name: asyncio.Event() for name in names}
    real = bench.runtime.generate

    async def generate(request: Any) -> Any:
        name = ",".join(request.prompts)
        events.append(f"submit:{name}")
        await release[name].wait()
        output = await real(request)
        events.append(f"generated:{name}")
        return output

    monkeypatch.setattr(bench.runtime, "generate", generate)
    return events, release


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_next_generation_is_submitted_before_the_current_one_completes(
    monkeypatch, tmp_path, streaming: bool
) -> None:
    """The engine sees request N+1 while request N is still being finalized."""

    bench = real_collector(monkeypatch, tmp_path, lifecycle=None if streaming else _SHARED_REWARD)
    events, release = _gated_engine(monkeypatch, bench, ["p0", "p1", "p2"])
    collection = asyncio.create_task(
        _collect(bench, [PromptExample(prompt=f"p{i}") for i in range(3)])
    )

    await _wait_until(lambda: len(events) == 2)
    # p1 is already submitted while p0 has not completed; p2 is not.
    assert events == ["submit:p0", "submit:p1"]
    release["p0"].set()
    await _wait_until(lambda: "submit:p2" in events)
    assert events == ["submit:p0", "submit:p1", "generated:p0", "submit:p2"]
    release["p1"].set()
    release["p2"].set()
    batches = await collection

    assert [batch.group_ids.item() for batch in batches] == [0, 1, 2]
    assert events.index("submit:p2") < events.index("generated:p1")


@pytest.mark.asyncio
async def test_generation_failure_cancels_the_prefetched_generation(monkeypatch, tmp_path) -> None:
    """A failed group also cancels the group already submitted behind it."""

    bench = real_collector(monkeypatch, tmp_path)
    events: list[str] = []
    cancelled: list[str] = []
    release = asyncio.Event()

    async def generate(request: Any) -> Any:
        name = ",".join(request.prompts)
        events.append(f"submit:{name}")
        if name == "p0":
            await asyncio.sleep(0)
            raise RuntimeError("generation failed")
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.append(name)
            raise
        raise AssertionError("p1 must be cancelled, never completed")

    monkeypatch.setattr(bench.runtime, "generate", generate)

    with pytest.raises(RuntimeError, match="generation failed"):
        await _collect(bench, [PromptExample(prompt="p0"), PromptExample(prompt="p1")])

    assert events == ["submit:p0", "submit:p1"]
    assert cancelled == ["p1"]


@pytest.mark.asyncio
async def test_generation_handoff_is_demand_driven_and_never_scores(monkeypatch, tmp_path) -> None:
    bench = real_collector(monkeypatch, tmp_path)
    collector = bench.collector
    groups = collector.build_generation_requests(
        prompts=["plain", PromptExample(prompt="example"), "later"],
        group_size=2,
        runtime_debug=False,
        policy_version=None,
    )
    first_request, first_indices = next(groups)
    assert _flow(bench) == []
    first = await collector.generate_rollout(first_request)
    assert first_indices == [0]
    assert _generated_prompts(bench) == ["plain"]
    second_request, second_indices = next(groups)
    second = await collector.generate_rollout(second_request)
    assert second_indices == [1]
    assert _generated_prompts(bench) == ["plain", "example"]
    groups.close()
    # Closing admission must not generate the remaining prompt or start reward.
    assert _flow(bench) == ["generate", "generate"]
    scored = collector.assemble_training_batches(await collector.evaluate_rollout([first, second]))
    assert _flow(bench) == ["generate", "generate", "score"]
    assert len(scored) == 2
    assert all(batch.rewards.shape == (2,) for batch in scored)
