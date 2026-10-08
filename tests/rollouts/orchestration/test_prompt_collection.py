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
import torch

from tests.rollouts.collector._helpers import CollectorBench, IndexReward, real_collector
from vrl.ray.resources import RayLifecyclePlan
from vrl.rewards import RewardOutput, RewardSample
from vrl.rollouts.collector.core import (
    PromptCollectionCleanupError,
    RolloutGenerationResult,
)
from vrl.rollouts.evaluators.trajectory import TrajectorySignalBuilder
from vrl.rollouts.stats import RolloutStats
from vrl.trainers.data.prompts import PromptExample

# The reward shares the rollout GPU: scoring needs a handoff, so the collector
# generates every group first and scores all of them in one call.
_SHARED_REWARD = RayLifecyclePlan(trainer=(0,), rollout=(1,), reward=(1,))


async def _collect(
    bench: CollectorBench,
    prompts: list[Any],
    *,
    group_size: int = 1,
    stats: RolloutStats | None = None,
) -> list[Any]:
    return await bench.collector.prepare_training_batches(
        prompts=prompts,
        group_size=group_size,
        runtime_debug=False,
        policy_version=None,
        stats=stats if stats is not None else RolloutStats(),
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


async def _wait_for(condition: Callable[[], bool], timeout_s: float = 10.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not condition():
        if loop.time() >= deadline:
            raise AssertionError("condition not reached before timeout")
        await asyncio.sleep(0.001)


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
async def test_prompt_example_scalar_remap_updates_trainer_group_ids(
    monkeypatch, tmp_path
) -> None:
    """Checks remaps update trainer grouping without rewriting stable identity."""

    bench = real_collector(monkeypatch, tmp_path)

    batches = await _collect(
        bench, [PromptExample(prompt="p0"), PromptExample(prompt="p1")], group_size=2
    )

    for expected_group, batch in enumerate(batches):
        expected = torch.full((2,), expected_group, dtype=torch.long)
        assert torch.equal(batch.group_ids, expected)
        assert batch.trajectory is not None
        assert torch.equal(TrajectorySignalBuilder(batch).group_ids, expected)
        assert [row.prompt_index for row in batch.trajectory.sample_rows] == [0, 0]


@pytest.mark.asyncio
async def test_plain_string_list_remap_updates_signal_group_ids(monkeypatch, tmp_path) -> None:
    """Checks evaluator signals consume the remapped trainer-owned groups."""

    bench = real_collector(monkeypatch, tmp_path)

    batches = await _collect(bench, ["p0", "p1"])

    assert [batch.group_ids.item() for batch in batches] == [0, 1]
    assert [TrajectorySignalBuilder(batch).group_ids.item() for batch in batches] == [0, 1]


@pytest.mark.asyncio
async def test_phase_times_accumulate_per_call(monkeypatch, tmp_path) -> None:
    """The out-param sums generation per group and score/build once per call."""

    monkeypatch.setenv("VRL_PROFILE", "1")
    bench = real_collector(monkeypatch, tmp_path, lifecycle=_SHARED_REWARD)
    bench.trace.watch(bench.collector, "generate_rollout", "generate_rollout")
    stats = RolloutStats()

    await _collect(bench, [PromptExample(prompt="p0"), PromptExample(prompt="p1")], stats=stats)

    groups = bench.trace.results["generate_rollout"]
    assert len(groups) == 2
    assert stats.phase_seconds["collect.engine_generate"] == pytest.approx(
        sum(group.phases["collect.engine_generate"] for group in groups)
    )
    # Call-level score/build timings land on the first group only.
    assert stats.phase_seconds["collect.reward_score"] == groups[0].phases["collect.reward_score"]
    assert stats.phase_seconds["collect.batch_build"] == groups[0].phases["collect.batch_build"]
    assert "collect.reward_score" not in groups[1].phases
    # Each group's generate timer starts at submission, so the prefetched
    # group's time overlaps the running one; the sum may exceed the wall.
    assert stats.phase_seconds["collect.wall"] > 0.0
    assert stats.phase_seconds["collect.generation_reward_overlap"] == 0.0
    assert stats.counters == {
        "collect.group_count": 2.0,
        "collect.sample_count": 2.0,
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


class _SlowReward(IndexReward):
    """Scoring whose cost is proportional to the batch, like a reward model."""

    def __init__(self, delay_s: float) -> None:
        super().__init__()
        self.delay_s = delay_s

    async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
        await asyncio.sleep(self.delay_s * len(samples))
        return await super().score_batch(samples)


@pytest.mark.asyncio
async def test_overlap_stats_support_batched_serial_vs_streaming_wall_ab(
    monkeypatch, tmp_path
) -> None:
    prompts = [PromptExample(prompt="p0"), PromptExample(prompt="p1")]
    # Enough sampling steps that a group's generation is comparable to its
    # scoring, so the streamed schedule has something to overlap.
    steps = ("sampling.num_steps=50",)
    serial = real_collector(
        monkeypatch,
        tmp_path / "serial",
        reward=_SlowReward(0.05),
        lifecycle=_SHARED_REWARD,
        overrides=steps,
    )
    # Each stack serves one snapshot; load the first policy before the second
    # stack installs its own pipeline. Both policies are loaded before timing
    # so neither wall includes a model build.
    await serial.collector.activate_generation_runtime()
    streaming = real_collector(
        monkeypatch, tmp_path / "streaming", reward=_SlowReward(0.05), overrides=steps
    )
    await streaming.collector.activate_generation_runtime()
    serial_stats = RolloutStats()
    overlap_stats = RolloutStats()

    await _collect(serial, prompts, stats=serial_stats)
    await _collect(streaming, prompts, stats=overlap_stats)

    # Wall time is noisy under load (the two walls sit within milliseconds of
    # each other on a loaded host); the overlap measurement is the signal: the
    # serial schedule never overlaps, the streamed one overlaps at least one
    # reward's scoring with the next group's generation.
    assert serial_stats.phase_seconds["collect.generation_reward_overlap"] == 0.0
    assert overlap_stats.phase_seconds["collect.generation_reward_overlap"] >= 0.02
    assert overlap_stats.counters == {
        "collect.group_count": 2.0,
        "collect.sample_count": 2.0,
    }


@pytest.mark.parametrize(
    "reward_devices",
    [(1,), (0,), (0, 1)],
    ids=["shares_rollout_gpu", "shares_trainer_gpu", "shares_both"],
)
@pytest.mark.asyncio
async def test_reward_handoff_keeps_generation_and_scoring_serial(
    monkeypatch, tmp_path, reward_devices: tuple[int, ...]
) -> None:
    bench = real_collector(
        monkeypatch,
        tmp_path,
        lifecycle=RayLifecyclePlan(trainer=(0,), rollout=(1,), reward=reward_devices),
    )

    await _collect(bench, [PromptExample(prompt="p0"), PromptExample(prompt="p1")])

    assert _flow(bench) == ["generate", "generate", "score"]


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

    await _wait_for(lambda: len(events) == 2)
    # p1 is already submitted while p0 has not completed; p2 is not.
    assert events == ["submit:p0", "submit:p1"]
    release["p0"].set()
    await _wait_for(lambda: "submit:p2" in events)
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
    first = RolloutGenerationResult(
        await collector.generate_rollout(first_request), first_indices, 0.0, 0.0
    )
    assert first.prompt_indices == [0]
    assert first.completed_at >= first.started_at
    assert _generated_prompts(bench) == ["plain"]
    second_request, second_indices = next(groups)
    second = RolloutGenerationResult(
        await collector.generate_rollout(second_request), second_indices, 0.0, 0.0
    )
    assert second.prompt_indices == [1]
    assert _generated_prompts(bench) == ["plain", "example"]
    groups.close()
    # Closing admission must not generate the remaining prompt or start reward.
    assert _flow(bench) == ["generate", "generate"]
    scored = collector.assemble_training_batches(
        await collector.evaluate_rollout([first.unscored, second.unscored])
    )
    assert _flow(bench) == ["generate", "generate", "score"]
    assert len(scored) == 2
    assert all(batch.rewards.shape == (2,) for batch in scored)
