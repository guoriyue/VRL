"""Topology-safe deferred and streamed reward behavior of prompt collection."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from tests.rollouts.collector._helpers import PromptCollectionFake
from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.collector.core import (
    CollectionSchedule,
    PromptCollectionCleanupError,
    RewardCollectionMode,
    RolloutGenerationResult,
)
from vrl.rollouts.evaluators.trajectory import TrajectorySignalBuilder
from vrl.rollouts.stats import RolloutStats
from vrl.trainers.data.prompts import PromptExample
from vrl.trajectory.builders import build_diffusion_trajectory


def prepare_training_batches(*, collector, stats: RolloutStats | None = None, **kwargs):
    """Test shim: production requires the accumulator (both real callers pass
    one); tests that do not assert on stats hand in a throwaway."""

    return collector.prepare_training_batches(
        stats=stats if stats is not None else RolloutStats(), **kwargs
    )


def _batch(prompts: list[str], group_size: int) -> RolloutBatch:
    batch_size = len(prompts) * group_size
    group_ids = torch.tensor(
        [prompt_idx for prompt_idx in range(len(prompts)) for _ in range(group_size)],
        dtype=torch.long,
    )
    return RolloutBatch(
        rewards=torch.arange(batch_size, dtype=torch.float32),
        group_ids=group_ids,
    )


def _batch_with_trajectory(prompts: list[str], group_size: int) -> RolloutBatch:
    """Build a trajectory-backed trainer batch for remap regressions."""

    batch = _batch(prompts, group_size)
    sample_rows = [
        GenerationSampleRow(
            prompt_index=prompt_index,
            sample_index=sample_index,
            prompt=prompt,
            sample_id=f"sample-{prompt_index}-{sample_index}",
        )
        for prompt_index, prompt in enumerate(prompts)
        for sample_index in range(group_size)
    ]
    request = GenerationRequest(
        request_id="remap-request",
        family="fake",
        task="t2i",
        inputs=prompts,
        samples_per_prompt=group_size,
    )
    batch_size = len(sample_rows)
    batch.trajectory = build_diffusion_trajectory(
        request=request,
        sample_rows=sample_rows,
        observations=torch.zeros(batch_size, 1, 1),
        actions=torch.zeros(batch_size, 1, 1),
        old_log_prob=torch.zeros(batch_size, 1),
        timesteps=torch.zeros(batch_size, 1),
        kl=torch.zeros(batch_size, 1),
        replay_tensors={},
        context={},
    )
    return batch


class _DeferredCollector(PromptCollectionFake):
    """Two-phase collector fake recording event order."""

    def __init__(
        self,
        *,
        rollout_reward_handoff: bool = True,
        trainer_reward_handoff: bool = False,
        supports_overlap: bool = False,
    ) -> None:
        self.events: list[str] = []
        self._prompt_names: dict[int, tuple[str, ...]] = {}
        self.requires_generation_offload_before_reward = rollout_reward_handoff
        self.requires_driver_model_offload_for_reward = trainer_reward_handoff
        self.supports_reward_generation_overlap = supports_overlap

    async def generate_rollout(self, request) -> Any:
        inputs = request.inputs
        kwargs = request.options
        prompts = [getattr(item, "prompt", item) for item in inputs]
        self.events.append(f"generate:{','.join(prompts)}")
        batch = _batch(prompts, int(kwargs["group_size"]))
        self._prompt_names[id(batch)] = tuple(prompts)
        return batch

    async def evaluate_rollout(self, pendings: list[Any]) -> list[RolloutBatch]:
        names = [",".join(dict.fromkeys(self._prompt_names[id(pending)])) for pending in pendings]
        self.events.append(f"evaluate_rollout:[{';'.join(names)}]")
        return list(pendings)


class _TrajectoryDeferredCollector(_DeferredCollector):
    """Deferred collector whose trainer and trajectory grouping never alias."""

    async def generate_rollout(self, request) -> RolloutBatch:
        inputs = request.inputs
        kwargs = request.options
        prompts = [getattr(item, "prompt", item) for item in inputs]
        self.events.append(f"generate:{','.join(prompts)}")
        batch = _batch_with_trajectory(prompts, int(kwargs["group_size"]))
        self._prompt_names[id(batch)] = tuple(prompts)
        return batch


@pytest.mark.asyncio
async def test_prompt_examples_generate_all_groups_before_one_scoring_call() -> None:
    """Checks all PromptExample groups generate before the single score call."""
    collector = _DeferredCollector()
    prompts = [PromptExample(prompt=f"p{i}") for i in range(3)]

    batches = await prepare_training_batches(
        collector=collector,
        prompts=prompts,
        group_size=2,
        runtime_debug=False,
        policy_version=7,
    )

    assert collector.events == [
        "generate:p0",
        "generate:p1",
        "generate:p2",
        "evaluate_rollout:[p0;p1;p2]",
    ]
    # One split batch per prompt group, remapped to the prompt index.
    assert len(batches) == 3
    for prompt_idx, batch in enumerate(batches):
        assert batch.group_ids.unique().tolist() == [prompt_idx]
        assert not hasattr(batch, "prompts")


@pytest.mark.asyncio
async def test_mixed_prompts_preserve_group_id_remap() -> None:
    """Checks plain strings and PromptExamples keep their prompt indices."""
    collector = _DeferredCollector()
    prompts: list[Any] = ["s0", PromptExample(prompt="e1"), "s2"]

    batches = await prepare_training_batches(
        collector=collector,
        prompts=prompts,
        group_size=1,
        runtime_debug=False,
        policy_version=None,
    )

    # Strings before the example flush as one group; scoring stays one call.
    assert collector.events == [
        "generate:s0",
        "generate:e1",
        "generate:s2",
        "evaluate_rollout:[s0;e1;s2]",
    ]
    assert [batch.group_ids.unique().tolist() for batch in batches] == [[0], [1], [2]]


@pytest.mark.asyncio
async def test_prompt_example_scalar_remap_updates_trainer_group_ids() -> None:
    """Checks remaps update trainer grouping without rewriting stable identity."""

    batches = await prepare_training_batches(
        collector=_TrajectoryDeferredCollector(),
        prompts=[PromptExample(prompt="p0"), PromptExample(prompt="p1")],
        group_size=2,
        runtime_debug=False,
        policy_version=None,
    )

    for expected_group, batch in enumerate(batches):
        expected = torch.full((2,), expected_group, dtype=torch.long)
        assert torch.equal(batch.group_ids, expected)
        assert batch.trajectory is not None
        assert torch.equal(TrajectorySignalBuilder(batch).group_ids, expected)
        assert [row.prompt_index for row in batch.trajectory.sample_rows] == [0, 0]


@pytest.mark.asyncio
async def test_plain_string_list_remap_updates_signal_group_ids() -> None:
    """Checks evaluator signals consume the remapped trainer-owned groups."""

    batches = await prepare_training_batches(
        collector=_TrajectoryDeferredCollector(),
        prompts=["p0", "p1"],
        group_size=1,
        runtime_debug=False,
        policy_version=None,
    )

    assert [batch.group_ids.item() for batch in batches] == [0, 1]
    assert [TrajectorySignalBuilder(batch).group_ids.item() for batch in batches] == [0, 1]


@dataclass
class _Unscored:
    batch: RolloutBatch
    phases: dict[str, float]


class _PhasedCollector(PromptCollectionFake):
    """Collector fake exposing per-call phase timings like RolloutCollector."""

    requires_generation_offload_before_reward = True
    requires_driver_model_offload_for_reward = False
    supports_reward_generation_overlap = False

    async def generate_rollout(self, request) -> _Unscored:
        inputs = request.inputs
        kwargs = request.options
        prompts = [getattr(item, "prompt", item) for item in inputs]
        return _Unscored(
            batch=_batch(prompts, int(kwargs["group_size"])),
            phases={"collect.engine_generate": 1.0},
        )

    async def evaluate_rollout(self, pendings: list[_Unscored]) -> list[RolloutBatch]:
        # Call-level timings on the first group only (RolloutCollector contract).
        pendings[0].phases["collect.reward_score"] = 0.5
        pendings[0].phases["collect.batch_build"] = 0.25
        return [pending.batch for pending in pendings]


@pytest.mark.asyncio
async def test_phase_times_accumulate_per_call() -> None:
    """Checks the out-param sums generation per group and score/build once."""
    from vrl.rollouts.stats import RolloutStats

    stats = RolloutStats()

    await prepare_training_batches(
        collector=_PhasedCollector(),
        prompts=[PromptExample(prompt="p0"), PromptExample(prompt="p1")],
        group_size=1,
        runtime_debug=False,
        policy_version=None,
        stats=stats,
    )

    assert stats.phase_seconds["collect.engine_generate"] == 2.0
    assert stats.phase_seconds["collect.reward_score"] == 0.5
    assert stats.phase_seconds["collect.batch_build"] == 0.25
    assert stats.phase_seconds["collect.wall"] >= 0.0
    assert stats.phase_seconds["collect.generation_reward_overlap"] == 0.0
    assert stats.counters == {
        "collect.group_count": 2.0,
        "collect.sample_count": 2.0,
    }


class _StreamingCollector(_DeferredCollector):
    """Deterministic one-task scoring pipeline fake."""

    def __init__(
        self,
        *,
        fail_generation: bool = False,
        fail_score: bool = False,
        fail_score_cleanup: bool = False,
    ) -> None:
        super().__init__(
            rollout_reward_handoff=False,
            trainer_reward_handoff=False,
            supports_overlap=True,
        )
        self.fail_generation = fail_generation
        self.fail_score = fail_score
        self.fail_score_cleanup = fail_score_cleanup
        self.score_started = asyncio.Event()
        self.finish_score = asyncio.Event()
        self.score_cancelled = asyncio.Event()
        self.active_scores = 0
        self.max_active_scores = 0

    async def generate_rollout(self, request) -> Any:
        inputs = request.inputs
        kwargs = request.options
        prompts = [getattr(item, "prompt", item) for item in inputs]
        name = ",".join(prompts)
        self.events.append(f"generate_start:{name}")
        if name == "p1":
            await self.score_started.wait()
            self.events.append("generation_overlapped_score:p1")
            if self.fail_generation:
                # Fail while p0's score is still in flight: the collector must
                # cancel that score, not let it finish first.
                raise RuntimeError("generation failed")
            self.finish_score.set()
        await asyncio.sleep(0)
        self.events.append(f"generate_done:{name}")
        batch = _batch(prompts, int(kwargs["group_size"]))
        self._prompt_names[id(batch)] = tuple(prompts)
        return batch

    async def evaluate_rollout(self, pendings: list[Any]) -> list[RolloutBatch]:
        names = [",".join(dict.fromkeys(self._prompt_names[id(pending)])) for pending in pendings]
        name = ";".join(names)
        self.active_scores += 1
        self.max_active_scores = max(self.max_active_scores, self.active_scores)
        self.events.append(f"score_start:{name}")
        self.score_started.set()
        try:
            if name == "p0":
                await self.finish_score.wait()
            if self.fail_score:
                raise RuntimeError("score failed")
            self.events.append(f"score_done:{name}")
            return list(pendings)
        except asyncio.CancelledError as error:
            self.score_cancelled.set()
            if self.fail_score_cleanup:
                raise RuntimeError("score cleanup failed") from error
            raise
        finally:
            self.active_scores -= 1


class _TimedCollector(_DeferredCollector):
    """Delay-proportional fake for the batched-serial vs overlap wall A/B."""

    def __init__(self, *, supports_overlap: bool, delay_s: float = 0.04) -> None:
        super().__init__(
            rollout_reward_handoff=False,
            trainer_reward_handoff=False,
            supports_overlap=supports_overlap,
        )
        self.delay_s = delay_s
        # One engine: a prefetched request queues behind the running one, as
        # the Ray dispatcher's per-engine slot does in production.
        self._engine = asyncio.Lock()

    async def generate_rollout(self, request) -> Any:
        inputs = request.inputs
        kwargs = request.options
        async with self._engine:
            await asyncio.sleep(self.delay_s)
        return await super().generate_rollout(super().request_builder.build(inputs, **kwargs))

    async def evaluate_rollout(self, pendings: list[Any]) -> list[RolloutBatch]:
        # Batched scoring keeps the same per-group work as streamed scoring, so
        # the only A/B difference is whether that work overlaps generation.
        await asyncio.sleep(self.delay_s * len(pendings))
        return await super().evaluate_rollout(pendings)


@pytest.mark.asyncio
async def test_capable_collector_overlaps_reward_with_next_generation() -> None:
    collector = _StreamingCollector()

    batches = await prepare_training_batches(
        collector=collector,
        prompts=[PromptExample(prompt="p0"), PromptExample(prompt="p1")],
        group_size=1,
        runtime_debug=False,
        policy_version=3,
    )

    assert collector.events.index("score_start:p0") < collector.events.index(
        "generation_overlapped_score:p1",
    )
    assert collector.events.index("generation_overlapped_score:p1") < collector.events.index(
        "score_done:p0",
    )
    assert collector.max_active_scores == 1
    assert collector.active_scores == 0
    assert [batch.group_ids.item() for batch in batches] == [0, 1]


@pytest.mark.asyncio
async def test_overlap_stats_support_batched_serial_vs_streaming_wall_ab() -> None:
    from vrl.rollouts.stats import RolloutStats

    prompts = [PromptExample(prompt="p0"), PromptExample(prompt="p1")]
    serial_stats = RolloutStats()
    overlap_stats = RolloutStats()

    await prepare_training_batches(
        collector=_TimedCollector(supports_overlap=False),
        prompts=prompts,
        group_size=1,
        runtime_debug=False,
        policy_version=3,
        stats=serial_stats,
    )
    await prepare_training_batches(
        collector=_TimedCollector(supports_overlap=True),
        prompts=prompts,
        group_size=1,
        runtime_debug=False,
        policy_version=3,
        stats=overlap_stats,
    )

    serial_wall = serial_stats.phase_seconds["collect.wall"]
    overlap_wall = overlap_stats.phase_seconds["collect.wall"]
    assert serial_stats.phase_seconds["collect.generation_reward_overlap"] == 0.0
    assert overlap_stats.phase_seconds["collect.generation_reward_overlap"] >= 0.02
    assert overlap_wall < serial_wall * 0.9
    assert overlap_stats.counters == {
        "collect.group_count": 2.0,
        "collect.sample_count": 2.0,
    }


@pytest.mark.parametrize(
    ("rollout_handoff", "trainer_handoff"),
    [(True, False), (False, True), (True, True)],
)
@pytest.mark.asyncio
async def test_reward_handoff_collector_keeps_generation_and_scoring_serial(
    rollout_handoff: bool,
    trainer_handoff: bool,
) -> None:
    collector = _DeferredCollector(
        rollout_reward_handoff=rollout_handoff,
        trainer_reward_handoff=trainer_handoff,
    )

    await prepare_training_batches(
        collector=collector,
        prompts=[PromptExample(prompt="p0"), PromptExample(prompt="p1")],
        group_size=1,
        runtime_debug=False,
        policy_version=None,
    )

    assert collector.events == [
        "generate:p0",
        "generate:p1",
        "evaluate_rollout:[p0;p1]",
    ]


@pytest.mark.asyncio
async def test_safe_topology_without_runtime_capability_stays_batched_and_serial() -> None:
    collector = _DeferredCollector(
        rollout_reward_handoff=False,
        trainer_reward_handoff=False,
        supports_overlap=False,
    )

    await prepare_training_batches(
        collector=collector,
        prompts=[PromptExample(prompt="p0"), PromptExample(prompt="p1")],
        group_size=1,
        runtime_debug=False,
        policy_version=None,
    )

    assert collector.events == [
        "generate:p0",
        "generate:p1",
        "evaluate_rollout:[p0;p1]",
    ]


@pytest.mark.asyncio
async def test_missing_overlap_capability_fails_loud() -> None:
    """The capability is part of the collector contract; absence is a bug, not
    a silent downgrade to batched scoring."""
    collector = _DeferredCollector(
        rollout_reward_handoff=False,
        trainer_reward_handoff=False,
    )
    del collector.supports_reward_generation_overlap

    with pytest.raises(AttributeError, match="supports_reward_generation_overlap"):
        await prepare_training_batches(
            collector=collector,
            prompts=[PromptExample(prompt="p0"), PromptExample(prompt="p1")],
            group_size=1,
            runtime_debug=False,
            policy_version=None,
        )


@pytest.mark.asyncio
async def test_score_failure_is_drained_without_task_leak() -> None:
    collector = _StreamingCollector(fail_score=True)

    with pytest.raises(RuntimeError, match="score failed"):
        await prepare_training_batches(
            collector=collector,
            prompts=[PromptExample(prompt="p0"), PromptExample(prompt="p1")],
            group_size=1,
            runtime_debug=False,
            policy_version=None,
        )

    assert collector.active_scores == 0
    assert collector.max_active_scores == 1
    assert collector.events.count("score_start:p0") == 1
    assert "score_start:p1" not in collector.events


@pytest.mark.asyncio
async def test_collection_cancellation_does_not_detach_score_task() -> None:
    collector = _StreamingCollector()
    collection = asyncio.create_task(
        prepare_training_batches(
            collector=collector,
            prompts=[PromptExample(prompt="p0")],
            group_size=1,
            runtime_debug=False,
            policy_version=None,
        ),
    )
    await collector.score_started.wait()

    collection.cancel()
    with pytest.raises(asyncio.CancelledError):
        await collection

    assert collector.score_cancelled.is_set()
    assert collector.active_scores == 0
    assert collector.max_active_scores == 1
    assert collector.events.count("score_start:p0") == 1
    assert "score_start:p1" not in collector.events


@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.asyncio
async def test_generation_failure_cancels_and_settles_inflight_score(
    cleanup_fails: bool,
) -> None:
    collector = _StreamingCollector(
        fail_generation=True,
        fail_score_cleanup=cleanup_fails,
    )

    if cleanup_fails:
        with pytest.raises(PromptCollectionCleanupError) as raised:
            await prepare_training_batches(
                collector=collector,
                prompts=[PromptExample(prompt="p0"), PromptExample(prompt="p1")],
                group_size=1,
                runtime_debug=False,
                policy_version=None,
            )
        assert str(raised.value.root_cause) == "generation failed"
        assert [str(error) for error in raised.value.cleanup_errors] == [
            "score cleanup failed",
        ]
    else:
        with pytest.raises(RuntimeError, match="generation failed"):
            await prepare_training_batches(
                collector=collector,
                prompts=[PromptExample(prompt="p0"), PromptExample(prompt="p1")],
                group_size=1,
                runtime_debug=False,
                policy_version=None,
            )

    assert collector.active_scores == 0
    assert collector.score_cancelled.is_set()


@pytest.mark.parametrize(
    ("overlap_capable", "reward_mode", "scoring", "early"),
    [
        (True, None, RewardCollectionMode.PER_GROUP_STREAMING, True),
        (False, None, RewardCollectionMode.BATCHED_SERIAL, True),
        (True, RewardCollectionMode.BATCHED_SERIAL, RewardCollectionMode.BATCHED_SERIAL, True),
        (
            True,
            RewardCollectionMode.PER_GROUP_SERIAL,
            RewardCollectionMode.PER_GROUP_SERIAL,
            False,
        ),
        (
            True,
            RewardCollectionMode.PER_GROUP_STREAMING,
            RewardCollectionMode.PER_GROUP_STREAMING,
            True,
        ),
    ],
)
def test_collection_schedule_keeps_generation_and_scoring_decisions_apart(
    overlap_capable: bool,
    reward_mode: RewardCollectionMode | None,
    scoring: RewardCollectionMode,
    early: bool,
) -> None:
    """Scoring follows capability and override; early generation is on unless
    the serial control arm asks for a fully sequential collection."""

    schedule = CollectionSchedule.resolve(
        overlap_capable=overlap_capable,
        reward_mode=reward_mode,
    )

    assert schedule.scoring is scoring
    assert schedule.submit_next_generation_early is early


@pytest.mark.parametrize(
    "reward_mode",
    [RewardCollectionMode.PER_GROUP_SERIAL, RewardCollectionMode.PER_GROUP_STREAMING],
)
def test_collection_schedule_cannot_force_per_group_scoring_without_capability(
    reward_mode: RewardCollectionMode,
) -> None:
    with pytest.raises(ValueError, match="cannot be forced on"):
        CollectionSchedule.resolve(overlap_capable=False, reward_mode=reward_mode)


async def _settle(hops: int = 20) -> None:
    """Let every ready task run; the loop under test has several awaits per group."""

    for _ in range(hops):
        await asyncio.sleep(0)


class _EngineOrderCollector(_DeferredCollector):
    """Records when each generation is submitted and when it completes."""

    def __init__(self, *, supports_overlap: bool) -> None:
        super().__init__(
            rollout_reward_handoff=False,
            trainer_reward_handoff=False,
            supports_overlap=supports_overlap,
        )
        self.release: dict[str, asyncio.Event] = {}

    async def generate_rollout(self, request) -> Any:
        prompts = [getattr(item, "prompt", item) for item in request.inputs]
        name = ",".join(prompts)
        self.events.append(f"submit:{name}")
        gate = self.release.setdefault(name, asyncio.Event())
        await gate.wait()
        self.events.append(f"generated:{name}")
        batch = _batch(prompts, int(request.options["group_size"]))
        self._prompt_names[id(batch)] = tuple(prompts)
        return batch


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("supports_overlap", "reward_mode"),
    [
        (True, None),
        (False, None),
        (True, RewardCollectionMode.BATCHED_SERIAL),
    ],
)
async def test_next_generation_is_submitted_before_the_current_one_completes(
    supports_overlap: bool,
    reward_mode: RewardCollectionMode | None,
) -> None:
    """The engine sees request N+1 while request N is still being finalized."""

    collector = _EngineOrderCollector(supports_overlap=supports_overlap)
    for name in ("p0", "p1", "p2"):
        collector.release[name] = asyncio.Event()
    collection = asyncio.create_task(
        prepare_training_batches(
            collector=collector,
            prompts=[PromptExample(prompt=f"p{i}") for i in range(3)],
            group_size=1,
            runtime_debug=False,
            policy_version=None,
            reward_mode=reward_mode,
        ),
    )
    await _settle()
    # p1 is already submitted while p0 has not completed; p2 is not.
    assert collector.events == ["submit:p0", "submit:p1"]
    collector.release["p0"].set()
    await _settle()
    # Streaming mode also starts p0's score here; its position relative to the
    # p2 submission is scheduler order, not a contract.
    engine_events = [event for event in collector.events if not event.startswith("evaluate")]
    assert engine_events == ["submit:p0", "submit:p1", "generated:p0", "submit:p2"]
    collector.release["p1"].set()
    collector.release["p2"].set()
    batches = await collection

    assert [batch.group_ids.item() for batch in batches] == [0, 1, 2]
    assert collector.events.index("submit:p2") < collector.events.index("generated:p1")


@pytest.mark.asyncio
async def test_per_group_serial_never_submits_the_next_generation_early() -> None:
    """The control arm keeps generation, scoring, and the next generation serial."""

    collector = _EngineOrderCollector(supports_overlap=True)
    for name in ("p0", "p1"):
        collector.release[name] = asyncio.Event()
        collector.release[name].set()

    await prepare_training_batches(
        collector=collector,
        prompts=[PromptExample(prompt="p0"), PromptExample(prompt="p1")],
        group_size=1,
        runtime_debug=False,
        policy_version=None,
        reward_mode=RewardCollectionMode.PER_GROUP_SERIAL,
    )

    assert collector.events == [
        "submit:p0",
        "generated:p0",
        "evaluate_rollout:[p0]",
        "submit:p1",
        "generated:p1",
        "evaluate_rollout:[p1]",
    ]


@pytest.mark.asyncio
async def test_generation_failure_cancels_the_prefetched_generation() -> None:
    """A failed group also cancels the group already submitted behind it."""

    collector = _EngineOrderCollector(supports_overlap=True)
    collector.release["p0"] = asyncio.Event()
    collector.release["p1"] = asyncio.Event()
    cancelled: list[str] = []

    async def failing_generate(request):
        name = ",".join(getattr(item, "prompt", item) for item in request.inputs)
        collector.events.append(f"submit:{name}")
        if name == "p0":
            await asyncio.sleep(0)
            raise RuntimeError("generation failed")
        try:
            await collector.release[name].wait()
        except asyncio.CancelledError:
            cancelled.append(name)
            raise
        raise AssertionError("p1 must be cancelled, never completed")

    collector.generate_rollout = failing_generate  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="generation failed"):
        await prepare_training_batches(
            collector=collector,
            prompts=[PromptExample(prompt="p0"), PromptExample(prompt="p1")],
            group_size=1,
            runtime_debug=False,
            policy_version=None,
        )

    assert collector.events == ["submit:p0", "submit:p1"]
    assert cancelled == ["p1"]


@pytest.mark.asyncio
async def test_per_group_serial_scores_each_group_before_the_next_generation() -> None:
    """Checks the acceptance control arm keeps per-group calls without overlap."""
    collector = _DeferredCollector(
        rollout_reward_handoff=False,
        supports_overlap=True,
    )
    prompts = [PromptExample(prompt=f"p{i}") for i in range(3)]

    batches = await prepare_training_batches(
        collector=collector,
        prompts=prompts,
        group_size=1,
        runtime_debug=False,
        policy_version=5,
        reward_mode=RewardCollectionMode.PER_GROUP_SERIAL,
    )

    # Per-group call granularity (same as streaming), strictly interleaved.
    assert collector.events == [
        "generate:p0",
        "evaluate_rollout:[p0]",
        "generate:p1",
        "evaluate_rollout:[p1]",
        "generate:p2",
        "evaluate_rollout:[p2]",
    ]
    assert [batch.group_ids.unique().tolist() for batch in batches] == [[0], [1], [2]]


@pytest.mark.parametrize(
    "mode",
    [
        RewardCollectionMode.PER_GROUP_SERIAL,
        RewardCollectionMode.PER_GROUP_STREAMING,
    ],
)
@pytest.mark.asyncio
async def test_forcing_per_group_mode_without_capability_raises(
    mode: RewardCollectionMode,
) -> None:
    """Checks an acceptance override cannot grant per-group execution."""
    collector = _DeferredCollector(supports_overlap=False)

    with pytest.raises(ValueError, match="cannot be forced on"):
        await prepare_training_batches(
            collector=collector,
            prompts=[PromptExample(prompt="p0")],
            group_size=1,
            runtime_debug=False,
            policy_version=None,
            reward_mode=mode,
        )

    assert collector.events == []


@pytest.mark.asyncio
async def test_capable_collector_can_be_restricted_to_the_batched_serial_arm() -> None:
    """Checks arm A stays reachable on a collector that could stream."""
    collector = _DeferredCollector(
        rollout_reward_handoff=False,
        supports_overlap=True,
    )

    await prepare_training_batches(
        collector=collector,
        prompts=[PromptExample(prompt="p0"), PromptExample(prompt="p1")],
        group_size=1,
        runtime_debug=False,
        policy_version=None,
        reward_mode=RewardCollectionMode.BATCHED_SERIAL,
    )

    assert collector.events == [
        "generate:p0",
        "generate:p1",
        "evaluate_rollout:[p0;p1]",
    ]


@pytest.mark.asyncio
async def test_three_acceptance_arms_isolate_overlap_from_per_group_call_tax() -> None:
    """Checks A/B/C produce identical batches and only C reports overlap.

    This is the in-repo shape of the hardware acceptance in
    ``docs/sprints/done/SPRINT_reward_service.md``: B exists so a C-vs-A wall
    win cannot be attributed to overlap without first pricing the per-group
    call granularity that C also introduces.
    """
    from vrl.rollouts.stats import RolloutStats

    prompts = [PromptExample(prompt="p0"), PromptExample(prompt="p1")]
    arms = {
        RewardCollectionMode.BATCHED_SERIAL: RolloutStats(),
        RewardCollectionMode.PER_GROUP_SERIAL: RolloutStats(),
        RewardCollectionMode.PER_GROUP_STREAMING: RolloutStats(),
    }
    rewards: dict[RewardCollectionMode, list[list[float]]] = {}

    for mode, stats in arms.items():
        batches = await prepare_training_batches(
            # Every arm runs on a capable collector so the only difference is
            # the requested mode, not the collector fake.
            collector=_TimedCollector(supports_overlap=True),
            prompts=prompts,
            group_size=1,
            runtime_debug=False,
            policy_version=3,
            stats=stats,
            reward_mode=mode,
        )
        rewards[mode] = [batch.rewards.tolist() for batch in batches]

    # Correctness: the arms are pure scheduling variants, so results match.
    assert (
        rewards[RewardCollectionMode.BATCHED_SERIAL]
        == rewards[RewardCollectionMode.PER_GROUP_SERIAL]
        == rewards[RewardCollectionMode.PER_GROUP_STREAMING]
    )

    overlap_key = "collect.generation_reward_overlap"
    assert arms[RewardCollectionMode.BATCHED_SERIAL].phase_seconds[overlap_key] == 0.0
    assert arms[RewardCollectionMode.PER_GROUP_SERIAL].phase_seconds[overlap_key] == 0.0
    assert arms[RewardCollectionMode.PER_GROUP_STREAMING].phase_seconds[overlap_key] >= 0.02

    # Only the streaming arm may shorten the collection wall.
    serial_wall = arms[RewardCollectionMode.BATCHED_SERIAL].phase_seconds["collect.wall"]
    control_wall = arms[RewardCollectionMode.PER_GROUP_SERIAL].phase_seconds["collect.wall"]
    streaming_wall = arms[RewardCollectionMode.PER_GROUP_STREAMING].phase_seconds["collect.wall"]
    assert streaming_wall < serial_wall * 0.9
    assert streaming_wall < control_wall * 0.9


@pytest.mark.asyncio
async def test_generation_handoff_is_demand_driven_and_never_scores() -> None:

    collector = _DeferredCollector()
    groups = collector.build_generation_requests(
        prompts=["plain", PromptExample(prompt="example"), "later"],
        group_size=2,
        runtime_debug=False,
        policy_version=7,
    )
    first_request, first_indices = next(groups)
    assert collector.events == []
    first = RolloutGenerationResult(
        await collector.generate_rollout(first_request), first_indices, 0.0, 0.0
    )
    assert first.prompt_indices == [0]
    assert first.completed_at >= first.started_at
    assert collector.events == ["generate:plain"]
    second_request, second_indices = next(groups)
    second = RolloutGenerationResult(
        await collector.generate_rollout(second_request), second_indices, 0.0, 0.0
    )
    assert second.prompt_indices == [1]
    assert collector.events == ["generate:plain", "generate:example"]
    groups.close()
    # Closing admission must not generate the remaining prompt or start reward.
    assert collector.events == ["generate:plain", "generate:example"]
    scored = collector.assemble_training_batches(
        await collector.evaluate_rollout([first.unscored, second.unscored])
    )
    assert len(scored) == 2
    assert all(batch.rewards.shape == (2,) for batch in scored)
