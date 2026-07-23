"""Topology-safe deferred and streamed reward behavior of prompt collection."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.trajectory import TrajectorySignalBuilder
from vrl.rollouts.orchestration.prompt_collection import (
    PromptCollectionCleanupError,
    collect_prompt_batches,
)
from vrl.rollouts.orchestration.types import RewardCollectionMode
from vrl.trainers.data import PromptExample
from vrl.trajectory import build_ar_discrete_trajectory


def _batch(prompts: list[str], group_size: int) -> RolloutBatch:
    batch_size = len(prompts) * group_size
    group_ids = torch.tensor(
        [prompt_idx for prompt_idx in range(len(prompts)) for _ in range(group_size)],
        dtype=torch.long,
    )
    return RolloutBatch(
        observations=torch.zeros(batch_size, 1, 1),
        actions=torch.zeros(batch_size, 1, 1),
        rewards=torch.arange(batch_size, dtype=torch.float32),
        group_ids=group_ids,
    )


def _batch_with_trajectory(prompts: list[str], group_size: int) -> RolloutBatch:
    """Build independent batch/trajectory group tensors for remap regressions."""

    batch = _batch(prompts, group_size)
    sample_rows = [
        GenerationSampleRow(
            prompt_index=prompt_index,
            sample_index=sample_index,
            prompt=prompt,
            group_id=f"group-{prompt_index}",
            sample_id=f"sample-{prompt_index}-{sample_index}",
            trajectory_id=f"trajectory-{prompt_index}-{sample_index}",
            seed=None,
        )
        for prompt_index, prompt in enumerate(prompts)
        for sample_index in range(group_size)
    ]
    request = GenerationRequest(
        request_id="remap-request",
        family="fake",
        task="ar_t2i",
        inputs=prompts,
        samples_per_prompt=group_size,
    )
    batch_size = len(sample_rows)
    batch.trajectory = build_ar_discrete_trajectory(
        request=request,
        sample_rows=sample_rows,
        token_ids=torch.zeros(batch_size, 1, dtype=torch.long),
        token_log_probs=torch.zeros(batch_size, 1),
        token_mask=torch.ones(batch_size, 1),
        prompt_input_ids=torch.zeros(batch_size, 1, dtype=torch.long),
        prompt_attention_mask=torch.ones(batch_size, 1, dtype=torch.long),
        uncond_input_ids=torch.zeros(batch_size, 1, dtype=torch.long),
        uncond_attention_mask=torch.ones(batch_size, 1, dtype=torch.long),
        context={},
    )
    assert batch.group_ids.data_ptr() != batch.trajectory.group_ids.data_ptr()
    return batch


class _DeferredCollector:
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
        self.requires_runtime_offload_before_reward = rollout_reward_handoff
        self.requires_driver_model_offload_for_reward = trainer_reward_handoff
        self.supports_reward_generation_overlap = supports_overlap

    async def collect_unscored(self, inputs: list[Any], **kwargs: Any) -> Any:
        prompts = [getattr(item, "prompt", item) for item in inputs]
        self.events.append(f"generate:{','.join(prompts)}")
        batch = _batch(prompts, int(kwargs["group_size"]))
        self._prompt_names[id(batch)] = tuple(prompts)
        return batch

    async def score_rollouts(self, pendings: list[Any]) -> list[RolloutBatch]:
        names = [",".join(dict.fromkeys(self._prompt_names[id(pending)])) for pending in pendings]
        self.events.append(f"score_rollouts:[{';'.join(names)}]")
        return list(pendings)


class _TrajectoryDeferredCollector(_DeferredCollector):
    """Deferred collector whose trainer and trajectory grouping never alias."""

    async def collect_unscored(self, inputs: list[Any], **kwargs: Any) -> RolloutBatch:
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

    batches = await collect_prompt_batches(
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
        "score_rollouts:[p0;p1;p2]",
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

    batches = await collect_prompt_batches(
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
        "score_rollouts:[s0;e1;s2]",
    ]
    assert [batch.group_ids.unique().tolist() for batch in batches] == [[0], [1], [2]]


@pytest.mark.asyncio
async def test_prompt_example_scalar_remap_updates_batch_and_trajectory_group_ids() -> None:
    """Checks scalar remaps update the evaluator's non-alias trajectory tensor."""

    batches = await collect_prompt_batches(
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
        assert torch.equal(batch.trajectory.group_ids, expected)
        assert torch.equal(TrajectorySignalBuilder(batch).group_ids, expected)


@pytest.mark.asyncio
async def test_plain_string_list_remap_updates_batch_and_trajectory_group_ids() -> None:
    """Checks list remaps retain the same batch/trajectory synchronization."""

    batches = await collect_prompt_batches(
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


class _PhasedCollector:
    """Collector fake exposing per-call phase timings like RolloutCollector."""

    requires_runtime_offload_before_reward = True
    requires_driver_model_offload_for_reward = False
    supports_reward_generation_overlap = False

    async def collect_unscored(self, inputs: list[Any], **kwargs: Any) -> _Unscored:
        prompts = [getattr(item, "prompt", item) for item in inputs]
        return _Unscored(
            batch=_batch(prompts, int(kwargs["group_size"])),
            phases={"collect.engine_generate": 1.0},
        )

    async def score_rollouts(self, pendings: list[_Unscored]) -> list[RolloutBatch]:
        # Call-level timings on the first group only (RolloutCollector contract).
        pendings[0].phases["collect.reward_score"] = 0.5
        pendings[0].phases["collect.batch_build"] = 0.25
        return [pending.batch for pending in pendings]


@pytest.mark.asyncio
async def test_phase_times_accumulate_per_call() -> None:
    """Checks the out-param sums generation per group and score/build once."""
    from vrl.utils.stats import RolloutStats

    stats = RolloutStats()

    await collect_prompt_batches(
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

    async def collect_unscored(self, inputs: list[Any], **kwargs: Any) -> Any:
        prompts = [getattr(item, "prompt", item) for item in inputs]
        name = ",".join(prompts)
        self.events.append(f"generate_start:{name}")
        if name == "p1":
            await self.score_started.wait()
            self.events.append("generation_overlapped_score:p1")
            self.finish_score.set()
            if self.fail_generation:
                raise RuntimeError("generation failed")
        await asyncio.sleep(0)
        self.events.append(f"generate_done:{name}")
        batch = _batch(prompts, int(kwargs["group_size"]))
        self._prompt_names[id(batch)] = tuple(prompts)
        return batch

    async def score_rollouts(self, pendings: list[Any]) -> list[RolloutBatch]:
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

    async def collect_unscored(self, inputs: list[Any], **kwargs: Any) -> Any:
        await asyncio.sleep(self.delay_s)
        return await super().collect_unscored(inputs, **kwargs)

    async def score_rollouts(self, pendings: list[Any]) -> list[RolloutBatch]:
        # Batched scoring keeps the same per-group work as streamed scoring, so
        # the only A/B difference is whether that work overlaps generation.
        await asyncio.sleep(self.delay_s * len(pendings))
        return await super().score_rollouts(pendings)


@pytest.mark.asyncio
async def test_capable_collector_overlaps_reward_with_next_generation() -> None:
    collector = _StreamingCollector()

    batches = await collect_prompt_batches(
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
    from vrl.utils.stats import RolloutStats

    prompts = [PromptExample(prompt="p0"), PromptExample(prompt="p1")]
    serial_stats = RolloutStats()
    overlap_stats = RolloutStats()

    await collect_prompt_batches(
        collector=_TimedCollector(supports_overlap=False),
        prompts=prompts,
        group_size=1,
        runtime_debug=False,
        policy_version=3,
        stats=serial_stats,
    )
    await collect_prompt_batches(
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

    await collect_prompt_batches(
        collector=collector,
        prompts=[PromptExample(prompt="p0"), PromptExample(prompt="p1")],
        group_size=1,
        runtime_debug=False,
        policy_version=None,
    )

    assert collector.events == [
        "generate:p0",
        "generate:p1",
        "score_rollouts:[p0;p1]",
    ]


@pytest.mark.asyncio
async def test_safe_topology_without_runtime_capability_stays_batched_and_serial() -> None:
    collector = _DeferredCollector(
        rollout_reward_handoff=False,
        trainer_reward_handoff=False,
        supports_overlap=False,
    )

    await collect_prompt_batches(
        collector=collector,
        prompts=[PromptExample(prompt="p0"), PromptExample(prompt="p1")],
        group_size=1,
        runtime_debug=False,
        policy_version=None,
    )

    assert collector.events == [
        "generate:p0",
        "generate:p1",
        "score_rollouts:[p0;p1]",
    ]


@pytest.mark.asyncio
async def test_missing_overlap_capability_fails_closed_to_batched_scoring() -> None:
    collector = _DeferredCollector(
        rollout_reward_handoff=False,
        trainer_reward_handoff=False,
    )
    del collector.supports_reward_generation_overlap

    await collect_prompt_batches(
        collector=collector,
        prompts=[PromptExample(prompt="p0"), PromptExample(prompt="p1")],
        group_size=1,
        runtime_debug=False,
        policy_version=None,
    )

    assert collector.events == [
        "generate:p0",
        "generate:p1",
        "score_rollouts:[p0;p1]",
    ]


@pytest.mark.asyncio
async def test_score_failure_is_drained_without_task_leak() -> None:
    collector = _StreamingCollector(fail_score=True)

    with pytest.raises(RuntimeError, match="score failed"):
        await collect_prompt_batches(
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
        collect_prompt_batches(
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
            await collect_prompt_batches(
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
            await collect_prompt_batches(
                collector=collector,
                prompts=[PromptExample(prompt="p0"), PromptExample(prompt="p1")],
                group_size=1,
                runtime_debug=False,
                policy_version=None,
            )

    assert collector.active_scores == 0
    assert collector.score_cancelled.is_set()


@pytest.mark.asyncio
async def test_per_group_serial_scores_each_group_before_the_next_generation() -> None:
    """Checks the acceptance control arm keeps per-group calls without overlap."""
    collector = _DeferredCollector(
        rollout_reward_handoff=False,
        supports_overlap=True,
    )
    prompts = [PromptExample(prompt=f"p{i}") for i in range(3)]

    batches = await collect_prompt_batches(
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
        "score_rollouts:[p0]",
        "generate:p1",
        "score_rollouts:[p1]",
        "generate:p2",
        "score_rollouts:[p2]",
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
        await collect_prompt_batches(
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

    await collect_prompt_batches(
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
        "score_rollouts:[p0;p1]",
    ]


@pytest.mark.asyncio
async def test_three_acceptance_arms_isolate_overlap_from_per_group_call_tax() -> None:
    """Checks A/B/C produce identical batches and only C reports overlap.

    This is the in-repo shape of the hardware acceptance in
    ``docs/sprints/done/SPRINT_reward_service.md``: B exists so a C-vs-A wall
    win cannot be attributed to overlap without first pricing the per-group
    call granularity that C also introduces.
    """
    from vrl.utils.stats import RolloutStats

    prompts = [PromptExample(prompt="p0"), PromptExample(prompt="p1")]
    arms = {
        RewardCollectionMode.BATCHED_SERIAL: RolloutStats(),
        RewardCollectionMode.PER_GROUP_SERIAL: RolloutStats(),
        RewardCollectionMode.PER_GROUP_STREAMING: RolloutStats(),
    }
    rewards: dict[RewardCollectionMode, list[list[float]]] = {}

    for mode, stats in arms.items():
        batches = await collect_prompt_batches(
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
