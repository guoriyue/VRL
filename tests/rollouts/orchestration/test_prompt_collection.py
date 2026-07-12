"""Topology-safe deferred and streamed reward behavior of prompt collection."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.orchestration.prompt_collection import (
    PromptCollectionCleanupError,
    collect_prompt_batches,
)
from vrl.trainers.data import PromptExample


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
        dones=torch.ones(batch_size, dtype=torch.bool),
        group_ids=group_ids,
        prompts=[prompt for prompt in prompts for _ in range(group_size)],
    )


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
        self.requires_runtime_offload_before_reward = rollout_reward_handoff
        self.requires_driver_model_offload_for_reward = trainer_reward_handoff
        self.supports_reward_generation_overlap = supports_overlap

    async def collect_unscored(self, inputs: list[Any], **kwargs: Any) -> Any:
        prompts = [getattr(item, "prompt", item) for item in inputs]
        self.events.append(f"generate:{','.join(prompts)}")
        return _batch(prompts, int(kwargs["group_size"]))

    async def score_rollouts(self, pendings: list[Any]) -> list[RolloutBatch]:
        names = [",".join(dict.fromkeys(pending.prompts)) for pending in pendings]
        self.events.append(f"score_rollouts:[{';'.join(names)}]")
        return list(pendings)


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
        assert batch.prompts == [f"p{prompt_idx}"] * 2


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
    assert [batch.prompts for batch in batches] == [["s0"], ["e1"], ["s2"]]


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

    assert stats.phase_seconds == {
        "collect.engine_generate": 2.0,
        "collect.reward_score": 0.5,
        "collect.batch_build": 0.25,
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
        return _batch(prompts, int(kwargs["group_size"]))

    async def score_rollouts(self, pendings: list[Any]) -> list[RolloutBatch]:
        names = [",".join(dict.fromkeys(pending.prompts)) for pending in pendings]
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
    assert [batch.prompts for batch in batches] == [["p0"], ["p1"]]
    assert [batch.group_ids.item() for batch in batches] == [0, 1]


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
