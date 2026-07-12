"""Deferred-scoring behavior of collect_prompt_batches.

The property these tests guard: all prompt groups are generated before any
group is scored, all groups score through ONE score_rollouts call, and
group-id remapping survives the deferral. On shared single-GPU runs this is
what keeps the rollout release and the reward actor cold start at once per
epoch instead of once per prompt group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.orchestration.prompt_collection import collect_prompt_batches
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

    def __init__(self) -> None:
        self.events: list[str] = []

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
