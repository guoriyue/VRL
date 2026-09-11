"""Shared helpers for rollout collector tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.collector.core import GeneratedPromptGroup, RolloutCollector
from vrl.rollouts.stats import RolloutStats


async def collect_scored(
    collector: RolloutCollector,
    inputs: list[Any],
    *,
    group_size: int,
    metadata: Mapping[str, Any] | None = None,
    request_overrides: Mapping[str, Any] | None = None,
    runtime_debug: bool = False,
    policy_version: int | None = None,
) -> RolloutBatch:
    """Collect one group and score it in a single call.

    Tests needing a single unsplit batch compose the low-level phases here.
    The public prompt-group API additionally restores prompt IDs and splits batches.
    """

    unscored = await collector.collect_unscored(
        inputs,
        group_size=group_size,
        metadata=metadata,
        request_overrides=request_overrides,
        runtime_debug=runtime_debug,
        policy_version=policy_version,
    )
    return (await collector.score_rollouts([unscored]))[0]


class PromptCollectionFake:
    """Run production prompt collection over fake generation and reward operations."""

    generate_prompt_groups = RolloutCollector.generate_prompt_groups
    collect_prompt_groups = RolloutCollector.collect_prompt_groups
    def finish_scored_prompt_groups(
        self,
        generated_groups: list[GeneratedPromptGroup],
        batches: list[RolloutBatch],
        stats: RolloutStats,
    ) -> list[RolloutBatch]:
        # Scheduling fakes may return a batch directly, without building a
        # generation request/output. Supply its omitted timing fields here;
        # production UnscoredRollout always owns both dictionaries.
        groups = [
            replace(
                group,
                unscored=SimpleNamespace(
                    phases=getattr(group.unscored, "phases", {}),
                    reward_timing_ms=getattr(group.unscored, "reward_timing_ms", {}),
                ),
            )
            for group in generated_groups
        ]
        return RolloutCollector.finish_scored_prompt_groups(self, groups, batches, stats)
