"""Shared helpers for rollout collector tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.collector.core import RolloutCollector


async def collect_scored(
    collector: RolloutCollector,
    inputs: list[Any],
    *,
    group_size: int,
    metadata: Mapping[str, Any] | None = None,
    request_overrides: Mapping[str, Any] | None = None,
    seed: int | None = None,
    runtime_debug: bool = False,
    policy_version: int | None = None,
) -> RolloutBatch:
    """Collect one group and score it in a single call.

    Production always splits the two phases (prompt_collection.py overlaps
    scoring with the next generation), so the collector exposes no combined
    entry point; tests that only need one scored batch compose it here.
    """

    unscored = await collector.collect_unscored(
        inputs,
        group_size=group_size,
        metadata=metadata,
        request_overrides=request_overrides,
        seed=seed,
        runtime_debug=runtime_debug,
        policy_version=policy_version,
    )
    return (await collector.score_rollouts([unscored]))[0]
