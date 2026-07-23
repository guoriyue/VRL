"""Derived rollout-iteration summaries and batch-context projection."""

from __future__ import annotations

import pytest
import torch

from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.orchestration.types import (
    RolloutScheduleMode,
    annotate_batch_context,
    build_rollout_iteration,
)


def _batch(sample_count: int, *, context: dict | None = None) -> RolloutBatch:
    return RolloutBatch(
        rewards=torch.zeros(sample_count),
        group_ids=torch.zeros(sample_count, dtype=torch.long),
        context=dict(context or {}),
    )


def test_empty_iteration_derives_zero_samples_without_summary_metadata() -> None:
    iteration = build_rollout_iteration(
        rollout_id=0,
        policy_version=None,
        mode=RolloutScheduleMode.STRICT_ON_POLICY,
        batches=[],
        prompt_count=0,
    )

    assert iteration.sample_count == 0
    assert iteration.metadata == {}


def test_nonempty_iteration_derives_samples_from_current_batch_rows() -> None:
    iteration = build_rollout_iteration(
        rollout_id=1,
        policy_version=4,
        mode=RolloutScheduleMode.CONTINUOUS,
        batches=[_batch(2)],
        prompt_count=1,
    )

    assert iteration.sample_count == 2
    iteration.batches.append(_batch(3))
    assert iteration.sample_count == 5
    with pytest.raises(AttributeError):
        iteration.sample_count = 99


def test_context_projection_typed_values_override_reserved_collisions() -> None:
    batch = _batch(
        2,
        context={
            "rollout_id": -1,
            "rollout_policy_version": -1,
            "schedule_mode": "forged-batch-mode",
            "prompt_count": -1,
            "sample_count": -1,
        },
    )
    iteration = build_rollout_iteration(
        rollout_id=7,
        policy_version=3,
        mode=RolloutScheduleMode.CONTINUOUS,
        batches=[batch],
        prompt_count=1,
    )
    iteration.metadata.update(
        {
            "rollout_id": 99,
            "rollout_policy_version": 99,
            "schedule_mode": "forged-metadata-mode",
            "prompt_count": 99,
            "sample_count": 99,
        },
    )

    annotate_batch_context(iteration)

    assert batch.context["rollout_id"] == 7
    assert batch.context["rollout_policy_version"] == 3
    assert batch.context["schedule_mode"] == "continuous"
    assert batch.context["prompt_count"] == 1
    assert batch.context["sample_count"] == 2


def test_context_projection_preserves_non_reserved_metadata() -> None:
    batch = _batch(1, context={"batch_only": "kept", "continuous_item_age_s": -1.0})
    iteration = build_rollout_iteration(
        rollout_id=2,
        policy_version=1,
        mode=RolloutScheduleMode.CONTINUOUS,
        batches=[batch],
        prompt_count=1,
    )
    iteration.metadata.update(
        {
            "consume_policy_version": 2,
            "stale_policy_versions": 1,
            "continuous_item_age_s": 0.25,
        },
    )

    annotate_batch_context(iteration)

    assert iteration.metadata == {
        "consume_policy_version": 2,
        "stale_policy_versions": 1,
        "continuous_item_age_s": 0.25,
    }
    assert batch.context["batch_only"] == "kept"
    assert batch.context["consume_policy_version"] == 2
    assert batch.context["stale_policy_versions"] == 1
    assert batch.context["continuous_item_age_s"] == 0.25
