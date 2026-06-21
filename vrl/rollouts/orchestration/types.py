"""Types for RL rollout scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vrl.rollouts.batch import RolloutBatch
from vrl.utils.stats import RolloutStats


class RolloutScheduleMode(str, Enum):
    """Supported RL rollout schedule modes."""

    STRICT_ON_POLICY = "strict_on_policy"
    CONTINUOUS = "continuous"



@dataclass(slots=True)
class RolloutScheduleState:
    """Mutable state owned by one rollout schedule."""

    rollout_id: int = 0


@dataclass(slots=True)
class RolloutIteration:
    """One rollout batch set handed from the schedule to the trainer."""

    rollout_id: int
    policy_version: int | None
    mode: RolloutScheduleMode
    batches: list[RolloutBatch]
    prompt_count: int
    sample_count: int
    stats: RolloutStats = field(default_factory=RolloutStats)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rollout_id < 0:
            raise ValueError("RolloutIteration.rollout_id must be >= 0")
        if self.prompt_count < 0:
            raise ValueError("RolloutIteration.prompt_count must be >= 0")
        if self.sample_count < 0:
            raise ValueError("RolloutIteration.sample_count must be >= 0")


def build_rollout_iteration(
    *,
    rollout_id: int,
    policy_version: int | None,
    mode: RolloutScheduleMode,
    batches: list[RolloutBatch],
    prompt_count: int,
    stats: RolloutStats | None = None,
) -> RolloutIteration:
    """Build a rollout iteration from collected batches.

    Pure: does not mutate the input batches. Callers that need the schedule
    metadata copied onto each batch's ``context`` should call
    ``annotate_batch_context`` explicitly.
    """

    sample_count = sum(int(batch.rewards.shape[0]) for batch in batches)
    metadata: dict[str, Any] = {
        "rollout_id": int(rollout_id),
        "rollout_policy_version": (
            None if policy_version is None else int(policy_version)
        ),
        "schedule_mode": mode.value,
        "prompt_count": int(prompt_count),
        "sample_count": int(sample_count),
    }
    return RolloutIteration(
        rollout_id=int(rollout_id),
        policy_version=None if policy_version is None else int(policy_version),
        mode=mode,
        batches=batches,
        prompt_count=int(prompt_count),
        sample_count=sample_count,
        stats=stats if stats is not None else RolloutStats(),
        metadata=metadata,
    )


def annotate_batch_context(iteration: RolloutIteration) -> RolloutIteration:
    """Copy the iteration's schedule metadata onto each batch's ``context``."""

    for batch in iteration.batches:
        batch.context = {**dict(batch.context), **iteration.metadata}
    return iteration


__all__ = [
    "RolloutIteration",
    "RolloutScheduleMode",
    "RolloutScheduleState",
    "annotate_batch_context",
    "build_rollout_iteration",
]
