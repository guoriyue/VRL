"""Types for RL rollout scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vrl.rollouts.batch import RolloutBatch


class RolloutScheduleMode(str, Enum):
    """Supported RL rollout schedule modes."""

    STRICT_ON_POLICY = "strict_on_policy"
    ONE_BATCH_OVERLAP = "one_batch_overlap"



@dataclass(slots=True)
class RolloutScheduleState:
    """Mutable state owned by one rollout schedule."""

    rollout_id: int = 0
    current_policy_version: int | None = None
    initialized: bool = False
    pending_rollout: Any | None = None


@dataclass(slots=True)
class RolloutIteration:
    """One rollout batch set handed from the schedule to the trainer."""

    rollout_id: int
    policy_version: int | None
    mode: RolloutScheduleMode
    batches: list[RolloutBatch]
    prompt_count: int
    sample_count: int
    phase_times: dict[str, float] = field(default_factory=dict)
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
    phase_times: dict[str, float] | None = None,
) -> RolloutIteration:
    """Attach schedule metadata to collected batches and return an iteration."""

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
    for batch in batches:
        batch.context = {**dict(batch.context), **metadata}
    return RolloutIteration(
        rollout_id=int(rollout_id),
        policy_version=None if policy_version is None else int(policy_version),
        mode=mode,
        batches=batches,
        prompt_count=int(prompt_count),
        sample_count=sample_count,
        phase_times=dict(phase_times or {}),
        metadata=metadata,
    )


__all__ = [
    "RolloutIteration",
    "RolloutScheduleMode",
    "RolloutScheduleState",
    "build_rollout_iteration",
]
