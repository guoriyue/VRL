"""Types for RL rollout scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.stats import RolloutStats


# Keep the exported modes' historical ``str(member)`` representation; metadata
# and config boundaries serialize ``.value`` explicitly.
class RolloutScheduleMode(str, Enum):  # noqa: UP042
    """Supported RL rollout schedule modes."""

    STRICT_ON_POLICY = "strict_on_policy"
    CONTINUOUS = "continuous"


class RewardCollectionMode(str, Enum):  # noqa: UP042
    """How prompt collection interleaves group generation and reward scoring.

    Production picks between ``BATCHED_SERIAL`` and ``PER_GROUP_STREAMING`` from
    the collector's overlap capability alone. ``PER_GROUP_SERIAL`` is the
    acceptance control arm required by ``docs/sprints/done/SPRINT_reward_service.md``:
    it moves scoring to per-group granularity *without* overlap, so the
    per-group call/transport tax can be measured separately from the overlap
    gain. Without it a streaming benchmark changes two variables at once and
    cannot attribute its own result.
    """

    BATCHED_SERIAL = "batched_serial"
    PER_GROUP_SERIAL = "per_group_serial"
    PER_GROUP_STREAMING = "per_group_streaming"


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
    stats: RolloutStats = field(default_factory=RolloutStats)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rollout_id < 0:
            raise ValueError("RolloutIteration.rollout_id must be >= 0")
        if self.prompt_count < 0:
            raise ValueError("RolloutIteration.prompt_count must be >= 0")

    @property
    def sample_count(self) -> int:
        """Derive the current sample total from the owned rollout batches."""

        return sum(int(batch.rewards.shape[0]) for batch in self.batches)

    def annotate_batch_context(self) -> RolloutIteration:
        """Copy this iteration's schedule metadata onto each batch's ``context``.

        Returns ``self`` so a schedule can build and annotate in one expression.
        """

        schedule_context = {
            **self.metadata,
            "rollout_id": self.rollout_id,
            "rollout_policy_version": self.policy_version,
            "schedule_mode": self.mode.value,
            "prompt_count": self.prompt_count,
            "sample_count": self.sample_count,
        }
        for batch in self.batches:
            batch.context = {**dict(batch.context), **schedule_context}
        return self


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
    ``RolloutIteration.annotate_batch_context`` explicitly.
    """

    return RolloutIteration(
        rollout_id=int(rollout_id),
        policy_version=None if policy_version is None else int(policy_version),
        mode=mode,
        batches=batches,
        prompt_count=int(prompt_count),
        stats=stats if stats is not None else RolloutStats(),
    )


__all__ = [
    "RewardCollectionMode",
    "RolloutIteration",
    "RolloutScheduleMode",
    "RolloutScheduleState",
    "build_rollout_iteration",
]
