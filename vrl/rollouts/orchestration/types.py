"""Types for RL rollout scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.stats import RolloutStats


# Keep the exported modes' historical ``str(member)`` representation; config
# boundaries serialize ``.value`` explicitly.
class RolloutScheduleMode(str, Enum):  # noqa: UP042
    """Supported RL rollout schedule modes."""

    STRICT_ON_POLICY = "strict_on_policy"
    CONTINUOUS = "continuous"


class RewardCollectionMode(str, Enum):  # noqa: UP042
    """How prompt collection interleaves group generation and reward scoring.

    Strict collection picks between ``BATCHED_SERIAL`` and
    ``PER_GROUP_STREAMING`` from the collector's overlap capability. Continuous
    collection already dispatches one task per group, so it uses
    ``BATCHED_SERIAL`` inside each task instead of creating an inner task that
    cannot add overlap. ``PER_GROUP_SERIAL`` is the acceptance control arm
    required by ``docs/sprints/done/SPRINT_reward_service.md``: it moves strict
    scoring to per-group granularity *without* overlap, so the per-group
    call/transport tax can be measured separately from the overlap gain.
    """

    BATCHED_SERIAL = "batched_serial"
    PER_GROUP_SERIAL = "per_group_serial"
    PER_GROUP_STREAMING = "per_group_streaming"


@dataclass(slots=True)
class RolloutIteration:
    """One rollout batch set handed from the schedule to the trainer."""

    batches: list[RolloutBatch]
    stats: RolloutStats = field(default_factory=RolloutStats)


__all__ = [
    "RewardCollectionMode",
    "RolloutIteration",
    "RolloutScheduleMode",
]
