"""Types for RL rollout scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from vrl.rollouts.batch import RolloutBatch


# Keep the exported modes' historical ``str(member)`` representation; config
# boundaries serialize ``.value`` explicitly.
class RolloutScheduleMode(str, Enum):  # noqa: UP042
    """Supported RL rollout schedule modes."""

    STRICT_ON_POLICY = "strict_on_policy"
    CONTINUOUS = "continuous"


@dataclass(slots=True)
class RolloutIteration:
    """One rollout batch set handed from the schedule to the trainer."""

    batches: list[RolloutBatch]


__all__ = [
    "RolloutIteration",
    "RolloutScheduleMode",
]
