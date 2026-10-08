"""Staleness policy for continuous rollout batches.

Staleness is ``trainer_current_version - rollout_item_version``. A negative
value means the item was produced by a policy newer than the trainer, which
can only happen through a bug, so callers should fail fast on it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StalenessPolicy:
    """Bound on how far an item's policy version may trail the trainer.

    A zero window remains useful for isolated producer/consumer invariant tests,
    but production ``ContinuousRolloutConfig`` rejects it: zero-staleness runs
    use the strict-on-policy schedule.
    """

    max_stale_policy_versions: int = 0

    def staleness(self, item_version: int, current_version: int) -> int:
        """Versions the item trails the trainer by."""

        return current_version - item_version

    def too_stale(self, item_version: int, current_version: int) -> bool:
        return self.staleness(item_version, current_version) > self.max_stale_policy_versions


__all__ = ["StalenessPolicy"]
