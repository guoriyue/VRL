"""Staleness policy for the continuous rollout queue.

Staleness is ``trainer_current_version - rollout_item_version``. A negative
value means the item was produced by a policy newer than the trainer, which
can only happen through a bug, so callers should fail fast on it.
"""

from __future__ import annotations

from dataclasses import dataclass

from vrl.utils.validation import require_int


@dataclass(slots=True)
class StalenessPolicy:
    """Bound on how far an item's policy version may trail the trainer.

    A zero window remains useful for isolated producer/consumer invariant tests,
    but production ``ContinuousRolloutConfig`` rejects it: zero-staleness runs
    use the strict-on-policy schedule.
    """

    max_stale_policy_versions: int = 0

    def __post_init__(self) -> None:
        require_int(self.max_stale_policy_versions, path="max_stale_policy_versions", minimum=0)

    def staleness(
        self,
        item_version: int | None,
        current_version: int | None,
    ) -> int | None:
        """Versions behind the trainer, or ``None`` when versions are absent."""

        if item_version is not None:
            require_int(item_version, path="item_policy_version", minimum=0)
        if current_version is not None:
            require_int(current_version, path="current_policy_version", minimum=0)
        if item_version is None or current_version is None:
            return None
        return current_version - item_version

    def is_future(self, item_version: int | None, current_version: int | None) -> bool:
        staleness = self.staleness(item_version, current_version)
        return staleness is not None and staleness < 0

    def too_stale(self, item_version: int | None, current_version: int | None) -> bool:
        staleness = self.staleness(item_version, current_version)
        return staleness is not None and staleness > self.max_stale_policy_versions


__all__ = ["StalenessPolicy"]
