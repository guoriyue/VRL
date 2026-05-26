"""Staleness policy for the continuous rollout queue.

Staleness is ``trainer_current_version - rollout_item_version``. A negative
value means the item was produced by a policy newer than the trainer, which
can only happen through a bug, so callers should fail fast on it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StalenessPolicy:
    """Bound on how far an item's policy version may trail the trainer."""

    max_stale_policy_versions: int = 0

    def __post_init__(self) -> None:
        if int(self.max_stale_policy_versions) < 0:
            raise ValueError("max_stale_policy_versions must be >= 0")

    def staleness(
        self,
        item_version: int | None,
        current_version: int | None,
    ) -> int | None:
        """Versions behind the trainer, or ``None`` when versions are absent."""

        if item_version is None or current_version is None:
            return None
        return int(current_version) - int(item_version)

    def is_future(self, item_version: int | None, current_version: int | None) -> bool:
        staleness = self.staleness(item_version, current_version)
        return staleness is not None and staleness < 0

    def admit(self, item_version: int | None, current_version: int | None) -> bool:
        """True when an item is fresh enough to train on."""

        staleness = self.staleness(item_version, current_version)
        if staleness is None:
            return True
        return 0 <= staleness <= int(self.max_stale_policy_versions)

    def too_stale(self, item_version: int | None, current_version: int | None) -> bool:
        staleness = self.staleness(item_version, current_version)
        return staleness is not None and staleness > int(self.max_stale_policy_versions)


__all__ = ["StalenessPolicy"]
