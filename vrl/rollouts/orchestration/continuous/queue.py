"""Bounded ready queue for completed continuous rollout items.

A plain in-process FIFO container of completed prompt groups, bounded by item
count and an approximate byte budget. It is pure *mechanism*: it holds the
deque, tracks bytes, and enforces a hard cap as a backpressure safety net. It
deliberately knows nothing about policy versions or staleness — the
``RolloutScheduler`` owns every version decision (drop-stale, select a
homogeneous iteration) and drives this container through ``snapshot`` /
``remove`` / ``note_dropped_stale``.
"""

from __future__ import annotations

from collections import deque

from vrl.rollouts.orchestration.continuous.types import ContinuousRolloutItem


class ContinuousRolloutQueue:
    """Bounded FIFO container of ready rollout items (no version logic)."""

    def __init__(
        self,
        *,
        max_items: int,
        max_bytes: int = 0,
    ) -> None:
        if int(max_items) < 1:
            raise ValueError("ContinuousRolloutQueue.max_items must be >= 1")
        self.max_items = int(max_items)
        self.max_bytes = int(max_bytes)
        self._items: deque[ContinuousRolloutItem] = deque()
        self._bytes = 0
        self.dropped_stale = 0
        self.dropped_prompt_set = 0
        self.dropped_backpressure = 0

    # -- size / stats ---------------------------------------------------

    def size(self) -> int:
        return len(self._items)

    def ready_bytes(self) -> int:
        return self._bytes

    def distinct_group_count(self) -> int:
        keys = {item.group_key for item in self._items}
        return len(keys)

    def stats(self) -> dict[str, float]:
        versions = {
            item.rollout_policy_version
            for item in self._items
            if item.rollout_policy_version is not None
        }
        oldest_age = max((item.age_s for item in self._items), default=0.0)
        return {
            "ready_items": float(len(self._items)),
            "ready_groups": float(self.distinct_group_count()),
            "ready_bytes": float(self._bytes),
            "ready_versions": float(len(versions)),
            "oldest_item_age_s": float(oldest_age),
            "dropped_stale": float(self.dropped_stale),
            "dropped_prompt_set": float(self.dropped_prompt_set),
            "dropped_backpressure": float(self.dropped_backpressure),
        }

    # -- mutation -------------------------------------------------------

    def put(self, item: ContinuousRolloutItem) -> None:
        self._items.append(item)
        self._bytes += int(item.nbytes)
        self._enforce_caps()

    def snapshot(self) -> list[ContinuousRolloutItem]:
        """FIFO-ordered view of the current items for the scheduler to inspect."""

        return list(self._items)

    def remove(self, items: list[ContinuousRolloutItem]) -> None:
        """Drop the given items (by identity) and fix the byte accounting."""

        remove_ids = {id(item) for item in items}
        kept: deque[ContinuousRolloutItem] = deque()
        for item in self._items:
            if id(item) in remove_ids:
                self._bytes -= int(item.nbytes)
            else:
                kept.append(item)
        self._items = kept

    def note_dropped_stale(self, count: int) -> None:
        """Record that the scheduler dropped ``count`` too-stale items."""

        self.dropped_stale += int(count)

    def note_dropped_prompt_set(self, count: int) -> None:
        """Record that the scheduler dropped ``count`` obsolete prompt-set items."""

        self.dropped_prompt_set += int(count)

    def close(self) -> None:
        self._items.clear()
        self._bytes = 0

    # -- internals ------------------------------------------------------

    def _enforce_caps(self) -> None:
        # Items arrive in completion order, so the FIFO head is always the
        # oldest item — evicting it is both drop-oldest and drop-oldest-stale.
        while self._items and self._over_capacity():
            victim = self._items.popleft()
            self._bytes -= int(victim.nbytes)
            self.dropped_backpressure += 1

    def _over_capacity(self) -> bool:
        if len(self._items) > self.max_items:
            return True
        return self.max_bytes > 0 and self._bytes > self.max_bytes


__all__ = ["ContinuousRolloutQueue"]
