"""Bounded ready queue for completed continuous rollout items.

A plain in-process FIFO container of completed prompt groups, bounded by the
active prompt-batch size and an approximate byte budget. It is pure
*mechanism*: it holds the deque, tracks bytes, and rejects an item before
mutation when either hard limit would be exceeded. It deliberately knows
nothing about policy versions or staleness; the consumer owns those decisions.
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
        if int(max_bytes) < 0:
            raise ValueError("ContinuousRolloutQueue.max_bytes must be >= 0")
        self.max_items = int(max_items)
        self.max_bytes = int(max_bytes)
        self._items: deque[ContinuousRolloutItem] = deque()
        self._bytes = 0

    # -- size / stats ---------------------------------------------------

    def size(self) -> int:
        return len(self._items)

    def stats(self) -> dict[str, float]:
        group_keys = {item.group_key for item in self._items}
        versions = {
            item.rollout_policy_version
            for item in self._items
            if item.rollout_policy_version is not None
        }
        oldest_age = max((item.age_s for item in self._items), default=0.0)
        return {
            "ready_items": float(len(self._items)),
            "ready_groups": float(len(group_keys)),
            "ready_bytes": float(self._bytes),
            "ready_versions": float(len(versions)),
            "oldest_item_age_s": float(oldest_age),
        }

    # -- mutation -------------------------------------------------------

    def set_item_limit(self, max_items: int) -> None:
        """Resize the item limit between finite prompt batches."""

        next_limit = int(max_items)
        if next_limit < 1:
            raise ValueError("ContinuousRolloutQueue.max_items must be >= 1")
        if self._items:
            raise RuntimeError(
                "continuous ready queue item limit can change only between batches "
                f"(ready={len(self._items)})",
            )
        self.max_items = next_limit

    def put(self, item: ContinuousRolloutItem) -> None:
        """Append one item, failing before mutation when a hard cap is exceeded.

        Every ready item belongs to the current finite prompt batch. Evicting an
        older item would make that batch impossible to complete exactly once, so
        overflow is a terminal capacity error rather than a replacement policy.
        """

        next_size = len(self._items) + 1
        if next_size > self.max_items:
            raise ValueError(
                "continuous ready queue exceeds its active prompt-batch item limit "
                f"(ready={len(self._items)}, limit={self.max_items})",
            )
        item_bytes = int(item.nbytes)
        next_bytes = self._bytes + item_bytes
        if self.max_bytes > 0 and next_bytes > self.max_bytes:
            raise ValueError(
                "continuous ready queue exceeds its byte limit "
                f"(ready_bytes={self._bytes}, item_bytes={item_bytes}, "
                f"limit={self.max_bytes})",
            )
        self._items.append(item)
        self._bytes = next_bytes

    def snapshot(self) -> list[ContinuousRolloutItem]:
        """FIFO-ordered view of the current items for the consumer to inspect."""

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

    def close(self) -> None:
        self._items.clear()
        self._bytes = 0


__all__ = ["ContinuousRolloutQueue"]
