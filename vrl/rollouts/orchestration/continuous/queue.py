"""Bounded ready queue for completed continuous rollout items.

A plain in-process FIFO container of completed prompt groups, bounded by the
installed prompt-batch window and an approximate byte budget. It is pure
*mechanism*: it holds the deque, tracks bytes, and rejects an item before
mutation when either hard limit would be exceeded. It deliberately knows
nothing about policy versions or staleness; the consumer owns those decisions.
"""

from __future__ import annotations

from collections import deque

from vrl.rollouts.orchestration.continuous.types import ContinuousRolloutItem
from vrl.utils.validation import require_int


class ContinuousRolloutQueue:
    """Bounded FIFO container of ready rollout items (no version logic)."""

    def __init__(
        self,
        *,
        max_items: int,
        max_bytes: int = 0,
    ) -> None:
        self.max_items = require_int(max_items, path="ContinuousRolloutQueue.max_items", minimum=1)
        self.max_bytes = require_int(max_bytes, path="ContinuousRolloutQueue.max_bytes", minimum=0)
        self._items: deque[ContinuousRolloutItem] = deque()
        self._bytes = 0

    # -- size / stats ---------------------------------------------------

    def size(self) -> int:
        return len(self._items)

    def stats(self) -> dict[str, float]:
        # Occupancy spans the installed current/preview window. The consumer
        # owns batch selection; this container reports physical occupancy.
        oldest_age = max((item.age_s for item in self._items), default=0.0)
        return {
            "ready_items": float(len(self._items)),
            "ready_bytes": float(self._bytes),
            "oldest_item_age_s": float(oldest_age),
        }

    # -- mutation -------------------------------------------------------

    def set_item_limit(self, max_items: int) -> None:
        """Resize for the installed batch window without discarding receipts."""

        next_limit = require_int(max_items, path="ContinuousRolloutQueue.max_items", minimum=1)
        if next_limit < len(self._items):
            raise RuntimeError(
                "continuous ready queue item limit cannot shrink below resident items "
                f"(ready={len(self._items)}, limit={next_limit})",
            )
        self.max_items = next_limit

    def put(self, item: ContinuousRolloutItem) -> None:
        """Append one item, failing before mutation when a hard cap is exceeded.

        Every ready item belongs to the installed current/prefetched batch window.
        Evicting an older item would prevent its batch completing exactly once, so
        overflow is a terminal capacity error rather than a replacement policy.
        """

        next_size = len(self._items) + 1
        if next_size > self.max_items:
            raise ValueError(
                "continuous ready queue exceeds its active prompt-batch item limit "
                f"(ready={len(self._items)}, limit={self.max_items})",
            )
        item_bytes = require_int(item.nbytes, path="ready item.nbytes", minimum=0)
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
                self._bytes -= item.nbytes
            else:
                kept.append(item)
        self._items = kept

    def clear(self) -> None:
        """Discard every ready item and reset occupancy; the container stays usable."""

        self._items.clear()
        self._bytes = 0


__all__ = ["ContinuousRolloutQueue"]
