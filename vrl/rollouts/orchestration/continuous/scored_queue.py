"""Bounded ready queue for completed continuous rollout items.

A plain in-process FIFO container of completed prompt groups, bounded by the
installed prompt batch. It is pure *mechanism*: it holds the deque and rejects
an item before mutation when the item cap would be exceeded. It deliberately
knows nothing about policy versions or staleness; the consumer owns those
decisions.
"""

from __future__ import annotations

from collections import deque

from vrl.rollouts.orchestration.continuous.types import ScoredRollout


class ScoredRolloutQueue:
    """Bounded FIFO container of ready rollout items (no version logic)."""

    def __init__(self, *, max_items: int) -> None:
        self.max_items = max_items
        self._items: deque[ScoredRollout] = deque()

    # -- size / stats ---------------------------------------------------

    def size(self) -> int:
        return len(self._items)

    def stats(self) -> dict[str, float]:
        oldest_age = max((item.age_s for item in self._items), default=0.0)
        return {
            "ready_items": float(len(self._items)),
            "oldest_item_age_s": float(oldest_age),
        }

    # -- mutation -------------------------------------------------------

    def set_item_limit(self, max_items: int) -> None:
        """Resize for the installed prompt batch without discarding receipts."""

        if max_items < len(self._items):
            raise RuntimeError(
                "continuous ready queue item limit cannot shrink below resident items "
                f"(ready={len(self._items)}, limit={max_items})",
            )
        self.max_items = max_items

    def put(self, item: ScoredRollout) -> None:
        """Append one item, failing before mutation when the item cap is exceeded.

        Every ready item belongs to the installed prompt batch. Evicting an
        older item would prevent its batch completing exactly once, so overflow
        is a terminal capacity error rather than a replacement policy.
        """

        if len(self._items) + 1 > self.max_items:
            raise ValueError(
                "continuous ready queue exceeds its active prompt-batch item limit "
                f"(ready={len(self._items)}, limit={self.max_items})",
            )
        self._items.append(item)

    def snapshot(self) -> list[ScoredRollout]:
        """FIFO-ordered view of the current items for the consumer to inspect."""

        return list(self._items)

    def remove(self, items: list[ScoredRollout]) -> None:
        """Drop the given items (by identity)."""

        remove_ids = {id(item) for item in items}
        self._items = deque(item for item in self._items if id(item) not in remove_ids)

    def clear(self) -> None:
        """Discard every ready item; the container stays usable."""

        self._items.clear()


__all__ = ["ScoredRolloutQueue"]
