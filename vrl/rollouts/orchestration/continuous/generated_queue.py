"""Reserved capacity for generated groups awaiting reward completion.

Capacity belongs to the whole generation-to-reward lifetime. Taking a group
for scoring does not free its reservation: the artifact is still resident.
The owner loop uses nonblocking operations so backpressure cannot block weight
commits or cancellation. This queue never accepts trainer-ready batches.
"""

from __future__ import annotations

from collections import deque

from vrl.rollouts.collector.core import GeneratedPromptGroup


class GeneratedRolloutQueue:
    """Bounded generation receipts, including in-flight capacity reservations."""

    def __init__(self, *, max_items: int, max_bytes: int) -> None:
        if max_items < 1 or max_bytes < 1:
            raise ValueError("generated queue requires positive item and byte limits")
        self.max_items = max_items
        self.max_bytes = max_bytes
        self._reserved: dict[tuple[int, int], int] = {}
        self._receipts: dict[tuple[int, int], GeneratedPromptGroup] = {}
        self._ready: deque[tuple[int, int]] = deque()
        self._scoring: set[tuple[int, int]] = set()
        self._bytes = 0
        self._closed = False

    def reserve(self, key: tuple[int, int], *, max_group_bytes: int) -> bool:
        """Reserve a declared per-group ceiling before launching generation.

        False is temporary backpressure; a group that can never fit is an
        error. No work should be launched without a successful reservation.
        """

        if self._closed:
            raise RuntimeError("generated queue is closed")
        if key in self._reserved:
            raise RuntimeError(f"duplicate generated group reservation: {key}")
        if not 0 < max_group_bytes <= self.max_bytes:
            raise ValueError("generated group byte ceiling must fit the queue budget")
        if len(self._reserved) >= self.max_items or self._bytes + max_group_bytes > self.max_bytes:
            return False
        self._reserved[key] = max_group_bytes
        self._bytes += max_group_bytes
        return True

    def put(self, key: tuple[int, int], receipt: GeneratedPromptGroup, *, nbytes: int) -> None:
        """Transfer a generated receipt into its existing reservation.

        Reconcile the ceiling to measured payload bytes. Oversized receipts
        fail before mutation; the owner must release the reservation while
        terminating the offending generation attempt.
        """

        if self._closed:
            raise RuntimeError("generated queue is closed")
        if key not in self._reserved:
            raise RuntimeError(f"unreserved generated group: {key}")
        if key in self._receipts:
            raise RuntimeError(f"duplicate generated receipt: {key}")
        ceiling = self._reserved[key]
        if nbytes < 0 or nbytes > ceiling:
            raise ValueError(
                f"generated group exceeds reserved byte ceiling: {key} "
                f"(bytes={nbytes}, ceiling={ceiling})",
            )
        self._reserved[key] = nbytes
        self._bytes += nbytes - ceiling
        self._receipts[key] = receipt
        self._ready.append(key)

    def take(self) -> tuple[tuple[int, int], GeneratedPromptGroup] | None:
        """Lend the next receipt to scoring while retaining its capacity."""

        if not self._ready:
            return None
        key = self._ready.popleft()
        self._scoring.add(key)
        return key, self._receipts[key]

    def release(self, key: tuple[int, int]) -> None:
        """Release a settled or cancelled group's artifact and capacity once."""

        if self._closed:
            return
        if key not in self._reserved:
            raise RuntimeError(f"generated group has no reservation to release: {key}")
        self._bytes -= self._reserved.pop(key)
        self._receipts.pop(key, None)
        self._scoring.discard(key)
        if key in self._ready:
            self._ready.remove(key)

    def stats(self) -> dict[str, float]:
        return {
            "reserved_groups": float(len(self._reserved)),
            "reserved_bytes": float(self._bytes),
            "unscored_items": float(len(self._ready)),
            "scoring_groups": float(len(self._scoring)),
        }

    def close(self) -> None:
        """Release references after the owner has settled generation and reward."""

        self._closed = True
        self._ready.clear()
        self._scoring.clear()
        self._receipts.clear()
        self._reserved.clear()
        self._bytes = 0
