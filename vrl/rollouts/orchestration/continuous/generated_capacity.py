"""Capacity accounting for groups being generated, awaiting reward, or scoring.

Reservations outlive generation slots and remain charged through reward retries.
This owner-loop object retains no payloads and performs no queueing or waiting.
"""

from __future__ import annotations

from vrl.utils.config import require_exact_int


class GeneratedRolloutCapacity:
    """Account group and byte capacity from generation admission through scoring."""

    def __init__(self, *, max_groups: int, max_bytes: int) -> None:
        require_exact_int(max_groups, path="generated capacity.max_groups", minimum=1)
        require_exact_int(max_bytes, path="generated capacity.max_bytes", minimum=1)
        self.max_groups = max_groups
        self.max_bytes = max_bytes
        self._reserved: dict[tuple[int, int], int] = {}
        self._waiting: set[tuple[int, int]] = set()
        self._scoring: set[tuple[int, int]] = set()
        self._bytes = 0
        self._closed = False

    def reserve(self, key: tuple[int, int], *, max_group_bytes: int) -> bool:
        """Reserve a declared per-group ceiling before launching generation.

        False is temporary backpressure; a group that can never fit is an
        error. No work should be launched without a successful reservation.
        """

        if self._closed:
            raise RuntimeError("generated capacity is closed")
        if key in self._reserved:
            raise RuntimeError(f"duplicate generated group reservation: {key}")
        require_exact_int(max_group_bytes, path="max_group_bytes", minimum=1)
        if max_group_bytes > self.max_bytes:
            raise ValueError("generated group byte ceiling must fit the capacity budget")
        if (
            len(self._reserved) >= self.max_groups
            or self._bytes + max_group_bytes > self.max_bytes
        ):
            return False
        self._reserved[key] = max_group_bytes
        self._bytes += max_group_bytes
        return True

    def record_generated(self, key: tuple[int, int], *, nbytes: int) -> None:
        """Replace the reserved ceiling with the estimated generated payload size.

        The producer retains the payload. Oversized or repeated reports fail
        before changing capacity; admission is released by the producer's cleanup.
        """

        if self._closed:
            raise RuntimeError("generated capacity is closed")
        if key not in self._reserved:
            raise RuntimeError(f"unreserved generated group: {key}")
        if key in self._waiting or key in self._scoring:
            raise RuntimeError(f"duplicate generated size report: {key}")
        ceiling = self._reserved[key]
        require_exact_int(nbytes, path="generated group.nbytes", minimum=0)
        if nbytes > ceiling:
            raise ValueError(
                f"generated group exceeds reserved byte ceiling: {key} "
                f"(bytes={nbytes}, ceiling={ceiling})",
            )
        self._reserved[key] = nbytes
        self._bytes += nbytes - ceiling
        self._waiting.add(key)

    def start_scoring(self, key: tuple[int, int]) -> None:
        """Mark a reserved group as scoring without releasing its capacity."""

        if key not in self._waiting:
            raise RuntimeError(f"generated group is not waiting for scoring: {key}")
        self._waiting.remove(key)
        self._scoring.add(key)

    def release(self, key: tuple[int, int]) -> None:
        """Release a settled or cancelled group's capacity once."""

        if self._closed:
            return
        if key not in self._reserved:
            raise RuntimeError(f"generated group has no reservation to release: {key}")
        self._bytes -= self._reserved.pop(key)
        self._waiting.discard(key)
        self._scoring.discard(key)

    def stats(self) -> dict[str, float]:
        return {
            "reserved_groups": float(len(self._reserved)),
            "reserved_bytes": float(self._bytes),
            "unscored_items": float(len(self._waiting)),
            "scoring_groups": float(len(self._scoring)),
        }

    def close(self) -> None:
        """Close admission after the owner has settled generation and reward."""

        self._closed = True
        self._waiting.clear()
        self._scoring.clear()
        self._reserved.clear()
        self._bytes = 0
