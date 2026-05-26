"""Bounded ready queue for completed continuous rollout items.

The queue is a plain in-process FIFO of completed prompt groups. It is bounded
by item count and an approximate byte budget; admission cadence is gated by the
producer, so ``put`` only enforces the hard caps as a safety net. Selection is
the off-policy-critical part: ``select_iteration`` returns a *single
homogeneous policy version*, never a mix.
"""

from __future__ import annotations

from collections import deque

from vrl.rollouts.orchestration.continuous.staleness import StalenessPolicy
from vrl.rollouts.orchestration.continuous.types import ContinuousRolloutItem


class ContinuousRolloutQueue:
    """Bounded FIFO of ready rollout items with same-policy drain."""

    def __init__(
        self,
        *,
        max_items: int,
        max_bytes: int = 0,
        drop_policy: str = "drop_oldest_stale",
    ) -> None:
        if int(max_items) < 1:
            raise ValueError("ContinuousRolloutQueue.max_items must be >= 1")
        if drop_policy not in {"drop_oldest_stale", "drop_oldest"}:
            raise ValueError(f"unknown drop_policy: {drop_policy!r}")
        self.max_items = int(max_items)
        self.max_bytes = int(max_bytes)
        self.drop_policy = drop_policy
        self._items: deque[ContinuousRolloutItem] = deque()
        self._bytes = 0
        self.dropped_stale = 0
        self.dropped_backpressure = 0

    # -- size / stats ---------------------------------------------------

    def size(self) -> int:
        return len(self._items)

    def ready_bytes(self) -> int:
        return self._bytes

    def distinct_group_count(self, version: int | None = None) -> int:
        keys = {
            item.group_key
            for item in self._items
            if version is None or item.rollout_policy_version == version
        }
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
            "dropped_backpressure": float(self.dropped_backpressure),
        }

    # -- mutation -------------------------------------------------------

    def put(self, item: ContinuousRolloutItem) -> None:
        self._items.append(item)
        self._bytes += int(item.nbytes)
        self._enforce_caps()

    def drop_too_stale(
        self,
        *,
        current_version: int | None,
        staleness: StalenessPolicy,
    ) -> int:
        """Drop items beyond the staleness bound; fail fast on future items."""

        if current_version is None:
            return 0
        kept: deque[ContinuousRolloutItem] = deque()
        dropped = 0
        for item in self._items:
            version = item.rollout_policy_version
            if staleness.is_future(version, current_version):
                raise RuntimeError(
                    "continuous queue item is newer than the trainer policy "
                    f"(item={version}, trainer={current_version}); weight-sync "
                    "barrier invariant violated",
                )
            if staleness.too_stale(version, current_version):
                self._bytes -= int(item.nbytes)
                dropped += 1
                continue
            kept.append(item)
        self._items = kept
        self.dropped_stale += dropped
        return dropped

    def select_iteration(
        self,
        *,
        min_groups: int,
        current_version: int | None,
        staleness: StalenessPolicy,
    ) -> tuple[int | None, list[ContinuousRolloutItem]] | None:
        """Pop ``min_groups`` distinct-group items at one homogeneous version.

        Drops too-stale items first, then prefers the newest version that is
        within the staleness bound and has enough distinct groups. Returns
        ``None`` when no single version can fill an iteration yet.
        """

        self.drop_too_stale(current_version=current_version, staleness=staleness)

        # Group items by version, preserving FIFO order within each version.
        by_version: dict[int | None, list[ContinuousRolloutItem]] = {}
        for item in self._items:
            by_version.setdefault(item.rollout_policy_version, []).append(item)

        for version in self._candidate_versions(by_version, current_version, staleness):
            chosen = self._take_distinct(by_version[version], min_groups)
            if chosen is not None:
                self._remove(chosen)
                return version, chosen
        return None

    def close(self) -> None:
        self._items.clear()
        self._bytes = 0

    # -- internals ------------------------------------------------------

    def _candidate_versions(
        self,
        by_version: dict[int | None, list[ContinuousRolloutItem]],
        current_version: int | None,
        staleness: StalenessPolicy,
    ) -> list[int | None]:
        # Newest first so an in-bound fresh version always wins over a stale one.
        versions = sorted(
            by_version.keys(),
            key=lambda v: (-1 if v is None else int(v)),
            reverse=True,
        )
        return [
            version
            for version in versions
            if staleness.admit(version, current_version)
        ]

    @staticmethod
    def _take_distinct(
        items: list[ContinuousRolloutItem],
        min_groups: int,
    ) -> list[ContinuousRolloutItem] | None:
        chosen: list[ContinuousRolloutItem] = []
        seen: set[int] = set()
        for item in items:
            if item.group_key in seen:
                continue
            seen.add(item.group_key)
            chosen.append(item)
            if len(chosen) >= min_groups:
                return chosen
        return None

    def _remove(self, items: list[ContinuousRolloutItem]) -> None:
        remove_ids = {id(item) for item in items}
        kept: deque[ContinuousRolloutItem] = deque()
        for item in self._items:
            if id(item) in remove_ids:
                self._bytes -= int(item.nbytes)
            else:
                kept.append(item)
        self._items = kept

    def _enforce_caps(self) -> None:
        while self._over_capacity():
            victim = self._pick_victim()
            if victim is None:
                break
            self._items.remove(victim)
            self._bytes -= int(victim.nbytes)
            self.dropped_backpressure += 1

    def _over_capacity(self) -> bool:
        if len(self._items) > self.max_items:
            return True
        return self.max_bytes > 0 and self._bytes > self.max_bytes

    def _pick_victim(self) -> ContinuousRolloutItem | None:
        if not self._items:
            return None
        if self.drop_policy == "drop_oldest":
            return self._items[0]
        # drop_oldest_stale: oldest by completion time wins; ties keep FIFO.
        return min(self._items, key=lambda item: item.completed_at)


__all__ = ["ContinuousRolloutQueue"]
