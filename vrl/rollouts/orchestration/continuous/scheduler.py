"""Single admission/budget/staleness owner for the continuous rollout pipeline.

Before this object existed, the admission decision was split: the producer held
two counters (``len(inflight) < max_inflight_groups`` *and*
``queue.size() + inflight < capacity``) while the ready queue independently
enforced its own item/byte caps and could silently evict an item the producer
had already counted as admitted. Staleness was reactive only — the producer
stamped a version at submit time and the consumer/queue dropped groups that had
gone stale *after* they were generated, so a deep queue could keep generating
work that was guaranteed to be thrown away at the configured window.

``RolloutScheduler`` makes one component own *every policy-version decision*
across the pipeline, while the producer/queue/consumer keep their *mechanisms*
(Ray dispatch, the bounded container, iteration build):

- ``can_admit`` — admission (below);
- ``discard_at_receipt`` — the producer's receipt-time freshness gate;
- ``discard_prompt_set_at_receipt`` / ``drop_obsolete_prompt_sets`` — keep a
  prompt swap from letting obsolete ready/in-flight work block the new set;
- ``drop_stale`` — purge ready items past the window (consumed by the post-sync
  purge and internally before every select);
- ``select_iteration`` — pick one homogeneous, in-window version to train on.

The queue holds the deque and its byte/count safety net but no longer knows what
"stale" means; the consumer drives ``select_iteration`` but no longer holds a
``StalenessPolicy``. ``can_admit`` folds together:

1. the in-flight cap,
2. one workload item budget spanning in-flight + ready (no two diverging
   counters),
3. the ready byte budget — now visible to *admission*, so the queue's byte
   eviction is a rare safety net instead of a second, disagreeing owner,
4. an admit-time **predicted-version** staleness throttle: a group submitted now
   lands a number of training steps later, so if the pipeline already holds more
   than ``max_stale`` iterations of lookahead, the group would be discarded at
   receipt — do not generate it. This applies the predicted-staleness idea
   ``weight_version = current_step + total_pending // rollouts_per_batch``,
   adapted to vrl's per-iteration cadence: one consumed iteration == one version
   bump.

The throttle never reduces what gets *trained* and never starves a full
iteration: ``predicted == 0`` until a complete iteration of work is queued, so
admission always reaches exactly ``(max_stale + 1)`` iterations of lookahead,
then stops. Raising ``max_ready_groups`` beyond that is staleness-capped, not a
deeper effective buffer (see the continuous schedule module docstring).
"""

from __future__ import annotations

from dataclasses import dataclass

from vrl.rollouts.orchestration.continuous.queue import ContinuousRolloutQueue
from vrl.rollouts.orchestration.continuous.staleness import StalenessPolicy
from vrl.rollouts.orchestration.continuous.types import ContinuousRolloutItem


@dataclass(slots=True, frozen=True)
class AdmitDecision:
    """Whether the producer may submit one more group, and why not."""

    admit: bool
    # "ok" | "inflight_full" | "item_budget_full" | "byte_budget_full"
    # | "would_land_too_stale". Surfaced as a producer gauge so "why is the
    # producer not admitting?" is answerable per step without attaching a
    # debugger.
    reason: str


class RolloutScheduler:
    """Owns continuous-rollout admission: budget + admit-time staleness throttle."""

    def __init__(
        self,
        *,
        staleness: StalenessPolicy,
        max_inflight_groups: int,
        capacity: int,
        max_bytes: int,
        groups_per_iteration: int,
    ) -> None:
        if int(max_inflight_groups) < 1:
            raise ValueError("RolloutScheduler.max_inflight_groups must be >= 1")
        if int(capacity) < 1:
            raise ValueError("RolloutScheduler.capacity must be >= 1")
        if int(groups_per_iteration) < 1:
            raise ValueError("RolloutScheduler.groups_per_iteration must be >= 1")
        self._staleness = staleness
        self.max_inflight_groups = int(max_inflight_groups)
        self.capacity = int(capacity)
        self.max_bytes = max(0, int(max_bytes))
        self.groups_per_iteration = int(groups_per_iteration)

    @property
    def staleness(self) -> StalenessPolicy:
        """The shared policy; queue/consumer still read it for select/drop."""

        return self._staleness

    def set_groups_per_iteration(self, groups_per_iteration: int) -> None:
        """Track a prompt-set swap so the predicted-version math stays correct."""

        if int(groups_per_iteration) < 1:
            raise ValueError("groups_per_iteration must be >= 1")
        self.groups_per_iteration = int(groups_per_iteration)

    def predicted_landing_staleness(
        self,
        *,
        inflight_count: int,
        ready_items: int,
    ) -> int:
        """Versions this group would trail by once the trainer reaches it.

        A group submitted now sits behind ``inflight + ready`` groups of work.
        The consumer drains one homogeneous iteration (``groups_per_iteration``
        distinct groups) per training step and each step bumps the policy
        version once, so the version advances by one per full iteration of
        backlog ahead of this group.
        """

        pending = max(0, int(inflight_count) + int(ready_items))
        return pending // self.groups_per_iteration

    def can_admit(
        self,
        *,
        inflight_count: int,
        ready_items: int,
        ready_bytes: int,
    ) -> AdmitDecision:
        """Single admission predicate (replaces the producer's two counters)."""

        if int(inflight_count) >= self.max_inflight_groups:
            return AdmitDecision(False, "inflight_full")
        if int(ready_items) + int(inflight_count) >= self.capacity:
            return AdmitDecision(False, "item_budget_full")
        # ready_bytes covers ready items only (an in-flight group's footprint is
        # unknown until it completes), so a burst of completions can still
        # overshoot max_bytes transiently — that residual is the queue's safety
        # net. Steady-state admission is byte-aware here, so the queue's eviction
        # no longer fights this owner.
        if self.max_bytes > 0 and int(ready_bytes) >= self.max_bytes:
            return AdmitDecision(False, "byte_budget_full")
        predicted = self.predicted_landing_staleness(
            inflight_count=inflight_count,
            ready_items=ready_items,
        )
        if predicted > int(self._staleness.max_stale_policy_versions):
            return AdmitDecision(False, "would_land_too_stale")
        return AdmitDecision(True, "ok")

    # -- post-admission version decisions (queue/producer call these) --------

    def discard_at_receipt(
        self,
        item_version: int | None,
        current_version: int | None,
    ) -> bool:
        """Whether a just-finished group is already past the window.

        The producer's receipt-time gate: a group stamped at submit time can
        finish only after the trainer advanced, so drop it here instead of
        queuing work the consumer would discard anyway. Absent versions and
        future items (a bug) return False so they still flow to the consumer's
        fail-fast.
        """

        return self._staleness.too_stale(item_version, current_version)

    def discard_prompt_set_at_receipt(
        self,
        item_prompt_set_id: int,
        current_prompt_set_id: int,
    ) -> bool:
        """Whether a completed group belongs to an obsolete prompt set."""

        return int(item_prompt_set_id) != int(current_prompt_set_id)

    def drop_obsolete_prompt_sets(
        self,
        queue: ContinuousRolloutQueue,
        *,
        prompt_set_id: int,
    ) -> int:
        """Drop ready items from prior prompt sets so they cannot block admission."""

        to_drop = [item for item in queue.snapshot() if item.prompt_set_id != int(prompt_set_id)]
        if to_drop:
            queue.remove(to_drop)
            queue.note_dropped_prompt_set(len(to_drop))
        return len(to_drop)

    def drop_stale(
        self,
        queue: ContinuousRolloutQueue,
        *,
        current_version: int | None,
    ) -> int:
        """Drop ready items beyond the window; fail fast on future items.

        The version verdict is the scheduler's; the queue only snapshots and
        removes (and tracks its dropped-stale stat).
        """

        if current_version is None:
            return 0
        to_drop: list[ContinuousRolloutItem] = []
        for item in queue.snapshot():
            version = item.rollout_policy_version
            if self._staleness.is_future(version, current_version):
                raise RuntimeError(
                    "continuous queue item is newer than the trainer policy "
                    f"(item={version}, trainer={current_version}); weight-sync "
                    "barrier invariant violated",
                )
            if self._staleness.too_stale(version, current_version):
                to_drop.append(item)
        if to_drop:
            queue.remove(to_drop)
            queue.note_dropped_stale(len(to_drop))
        return len(to_drop)

    def select_iteration(
        self,
        queue: ContinuousRolloutQueue,
        *,
        min_groups: int,
        current_version: int | None,
        prompt_set_id: int = 0,
    ) -> tuple[int | None, list[ContinuousRolloutItem]] | None:
        """Pop ``min_groups`` distinct-group items at one homogeneous version.

        Only items from the requested ``prompt_set_id`` are eligible — a prompt
        swap must not let the previous set's ready items fill this iteration,
        even when they share a policy version and slot numbering. Drops too-stale
        items first, then prefers the newest in-window version with enough
        distinct groups. Returns ``None`` when no single version can fill an
        iteration yet. Homogeneity is the off-policy-critical invariant: an
        iteration never mixes versions (and never mixes prompt sets).
        """

        self.drop_stale(queue, current_version=current_version)

        by_version: dict[int | None, list[ContinuousRolloutItem]] = {}
        for item in queue.snapshot():
            if item.prompt_set_id != prompt_set_id:
                continue
            by_version.setdefault(item.rollout_policy_version, []).append(item)

        for version in self._candidate_versions(by_version, current_version):
            chosen = self._take_distinct(by_version[version], min_groups)
            if chosen is not None:
                queue.remove(chosen)
                return version, chosen
        return None

    def _candidate_versions(
        self,
        by_version: dict[int | None, list[ContinuousRolloutItem]],
        current_version: int | None,
    ) -> list[int | None]:
        # Newest first so an in-bound fresh version always wins over a stale one.
        versions = sorted(
            by_version.keys(),
            key=lambda v: -1 if v is None else int(v),
            reverse=True,
        )
        return [version for version in versions if self._staleness.admit(version, current_version)]

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


__all__ = ["AdmitDecision", "RolloutScheduler"]
