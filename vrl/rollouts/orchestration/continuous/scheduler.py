"""Admission and policy-version decisions for continuous rollout.

``RolloutScheduler`` owns the decisions shared across the pipeline while the
producer, queue, and consumer keep their mechanisms:

- ``can_admit`` enforces the live in-flight and ready-byte limits;
- ``is_stale_at_receipt`` detects work that became stale during generation;
- ``validate_ready_versions`` rejects a finite batch that can no longer be trained;
- ``select_iteration`` picks one homogeneous, in-window version to train on.

The finite producer installs exactly one prompt batch at a time. Its pending
slots already bound item count by the queue capacity, so admission does not keep
a second item-budget or predicted-lookahead model that production cannot reach.
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
    # "ok" | "inflight_full" | "byte_budget_full".
    reason: str


class RolloutScheduler:
    """Owns continuous-rollout admission: budget + admit-time staleness throttle."""

    def __init__(
        self,
        *,
        staleness: StalenessPolicy,
        max_inflight_groups: int,
        max_bytes: int,
    ) -> None:
        if int(max_inflight_groups) < 1:
            raise ValueError("RolloutScheduler.max_inflight_groups must be >= 1")
        self._staleness = staleness
        self.max_inflight_groups = int(max_inflight_groups)
        self.max_bytes = max(0, int(max_bytes))

    @property
    def staleness(self) -> StalenessPolicy:
        """The shared policy; queue/consumer still read it for select/drop."""

        return self._staleness

    def can_admit(
        self,
        *,
        inflight_count: int,
        ready_bytes: int,
    ) -> AdmitDecision:
        """Return whether one more prompt slot may start collection."""

        if int(inflight_count) >= self.max_inflight_groups:
            return AdmitDecision(False, "inflight_full")
        # ready_bytes covers ready items only (an in-flight group's footprint is
        # unknown until it completes), so a burst of completions can still
        # overshoot max_bytes transiently — that residual is the queue's safety
        # net. Steady-state admission is byte-aware here, so the queue's eviction
        # no longer fights this owner.
        if self.max_bytes > 0 and int(ready_bytes) >= self.max_bytes:
            return AdmitDecision(False, "byte_budget_full")
        return AdmitDecision(True, "ok")

    # -- post-admission version decisions (queue/producer call these) --------

    def is_stale_at_receipt(
        self,
        item_version: int | None,
        current_version: int | None,
    ) -> bool:
        """Whether a just-finished group is already past the window.

        The producer's receipt-time gate: a group stamped at submit time can
        finish only after the trainer advanced. Absent versions and future items
        (a bug) return False so they still flow to the consumer's fail-fast.
        """

        return self._staleness.too_stale(item_version, current_version)

    def validate_ready_versions(
        self,
        queue: ContinuousRolloutQueue,
        *,
        current_version: int | None,
    ) -> None:
        """Fail if any ready slot is outside the trainable version window.

        A finite prompt batch cannot drop and regenerate one ready slot without
        changing its frozen policy version. Failing here preserves the original
        cause instead of leaving the consumer waiting for an impossible batch.
        """

        if current_version is None:
            return
        for item in queue.snapshot():
            version = item.rollout_policy_version
            if self._staleness.is_future(version, current_version):
                raise RuntimeError(
                    "continuous queue item is newer than the trainer policy "
                    f"(item={version}, trainer={current_version}); weight-sync "
                    "barrier invariant violated",
                )
            if self._staleness.too_stale(version, current_version):
                raise RuntimeError(
                    "continuous ready prompt batch is older than the policy window "
                    f"(item={version}, trainer={current_version})",
                )

    def select_iteration(
        self,
        queue: ContinuousRolloutQueue,
        *,
        min_groups: int,
        current_version: int | None,
    ) -> tuple[int | None, list[ContinuousRolloutItem]] | None:
        """Pop ``min_groups`` distinct-group items at one homogeneous version.

        Rejects stale, future, duplicate, or mixed-version ready state. Returns
        ``None`` until the active finite batch has enough groups. Homogeneity is
        the off-policy-critical invariant: an iteration never mixes versions.
        """

        if int(min_groups) < 1:
            raise ValueError("continuous min_groups must be >= 1")
        self.validate_ready_versions(queue, current_version=current_version)

        items = queue.snapshot()
        versions = {item.rollout_policy_version for item in items}
        if len(versions) > 1:
            raise RuntimeError(
                "continuous ready prompt batch mixes policy versions "
                f"{sorted(versions, key=lambda version: -1 if version is None else version)}",
            )
        group_keys = [item.group_key for item in items]
        if len(group_keys) != len(set(group_keys)):
            raise RuntimeError(
                "continuous ready prompt batch contains duplicate group slots",
            )
        if len(items) < min_groups:
            return None
        if len(items) > min_groups:
            raise RuntimeError(
                "continuous ready prompt batch exceeds its expected group count "
                f"(ready={len(items)}, expected={min_groups})",
            )
        queue.remove(items)
        version = items[0].rollout_policy_version
        return version, items


__all__ = ["AdmitDecision", "RolloutScheduler"]
