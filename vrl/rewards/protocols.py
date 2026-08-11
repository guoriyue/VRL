"""Public runtime contract for reward scoring."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from vrl.rewards.types import RewardOutput, RewardSample


@runtime_checkable
class RewardRuntime(Protocol):
    """Reward runtime consumed by rollout collectors."""

    @property
    def scoring_is_nonblocking(self) -> bool:
        """Whether scoring yields while its accelerator work runs elsewhere."""
        ...

    @property
    def external_accelerator_isolation_verified(self) -> bool:
        """Whether accelerator work outside the resource plan is isolated."""
        ...

    async def preflight(self) -> None:
        """Validate external scoring dependencies before generation starts."""
        ...

    async def score(
        self,
        samples: Sequence[RewardSample],
        *,
        require_memory_release: bool = False,
    ) -> RewardOutput:
        """Score one ordered sample collection and return aligned results."""
        ...

    async def park_memory(
        self,
        *,
        required: bool,
    ) -> None:
        """Release reward-owned accelerator memory and enforce the parking gate."""
        ...

    async def shutdown(self) -> None:
        """Release every runtime-owned reward resource."""
        ...


__all__ = ["RewardRuntime"]
