"""Public runtime contract for reward scoring."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vrl.rewards.inference import RewardMemoryReleaseProof
from vrl.rewards.types import RewardOutput, RewardRequest


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
        request: RewardRequest,
        *,
        require_memory_release: bool = False,
    ) -> RewardOutput:
        """Score one request and return sample-aligned results."""
        ...

    async def park_memory(
        self,
        *,
        required: bool,
    ) -> tuple[RewardMemoryReleaseProof, ...]:
        """Release reward-owned accelerator memory and validate fresh proofs."""
        ...

    async def shutdown(self) -> None:
        """Release every runtime-owned reward resource."""
        ...


__all__ = ["RewardRuntime"]
