"""Public runtime contract for reward scoring.

``RewardRuntime`` is the reward engine's only face toward vrl/rollouts — the
dual of ``GenerationRuntime`` in vrl/generation/protocols.py. The collector
isinstance-checks it (vrl/rollouts/collector/core.py), so this module and
types.py must stay importable without torch, CUDA utilities, or aiohttp; the
implementation (``RewardFunctionRuntime`` in runtime.py) carries those
dependencies. ``activate``/``park_memory`` mirror the generation runtime's
activate/offload pair so both engines hand the GPU back and forth through the
same lease vocabulary.

One layer below, ``RewardScorer`` is the transport seam under
``RewardFunction`` — the reward dual of the generation engine's Ray executor
layer, with two implementations: ``InProcessRewardScorer`` (runtime.py) and
``HttpRewardScorer`` (service/client.py). ``RemoteReadyScorer``,
``MemoryParkingScorer`` (the reward twin of ``BatchSizeProbeExecutor``), and
``ArtifactRetainingError`` are isinstance-probed optional capabilities.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from vrl.rewards.inference import RewardInferenceRequest, RewardInferenceResult
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

    async def activate(self) -> None:
        """Pre-warm reward model ownership at a GPU handoff.

        The inverse of :meth:`park_memory`, mirroring the generation runtime's
        activate/offload pair: parking-capable in-process rewards build or wake
        their model now so the first score does not pay load latency inside the
        measured scoring phase. CPU and remote rewards need no warm-up.
        """
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


@runtime_checkable
class RewardScorer(Protocol):
    """Transport boundary for model-backed reward inference.

    Runtime-checkable so the scorer injection point validates the COMPLETE
    protocol once (methods and capability flags together) instead of probing
    a hand-picked subset that can drift from this declaration.
    """

    @property
    def scoring_is_nonblocking(self) -> bool:
        """Whether scoring yields the caller while model work runs elsewhere."""
        ...

    @property
    def external_accelerator_isolation_verified(self) -> bool:
        """Whether accelerator work outside the resource plan is isolated."""
        ...

    async def score_batch(
        self,
        request: RewardInferenceRequest,
    ) -> list[RewardInferenceResult]: ...

    async def shutdown(self) -> None: ...


@runtime_checkable
class RemoteReadyScorer(Protocol):
    """Optional capability: a remote scorer exposes a startup readiness check.

    The in-process scorer loads lazily on the reward device and has nothing to
    verify at preflight; remote transports check reachability and identity so
    a broken service fails before the first generation batch is paid for.
    """

    async def ensure_ready(self) -> None: ...


@runtime_checkable
class ArtifactRetainingError(Protocol):
    """Error capability: scoring state is unknown, artifacts may still be read.

    Declared by the service error type (and any future transport error) so the
    artifact owner can keep shared files alive until remote completion or
    cancellation is confirmed — checked by isinstance, never by getattr.
    """

    retain_reward_artifacts: bool


@runtime_checkable
class MemoryParkingScorer(Protocol):
    """Transport boundary for retryable, verifiable reward GPU parking."""

    @property
    def requires_memory_parking(self) -> bool: ...

    async def activate(self) -> None:
        """Build or wake the model at a GPU handoff (inverse of park_memory)."""
        ...

    async def park_memory(self) -> None: ...


__all__ = [
    "ArtifactRetainingError",
    "MemoryParkingScorer",
    "RemoteReadyScorer",
    "RewardRuntime",
    "RewardScorer",
]
