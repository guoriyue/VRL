"""Ray generation worker config."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vrl.config.schema import RootConfig

# Engine-level stall budget of one generation batch (the pipelined path grants
# it once per batch of an engine's share). One hour covers the observed
# ~30-minute cold torch.compile plus a 733-second Cosmos batch with margin;
# opaque control calls keep the tighter distributed.rollout.worker_rpc_timeout_s.
GENERATION_STALL_TIMEOUT_S = 3600.0


@dataclass(frozen=True, slots=True)
class RolloutWorkerConfig:
    """Frozen runtime projection of the public rollout-worker section.

    ``generation_stall_timeout_s`` is not a YAML key: ``from_root`` fills it
    from ``GENERATION_STALL_TIMEOUT_S``, which a Ray runtime test patches
    before the launch freezes this config.
    """

    cpus_per_worker: float
    worker_rpc_timeout_s: float
    generation_stall_timeout_s: float
    # Opt-in per-request rollout: each engine runs its round-robin share of a
    # request's batches in one RPC and stages the payloads for a finalizer.
    pipelined: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.cpus_per_worker) or self.cpus_per_worker <= 0:
            raise ValueError("cpus_per_worker must be finite and > 0")
        if not math.isfinite(self.worker_rpc_timeout_s) or self.worker_rpc_timeout_s <= 0:
            raise ValueError("worker_rpc_timeout_s must be finite and > 0")
        if (
            not math.isfinite(self.generation_stall_timeout_s)
            or self.generation_stall_timeout_s <= 0
        ):
            raise ValueError("generation_stall_timeout_s must be finite and > 0")

    @classmethod
    def from_root(cls, root: RootConfig) -> RolloutWorkerConfig:
        """Freeze ``distributed.rollout`` of a validated root; an absent block means its defaults."""

        from vrl.config.schema import RolloutRuntimeSection

        distributed = root.distributed
        section = distributed.rollout if distributed is not None else None
        return cls(
            generation_stall_timeout_s=GENERATION_STALL_TIMEOUT_S,
            **(section or RolloutRuntimeSection()).model_dump(),
        )


__all__ = ["GENERATION_STALL_TIMEOUT_S", "RolloutWorkerConfig"]
