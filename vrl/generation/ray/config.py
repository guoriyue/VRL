"""Ray generation worker config."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vrl.config.schema import RolloutRuntimeSection, RootConfig


@dataclass(frozen=True, slots=True)
class RolloutWorkerConfig:
    """Frozen runtime projection of the public rollout-worker section."""

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
    def from_public_section(cls, section: RolloutRuntimeSection | None) -> RolloutWorkerConfig:
        """Freeze the validated public section; an absent block means its defaults."""

        from vrl.config.schema import RolloutRuntimeSection

        return cls(**(section or RolloutRuntimeSection()).model_dump())

    @classmethod
    def from_root(cls, root: RootConfig) -> RolloutWorkerConfig:
        """Freeze ``distributed.rollout`` of a validated root."""

        distributed = root.distributed
        return cls.from_public_section(distributed.rollout if distributed is not None else None)


__all__ = ["RolloutWorkerConfig"]
