"""Generation runtime and batch executor protocols.

Each protocol here corresponds to one physical or ownership boundary of the
engine — none exists for taste. One request flows as::

    driver process                        |  Ray worker process (GPU + model)
                                          |
    collector -> GenerationRuntime        |
                 split into batches,       |
                 dispatch to workers -----|-> GenerationBatchExecutor
                                          |     .forward_batch(batch)
                 batch results return <---|--- BatchPayload (family-owned shape)
                 GenerationBatchGatherer            |
                 .gather_batches()         |
                 reassemble full output   |

- ``GenerationRuntime``: the engine's only face toward vrl/rollouts (the dual
  of ``RewardRuntime``); the collector isinstance-checks it so orchestration
  never imports Ray code.
- ``GenerationBatchExecutor``: the model-family plugin contract (wan, sana,
  cosmos, ...); keeps ``if family == ...`` branches out of neutral execution.
- ``GenerationBatchGatherer``: the model-free slice of the executor, split out because
  reassembly runs driver-side where no model is loaded; it ships across the
  Ray launch contract as a serializable object.
- ``BatchSizeProbeExecutor``: an optional capability flag ("can you execute a
  truncated batch so the worker can measure peak memory for automatic
  samples_per_generation_batch sizing"), probed by isinstance like the reward side's
  ``MemoryParkingScorer``. Only diffusion families can truncate meaningfully:
  their memory peaks in the first denoise steps, while AR peaks at the last
  token, so a truncated AR run would measure a lie.
- ``BatchPayload``: deliberately ``Any`` — the batch payload's shape is owned by
  the binding that produced it (diffusion latents vs AR token results share no
  useful common structure); the alias documents that ownership in signatures.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vrl.generation.execution.sample_batches import GenerationSampleBatch
    from vrl.generation.types import (
        GenerationOutput,
        GenerationRequest,
        GenerationSampleRow,
    )


BatchPayload = Any


@runtime_checkable
class GenerationBatchGatherer(Protocol):
    """Pure batch gather contract that does not require an executor/model."""

    def gather_batches(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        batches: Sequence[BatchPayload],
    ) -> GenerationOutput: ...


@runtime_checkable
class GenerationRuntime(Protocol):
    """Generation runtime consumed by rollout collectors.

    The runtime is a transport boundary: schedules explicitly activate it before
    generation and offload it at a shared-GPU handoff. Whether to offload before
    reward scoring is no longer the runtime's decision — that is derived once
    from GPU topology into the
    ``RayLifecyclePlan`` and read by the collector (see vrl/ray/resources.py).
    """

    current_policy_version: int | None

    @property
    def requires_driver_model_offload(self) -> bool:
        """Whether shared ownership requires parking trainer state for generation."""
        ...

    async def preflight(self) -> None:
        """Fail fast on an unhealthy engine before the schedule starts.

        The generation twin of ``RewardRuntime.preflight``: a live fleet must
        answer one bounded health probe per worker; deferred sessions have
        nothing to probe yet (their launch validates itself).
        """
        ...

    async def activate(self) -> None:
        """Make generation ready and complete any policy install staged while inactive."""
        ...

    async def generate(self, request: GenerationRequest) -> GenerationOutput: ...

    async def offload(self) -> None:
        """Yield GPU memory after the schedule has drained generation.

        Resident runtimes may implement this as a no-op. Shared-GPU runtimes park
        physical GPU memory at a handoff and restore it during the next activate;
        terminal actor destruction remains the responsibility of shutdown().
        """
        ...

    async def shutdown(self) -> None:
        """Close admission and release every runtime-owned worker/resource."""
        ...


@runtime_checkable
class GenerationBatchExecutor(Protocol):
    """Family-specific distributed batch executor."""

    family: str
    task: str

    def forward_batch(
        self,
        request: GenerationRequest,
        batch: GenerationSampleBatch,
    ) -> BatchPayload: ...

    def gather_batches(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        batches: Sequence[BatchPayload],
    ) -> GenerationOutput: ...


@runtime_checkable
class BatchSizeProbeExecutor(Protocol):
    """Optional capability: truncated-batch execution for automatic sizing.

    ``samples_per_generation_batch: auto`` makes the worker trial-run a few denoise steps
    per candidate width to measure peak memory. Only executors whose memory
    peaks early under truncation (diffusion) implement this; the worker
    isinstance-probes it and requires an explicit width otherwise.
    """

    def forward_probe_batch(
        self,
        request: GenerationRequest,
        batch: GenerationSampleBatch,
        *,
        execute_steps: int,
    ) -> BatchPayload: ...


__all__ = [
    "BatchPayload",
    "BatchSizeProbeExecutor",
    "GenerationBatchExecutor",
    "GenerationBatchGatherer",
    "GenerationRuntime",
]
