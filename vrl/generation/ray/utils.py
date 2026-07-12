"""Shared validation helpers for Ray generation boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.generation.execution.types import (
    ChunkExecutionEnvelope,
    ChunkExecutionResult,
    DistributedWorkerHandle,
)
from vrl.generation.protocols import ChunkGatherer
from vrl.generation.ray.config import RayGenerationConfig
from vrl.ray.dependencies import current_node_ip
from vrl.ray.placement import validate_actor_gpu_ids


def require_chunk_gatherer(gatherer: Any) -> ChunkGatherer:
    """Require the collector-facing chunk-gatherer protocol."""

    gather_chunks = getattr(gatherer, "gather_chunks", None)
    if not callable(gather_chunks):
        raise TypeError(
            f"{type(gatherer).__name__} does not implement gather_chunks(...)",
        )
    return gatherer


def all_workers_support_versioned_slots(
    ray: Any,
    workers: list[DistributedWorkerHandle],
    *,
    weight_sync: Any | None,
) -> bool:
    """Return whether every worker supports versioned trainable-state slots.

    Non-draining weight sync needs slots on all workers because a chunk stamped
    with an older policy version can be placed on any worker. A missing weight
    syncer, an empty worker set, or any failed capability query keeps the safe
    draining barrier.
    """

    if weight_sync is None:
        return False
    actors = [worker.actor for worker in workers if worker.actor is not None]
    if not actors:
        return False
    try:
        results = ray.get(
            [actor.supports_versioned_trainable_state.remote() for actor in actors],
        )
    except Exception:
        return False
    return bool(results) and all(bool(result) for result in results)


def validate_worker_gpu_ids(
    config: RayGenerationConfig,
    metadata: list[Mapping[str, Any]],
    *,
    expected_gpu_ids: tuple[int, ...] | None = None,
) -> None:
    """Validate launched workers against the resolved rollout placement."""

    resources = config.resources
    if resources is None or resources.rollout_gpus_per_worker <= 0:
        return

    driver_node_ip: str | None = None
    if resources.cross_node:
        try:
            driver_node_ip = current_node_ip()
        except Exception:
            driver_node_ip = None

    # The placement owner supplies the role's expected GPUs (empty under
    # cross-node, where the node-aware check applies instead).
    expected = resources.rollout_devices if expected_gpu_ids is None else expected_gpu_ids
    validate_actor_gpu_ids(
        metadata,
        expected_gpu_ids=expected,
        role="generation",
        cross_node=resources.cross_node,
        driver_node_ip=driver_node_ip,
    )


def require_installed_policy_version(
    *,
    worker_id: str,
    installed: Any,
    expected: int,
) -> None:
    """Require a worker ACK for the expected installed policy version."""

    try:
        installed_version = int(installed)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"worker {worker_id!r} returned invalid installed policy version {installed!r}",
        ) from exc
    if installed_version != int(expected):
        raise RuntimeError(
            f"worker {worker_id!r} installed policy version {installed_version}, "
            f"expected {int(expected)}",
        )


def require_correlated_result(
    result: ChunkExecutionResult,
    envelope_by_chunk_key: dict[str, ChunkExecutionEnvelope],
) -> ChunkExecutionEnvelope:
    """Require a worker result to match a submitted request and chunk."""

    envelope = envelope_by_chunk_key.get(result.chunk.chunk_key)
    if envelope is None:
        raise RuntimeError(
            "distributed rollout returned an unknown chunk "
            f"(worker_id={result.worker_id}, chunk={result.chunk.chunk_key})",
        )
    expected_request_id = envelope.request.request_id
    if result.request_id != expected_request_id:
        raise RuntimeError(
            "distributed rollout request_id mismatch "
            f"(worker_id={result.worker_id}, chunk={result.chunk.chunk_key}, "
            f"expected={expected_request_id!r}, actual={result.request_id!r})",
        )
    return envelope


def is_oom_error(message: str) -> bool:
    """Match CUDA/HIP allocator failures flattened to text by a worker."""

    return "out of memory" in message.lower()


__all__ = [
    "all_workers_support_versioned_slots",
    "is_oom_error",
    "require_chunk_gatherer",
    "require_correlated_result",
    "require_installed_policy_version",
    "validate_worker_gpu_ids",
]
