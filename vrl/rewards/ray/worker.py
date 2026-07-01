"""Reward model worker: load a RewardModel and score request shards."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

from vrl.ray.dependencies import current_gpu_ids, current_node_ip, import_from_path
from vrl.rewards.inference import (
    RewardInferenceRequest,
    RewardInferenceResult,
    score_artifacts_with_model,
)
from vrl.utils.logging import init_logger, kv

logger = init_logger(__name__)


class RewardModelWorker:
    """Load one RewardModel via worker_config.model_factory and score shards."""

    def __init__(self, worker_id: str, worker_config: Mapping[str, Any]) -> None:
        self.worker_id = str(worker_id)
        self.worker_config = _validate_worker_config(worker_config)
        self._model: Any | None = None
        # Resident-reward parking (single-GPU staged pipeline): instead of the
        # actor being killed + rebuilt each step (~8s reload), keep it alive and
        # move the model GPU<->CPU around scoring (~1s). The model sits on CPU
        # during rollout/training (no GPU memory) and only moves to the GPU for
        # scoring, when the rollout has already released the GPU. Opt-in via
        # reward worker_config.resident; the model must implement .to(device).
        self._resident = bool(self.worker_config.get("resident", False))
        self._gpu_device: str | None = None
        if self._resident:
            # A resident reward reserves only a small CPU slice, but native decode
            # libraries can still spawn broad process-level thread pools and starve
            # the raylet's heartbeat. Cap the worker to a slice of the cores (set
            # before torch initializes its pools) so the raylet keeps cores. ~1/4
            # of the node.
            threads = max(1, (os.cpu_count() or 8) // 4)
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                os.environ[var] = str(threads)
            self._cpu_threads = threads
        else:
            self._cpu_threads = 0

    def load_model(self) -> None:
        """Import the configured factory and build the reward model."""

        factory_path = str(self.worker_config["model_factory"])
        started = time.perf_counter()
        logger.info(
            "reward worker loading model %s",
            kv(
                worker_id=self.worker_id,
                factory=factory_path,
                reward_model_version=str(
                    self.worker_config.get("reward_model_version", ""),
                ),
            ),
        )
        try:
            import_started = time.perf_counter()
            factory = import_from_path(factory_path)
            logger.info(
                "reward worker imported model factory %s",
                kv(
                    worker_id=self.worker_id,
                    factory=factory_path,
                    elapsed_s=time.perf_counter() - import_started,
                ),
            )

            build_started = time.perf_counter()
            self._model = factory(self.worker_config)
            # Remember where the model lives so resident parking can move it back
            # to the GPU for scoring (set here, not in score_batch, because
            # startup_method='load_model' loads at launch before any score).
            self._gpu_device = str(getattr(self._model, "device", "cuda"))
            if self._resident and self._cpu_threads:
                # Authoritative intra-op cap (env vars only apply at torch import);
                # keeps the resident reward's decode/preprocess off the raylet cores.
                try:
                    import torch

                    torch.set_num_threads(self._cpu_threads)
                except Exception:
                    pass
            logger.info(
                "reward worker built model %s",
                kv(
                    worker_id=self.worker_id,
                    factory=factory_path,
                    elapsed_s=time.perf_counter() - build_started,
                    total_s=time.perf_counter() - started,
                ),
            )
        except Exception as exc:
            logger.exception(
                "reward worker failed to load model worker_id=%s factory=%s",
                self.worker_id,
                factory_path,
            )
            raise RuntimeError(
                "reward worker failed to load model "
                f"worker_id={self.worker_id!r} factory={factory_path!r}",
            ) from exc

    def shutdown(self) -> None:
        """Release the loaded model."""

        self._model = None
        self._gpu_device = None

    def _can_park(self) -> bool:
        return (
            self._resident
            and self._model is not None
            and callable(getattr(self._model, "to", None))
            and self._gpu_device is not None
        )

    def _park(self) -> None:
        """Move the resident model to CPU so it holds no GPU memory between scores.

        ``.to('cpu')`` frees the model's tensors but PyTorch keeps the freed
        blocks in its CUDA caching allocator, so the reward process would still
        hold GPU memory the rollout/trainer need. empty_cache returns it.
        """
        if self._can_park():
            self._model.to("cpu")
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    def _unpark(self) -> None:
        """Move the resident model back to its GPU for scoring."""
        if self._can_park():
            self._model.to(self._gpu_device)

    def worker_metadata(self) -> dict[str, Any]:
        try:
            node_ip = current_node_ip()
            gpu_ids = current_gpu_ids()
        except Exception:
            node_ip = "unknown"
            gpu_ids = []
        return {
            "worker_id": self.worker_id,
            "node_ip": node_ip,
            "gpu_ids": gpu_ids,
            "reward_model_version": self.worker_config.get("reward_model_version", ""),
        }

    def score_batch(self, request: RewardInferenceRequest) -> list[RewardInferenceResult]:
        if self._model is None:
            # First load runs after the rollout has released the GPU (score_rollouts
            # offloads it first), so building on the GPU here is safe even in
            # resident mode; subsequent steps move it back from CPU via _unpark.
            self.load_model()
        else:
            self._unpark()
        batch_started_ns = time.perf_counter_ns()
        queued_at_ns = request.metadata.get("ray_queued_at_ns")
        queue_wait_ms = (
            (batch_started_ns - int(queued_at_ns)) / 1_000_000.0
            if queued_at_ns is not None
            else 0.0
        )
        worker_metadata = self.worker_metadata()
        try:
            return score_artifacts_with_model(
                self._model,
                request,
                worker_id=self.worker_id,
                reward_model_version=str(self.worker_config.get("reward_model_version", "")),
                queue_wait_ms=queue_wait_ms,
                extra_metadata={
                    "worker": worker_metadata,
                    "gpu_ids": worker_metadata.get("gpu_ids", []),
                    "node_ip": worker_metadata.get("node_ip", ""),
                },
            )
        finally:
            # Park back to CPU so the GPU is free for the next rollout/training step.
            self._park()


def _validate_worker_config(worker_config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(worker_config, Mapping):
        raise TypeError("reward worker_config must be a mapping")
    config = dict(worker_config)
    model_factory = str(config.get("model_factory", "")).strip()
    if not model_factory:
        raise ValueError(
            "reward worker_config.model_factory is required: an import path to a "
            "RewardModel factory (callable taking worker_config -> RewardModel)",
        )
    config["model_factory"] = model_factory
    return config


__all__ = ["RewardModelWorker"]
