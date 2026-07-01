"""Reward model worker: load a RewardModel and score request shards."""

from __future__ import annotations

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
            self.load_model()
        batch_started_ns = time.perf_counter_ns()
        queued_at_ns = request.metadata.get("ray_queued_at_ns")
        queue_wait_ms = (
            (batch_started_ns - int(queued_at_ns)) / 1_000_000.0
            if queued_at_ns is not None
            else 0.0
        )
        worker_metadata = self.worker_metadata()
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
