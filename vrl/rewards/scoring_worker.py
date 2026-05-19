"""Reward scorer worker semantics."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from vrl.ray.dependencies import current_gpu_ids, current_node_ip
from vrl.rewards.inference import (
    RewardInferenceRequest,
    RewardInferenceResult,
    select_score,
)


class RewardScoringWorker:
    """Load a reward scorer and score request shards."""

    def __init__(self, worker_id: str, worker_config: Mapping[str, Any]) -> None:
        self.worker_id = str(worker_id)
        self.worker_config = _validate_worker_config(worker_config)
        self._loaded = False

    def load_scorer(self) -> None:
        """Validate and mark scorer configuration loaded."""

        self._loaded = True

    def shutdown(self) -> None:
        """Release worker-owned state."""

        self._loaded = False

    def worker_metadata(self) -> dict[str, Any]:
        try:
            node_ip = current_node_ip()
            gpu_ids = current_gpu_ids()
        except Exception:
            node_ip = "local"
            gpu_ids = []
        return {
            "worker_id": self.worker_id,
            "node_ip": node_ip,
            "gpu_ids": gpu_ids,
            "reward_model_version": self.worker_config.get("reward_model_version", ""),
        }

    def score_batch(self, request: RewardInferenceRequest) -> list[RewardInferenceResult]:
        if not self._loaded:
            self.load_scorer()
        started = time.perf_counter()
        results: list[RewardInferenceResult] = []
        for artifact in request.artifacts:
            scores = self._score_artifact(artifact.path, request)
            selected = select_score(
                scores,
                request.score_key,
                score_aggregation=request.score_aggregation,
            )
            results.append(
                RewardInferenceResult(
                    artifact_id=artifact.artifact_id,
                    scores=scores,
                    selected_score=selected,
                    reward_name=request.reward_name,
                    score_key=request.score_key,
                    score_aggregation=request.score_aggregation,
                    policy_version=artifact.policy_version
                    if artifact.policy_version is not None
                    else request.policy_version,
                    reward_model_version=str(
                        self.worker_config.get("reward_model_version", ""),
                    ),
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    worker_id=self.worker_id,
                    metadata={"artifact_path": artifact.path},
                ),
            )
        return results

    def _score_artifact(
        self,
        artifact_path: str,
        request: RewardInferenceRequest,
    ) -> dict[str, float]:
        scorer = str(self.worker_config["scorer"])
        if scorer == "constant":
            scores = self.worker_config["scores"]
            return {str(key): float(value) for key, value in scores.items()}
        if scorer == "tensor_mean":
            score_key = request.score_key
            if "+" in score_key:
                score_key = request.score_key.split("+", 1)[0].strip()
            value = float(torch.load(Path(artifact_path), map_location="cpu").float().mean().item())
            return {score_key: value}
        raise ValueError(f"unsupported reward worker scorer={scorer!r}")


def _validate_worker_config(worker_config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(worker_config, Mapping):
        raise TypeError("reward worker_config must be a mapping")
    config = dict(worker_config)
    scorer = str(config.get("scorer", ""))
    if scorer not in {"constant", "tensor_mean"}:
        raise ValueError(
            "reward worker_config.scorer must be one of: constant, tensor_mean",
        )
    if scorer == "constant":
        scores = config.get("scores")
        if not isinstance(scores, Mapping) or not scores:
            raise ValueError("constant reward worker requires non-empty scores mapping")
        config["scores"] = {str(key): float(value) for key, value in scores.items()}
    return config


RewardInferenceWorker = RewardScoringWorker


__all__ = ["RewardInferenceWorker", "RewardScoringWorker"]
