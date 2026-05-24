"""Reward scorer worker semantics."""

from __future__ import annotations

import time
from collections.abc import Mapping
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

import torch

from vrl.ray.dependencies import current_gpu_ids, current_node_ip, import_from_path
from vrl.rewards.inference import (
    RewardInferenceArtifact,
    RewardInferenceRequest,
    RewardInferenceResult,
    select_score,
)


class RewardScoringWorker:
    """Load a reward scorer and score request shards."""

    def __init__(self, worker_id: str, worker_config: Mapping[str, Any]) -> None:
        self.worker_id = str(worker_id)
        self.worker_config = _validate_worker_config(worker_config)
        self._scorer_callable: Any | None = None
        self._loaded = False

    def load_scorer(self) -> None:
        """Validate and mark scorer configuration loaded."""

        if self.worker_config["scorer"] == "import_path":
            self._scorer_callable = import_from_path(str(self.worker_config["import_path"]))
        self._loaded = True

    def shutdown(self) -> None:
        """Release worker-owned state."""

        self._loaded = False

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
        if not self._loaded:
            self.load_scorer()
        batch_started_ns = time.perf_counter_ns()
        queued_at_ns = request.metadata.get("ray_queued_at_ns")
        queue_wait_ms = (
            (batch_started_ns - int(queued_at_ns)) / 1_000_000.0
            if queued_at_ns is not None
            else 0.0
        )
        worker_metadata = self.worker_metadata()
        results: list[RewardInferenceResult] = []
        for artifact in request.artifacts:
            artifact_started = time.perf_counter()
            scores = self._score_artifact(artifact, request)
            inference_ms = (time.perf_counter() - artifact_started) * 1000.0
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
                    latency_ms=queue_wait_ms + inference_ms,
                    queue_wait_ms=queue_wait_ms,
                    inference_ms=inference_ms,
                    worker_id=self.worker_id,
                    metadata={
                        "artifact_path": artifact.path,
                        "worker": worker_metadata,
                        "gpu_ids": worker_metadata.get("gpu_ids", []),
                        "node_ip": worker_metadata.get("node_ip", ""),
                    },
                ),
            )
        return results

    def _score_artifact(
        self,
        artifact: RewardInferenceArtifact,
        request: RewardInferenceRequest,
    ) -> dict[str, float]:
        scorer = str(self.worker_config["scorer"])
        if scorer == "constant":
            scores = self.worker_config["scores"]
            return {str(key): float(value) for key, value in scores.items()}
        if scorer == "tensor_mean":
            value = tensor_mean_scorer(artifact_path=artifact.path)
            score_key = _primary_score_key(request.score_key)
            return {score_key: value}
        if scorer == "import_path":
            scores = _call_import_path_scorer(
                self._scorer_callable,
                artifact=artifact,
                request=request,
                worker_config=self.worker_config,
            )
            if not isinstance(scores, Mapping):
                raise TypeError("import_path reward scorer must return a mapping of scores")
            return {str(key): float(value) for key, value in scores.items()}
        raise ValueError(f"unsupported reward worker scorer={scorer!r}")


def tensor_mean_scorer(*, artifact_path: str, **_: Any) -> float:
    """Small scorer used by smoke tests and in-process tensor fixtures."""

    return float(torch.load(Path(artifact_path), map_location="cpu").float().mean().item())


def _primary_score_key(score_key: str) -> str:
    if "+" in score_key:
        return score_key.split("+", 1)[0].strip()
    return score_key


def _call_import_path_scorer(
    scorer: Any,
    *,
    artifact: RewardInferenceArtifact,
    request: RewardInferenceRequest,
    worker_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    if scorer is None or not callable(scorer):
        raise TypeError("import_path reward scorer did not load a callable")
    try:
        sig = signature(scorer)
    except (TypeError, ValueError):
        return scorer(artifact.path)
    accepts_kwargs = any(
        parameter.kind == Parameter.VAR_KEYWORD for parameter in sig.parameters.values()
    )
    kwargs = {
        "artifact": artifact,
        "artifact_path": artifact.path,
        "request": request,
        "worker_config": worker_config,
    }
    if accepts_kwargs:
        return scorer(**kwargs)
    filtered = {key: value for key, value in kwargs.items() if key in sig.parameters}
    if filtered:
        return scorer(**filtered)
    return scorer(artifact.path)


def _validate_worker_config(worker_config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(worker_config, Mapping):
        raise TypeError("reward worker_config must be a mapping")
    config = dict(worker_config)
    scorer = str(config.get("scorer", ""))
    if scorer not in {"constant", "tensor_mean", "import_path"}:
        raise ValueError(
            "reward worker_config.scorer must be one of: constant, tensor_mean, import_path",
        )
    if scorer == "constant":
        scores = config.get("scores")
        if not isinstance(scores, Mapping) or not scores:
            raise ValueError("constant reward worker requires non-empty scores mapping")
        config["scores"] = {str(key): float(value) for key, value in scores.items()}
    if scorer == "import_path" and not str(config.get("import_path", "")).strip():
        raise ValueError("import_path reward worker requires worker_config.import_path")
    return config


RewardInferenceWorker = RewardScoringWorker


__all__ = ["RewardInferenceWorker", "RewardScoringWorker", "tensor_mean_scorer"]
