"""Reward inference contract and factory."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class RewardInferenceArtifact:
    """Stable artifact consumed by reward inference workers."""

    artifact_id: str
    path: str
    media_type: str
    prompt: str = ""
    sample_id: str | None = None
    policy_version: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("RewardInferenceArtifact.artifact_id is required")
        if not self.path:
            raise ValueError("RewardInferenceArtifact.path is required")
        if self.media_type not in {"image", "video", "tensor"}:
            raise ValueError(
                "RewardInferenceArtifact.media_type must be image, video, or tensor",
            )


def split_score_key(score_key: str) -> tuple[str, ...]:
    """Split a scalar or composite score key."""

    keys = tuple(part.strip() for part in str(score_key).split("+") if part.strip())
    if not keys:
        raise ValueError(f"invalid reward score_key: {score_key!r}")
    return keys


def select_score(
    scores: dict[str, float],
    score_key: str,
    *,
    score_aggregation: str = "sum",
) -> float:
    """Select and aggregate the training score from a score dictionary."""

    keys = split_score_key(score_key)
    missing = [key for key in keys if key not in scores]
    if missing:
        raise KeyError(
            "reward inference result missing score keys: "
            f"missing={missing}, requested={score_key!r}, available={sorted(scores)}",
        )
    if score_aggregation != "sum":
        raise ValueError(f"unsupported score_aggregation={score_aggregation!r}")
    value = float(sum(float(scores[key]) for key in keys))
    if not math.isfinite(value):
        raise ValueError(f"reward score_key={score_key!r} selected non-finite score: {value}")
    return value


@dataclass(frozen=True, slots=True)
class RewardInferenceRequest:
    """A batch of stable artifacts to score."""

    request_id: str
    artifacts: tuple[RewardInferenceArtifact, ...]
    reward_name: str
    score_key: str
    score_aggregation: str = "sum"
    policy_version: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("RewardInferenceRequest.request_id is required")
        if not self.reward_name:
            raise ValueError("RewardInferenceRequest.reward_name is required")
        split_score_key(self.score_key)
        if self.score_aggregation != "sum":
            raise ValueError("RewardInferenceRequest.score_aggregation only supports 'sum'")
        artifact_ids = [artifact.artifact_id for artifact in self.artifacts]
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError(f"duplicate reward artifact ids: {artifact_ids}")

    def with_artifacts(
        self,
        artifacts: tuple[RewardInferenceArtifact, ...],
        *,
        shard_index: int | None = None,
    ) -> RewardInferenceRequest:
        metadata = dict(self.metadata)
        if shard_index is not None:
            metadata["shard_index"] = int(shard_index)
        return replace(self, artifacts=artifacts, metadata=metadata)


@dataclass(frozen=True, slots=True)
class RewardInferenceResult:
    """One artifact's scored reward result."""

    artifact_id: str
    scores: dict[str, float]
    selected_score: float
    reward_name: str
    score_key: str
    score_aggregation: str = "sum"
    policy_version: int | None = None
    reward_model_version: str | None = None
    latency_ms: float | None = None
    worker_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("RewardInferenceResult.artifact_id is required")
        if self.error:
            return
        expected = select_score(
            self.scores,
            self.score_key,
            score_aggregation=self.score_aggregation,
        )
        if not math.isfinite(float(self.selected_score)):
            raise ValueError(
                f"reward result {self.artifact_id!r} selected_score is non-finite",
            )
        if abs(float(self.selected_score) - expected) > 1e-6:
            raise ValueError(
                f"reward result {self.artifact_id!r} selected_score mismatch: "
                f"{self.selected_score} != {expected}",
            )
        if self.latency_ms is not None and float(self.latency_ms) < 0:
            raise ValueError("RewardInferenceResult.latency_ms must be non-negative")


class RewardInferenceRuntime(Protocol):
    """Runtime boundary for model-backed reward inference."""

    async def score_batch(
        self,
        request: RewardInferenceRequest,
    ) -> list[RewardInferenceResult]: ...

    async def shutdown(self) -> None: ...


def shard_reward_request(
    request: RewardInferenceRequest,
    *,
    num_shards: int,
) -> list[RewardInferenceRequest]:
    """Split a reward request into artifact-count-balanced shards."""

    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not request.artifacts:
        return []

    buckets = [[] for _ in range(min(num_shards, len(request.artifacts)))]
    for index, artifact in enumerate(request.artifacts):
        buckets[index % len(buckets)].append(artifact)
    return [
        request.with_artifacts(tuple(bucket), shard_index=shard_index)
        for shard_index, bucket in enumerate(buckets)
        if bucket
    ]


def validate_reward_results(
    request: RewardInferenceRequest,
    results: list[RewardInferenceResult],
) -> list[RewardInferenceResult]:
    """Validate one finite result per request artifact in original order."""

    expected_ids = [artifact.artifact_id for artifact in request.artifacts]
    by_id: dict[str, RewardInferenceResult] = {}
    for result in results:
        if result.error:
            raise RuntimeError(
                f"reward inference failed for artifact {result.artifact_id}: {result.error}",
            )
        if result.artifact_id in by_id:
            raise RuntimeError(f"duplicate reward result for artifact {result.artifact_id}")
        by_id[result.artifact_id] = result
    missing = [artifact_id for artifact_id in expected_ids if artifact_id not in by_id]
    extra = sorted(set(by_id) - set(expected_ids))
    if missing or extra:
        raise RuntimeError(
            "reward inference result/artifact mismatch: "
            f"missing={missing}, extra={extra}",
        )
    return [by_id[artifact_id] for artifact_id in expected_ids]


def build_reward_inference_runtime(
    cfg: Mapping[str, Any],
    *,
    init_ray: bool = True,
    ray_init_kwargs: dict[str, Any] | None = None,
) -> RewardInferenceRuntime:
    """Build the reward inference runtime selected by config."""

    runtime = str(cfg.get("inference_runtime", ""))
    if runtime != "ray":
        raise ValueError("reward inference_runtime must be 'ray'")
    worker_config = cfg.get("worker_config")
    if worker_config is None:
        raise ValueError(
            "reward inference_runtime='ray' requires reward.kwargs.video_reward.worker_config",
        )
    if not isinstance(worker_config, Mapping):
        raise TypeError("reward worker_config must be a mapping")

    from vrl.rewards.ray.launcher import build_reward_ray_runtime

    return build_reward_ray_runtime(
        cfg,
        init_ray=init_ray,
        ray_init_kwargs=ray_init_kwargs,
    )


__all__ = [
    "RewardInferenceArtifact",
    "RewardInferenceRequest",
    "RewardInferenceResult",
    "RewardInferenceRuntime",
    "build_reward_inference_runtime",
    "select_score",
    "shard_reward_request",
    "split_score_key",
    "validate_reward_results",
]
