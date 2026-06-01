"""Reward inference contract."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

# Protocol-level set of valid artifact media kinds. Single source of truth —
# the disk artifact store (vrl.rewards.artifacts) imports this rather than
# re-listing the literals.
MEDIA_TYPES = frozenset({"image", "video", "tensor"})


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
    # Optional in-memory payload (e.g. an image/video tensor). The Ray transport
    # ships file paths; the local transport can pass media in-memory to avoid disk.
    media: Any = None

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("RewardInferenceArtifact.artifact_id is required")
        if not self.path and self.media is None:
            raise ValueError(
                "RewardInferenceArtifact requires a materialized path or in-memory media",
            )
        if self.media_type not in MEDIA_TYPES:
            raise ValueError(
                "RewardInferenceArtifact.media_type must be image, video, or tensor",
            )

    def as_media(self) -> Any:
        """Return in-memory media, loading a ``.pt`` tensor from ``path`` if needed."""

        if self.media is not None:
            return self.media
        if self.path.endswith(".pt"):
            import torch

            return torch.load(self.path, map_location="cpu", weights_only=True)
        raise ValueError(
            f"reward artifact {self.artifact_id!r} has no in-memory media and "
            f"path is not a loadable tensor: {self.path!r}",
        )

    def as_path(self) -> str:
        """Return the materialized file path (required by file-based models)."""

        if self.path:
            return self.path
        raise ValueError(
            f"reward artifact {self.artifact_id!r} has no materialized path; "
            "use inference_runtime='ray' or materialize the artifact first",
        )


class _ScoreSelection:
    score_key: str
    score_aggregation: str

    def score_keys(self) -> tuple[str, ...]:
        """Return the scalar score names requested by this inference object."""

        keys = tuple(part.strip() for part in str(self.score_key).split("+") if part.strip())
        if not keys:
            raise ValueError(f"invalid reward score_key: {self.score_key!r}")
        return keys

    def select_score(self, scores: Mapping[str, Any]) -> float:
        """Select and aggregate the training score from a score dictionary."""

        keys = self.score_keys()
        missing = [key for key in keys if key not in scores]
        if missing:
            raise KeyError(
                "reward inference result missing score keys: "
                f"missing={missing}, requested={self.score_key!r}, available={sorted(scores)}",
            )
        if self.score_aggregation != "sum":
            raise ValueError(f"unsupported score_aggregation={self.score_aggregation!r}")
        value = float(sum(float(scores[key]) for key in keys))
        if not math.isfinite(value):
            raise ValueError(
                f"reward score_key={self.score_key!r} selected non-finite score: {value}",
            )
        return value


@dataclass(frozen=True, slots=True)
class RewardInferenceRequest(_ScoreSelection):
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
        self.score_keys()
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

    def with_metadata(self, values: Mapping[str, Any]) -> RewardInferenceRequest:
        """Return a copy with additional request metadata."""

        metadata = dict(self.metadata)
        metadata.update(dict(values))
        return replace(self, metadata=metadata)


@dataclass(frozen=True, slots=True)
class RewardInferenceResult(_ScoreSelection):
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
    queue_wait_ms: float | None = None
    inference_ms: float | None = None
    worker_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("RewardInferenceResult.artifact_id is required")
        if self.error:
            return
        expected = self.select_score(self.scores)
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
        if self.queue_wait_ms is not None and float(self.queue_wait_ms) < 0:
            raise ValueError("RewardInferenceResult.queue_wait_ms must be non-negative")
        if self.inference_ms is not None and float(self.inference_ms) < 0:
            raise ValueError("RewardInferenceResult.inference_ms must be non-negative")


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


def score_artifacts_with_model(
    model: Any,
    request: RewardInferenceRequest,
    *,
    worker_id: str,
    reward_model_version: str = "",
    queue_wait_ms: float = 0.0,
    extra_metadata: Mapping[str, Any] | None = None,
) -> list[RewardInferenceResult]:
    """Run a ``RewardModel`` over a request's artifacts and build result rows.

    Shared by the in-process (local) runtime and the Ray worker so both
    transports score artifacts through identical logic. A model may expose a
    ``score_request(request) -> list[Mapping]`` batch hook (aligned to
    ``request.artifacts``); otherwise its per-artifact ``__call__`` is looped.
    """

    def build_result(
        artifact: RewardInferenceArtifact,
        raw_scores: Mapping[str, Any],
        inference_ms: float,
    ) -> RewardInferenceResult:
        if not isinstance(raw_scores, Mapping):
            raise TypeError("reward model must return a mapping of scores")
        scores = {str(key): float(value) for key, value in raw_scores.items()}
        selected = request.select_score(scores)
        metadata: dict[str, Any] = {"artifact_path": artifact.path}
        if extra_metadata:
            metadata.update(dict(extra_metadata))
        return RewardInferenceResult(
            artifact_id=artifact.artifact_id,
            scores=scores,
            selected_score=selected,
            reward_name=request.reward_name,
            score_key=request.score_key,
            score_aggregation=request.score_aggregation,
            policy_version=artifact.policy_version
            if artifact.policy_version is not None
            else request.policy_version,
            reward_model_version=str(reward_model_version),
            latency_ms=queue_wait_ms + inference_ms,
            queue_wait_ms=queue_wait_ms,
            inference_ms=inference_ms,
            worker_id=worker_id,
            metadata=metadata,
        )

    batch_score = getattr(model, "score_request", None)
    if callable(batch_score):
        started = time.perf_counter()
        score_maps = list(batch_score(request))
        if len(score_maps) != len(request.artifacts):
            raise ValueError(
                "RewardModel.score_request returned wrong number of score maps: "
                f"got {len(score_maps)}, expected {len(request.artifacts)}",
            )
        per_artifact_ms = (time.perf_counter() - started) * 1000.0 / max(1, len(score_maps))
        return [
            build_result(artifact, raw_scores, inference_ms=per_artifact_ms)
            for artifact, raw_scores in zip(request.artifacts, score_maps, strict=True)
        ]

    results: list[RewardInferenceResult] = []
    for artifact in request.artifacts:
        started = time.perf_counter()
        raw_scores = model(artifact=artifact, request=request)
        inference_ms = (time.perf_counter() - started) * 1000.0
        results.append(build_result(artifact, raw_scores, inference_ms))
    return results


__all__ = [
    "MEDIA_TYPES",
    "RewardInferenceArtifact",
    "RewardInferenceRequest",
    "RewardInferenceResult",
    "RewardInferenceRuntime",
    "score_artifacts_with_model",
    "shard_reward_request",
    "validate_reward_results",
]
