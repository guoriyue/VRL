"""Reward inference contract."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, get_args, runtime_checkable

# Valid artifact media kinds. ``MediaType`` is the single source of truth; the
# disk artifact store (vrl.rewards.artifacts) imports MEDIA_TYPES, and both it
# and the runtime check derive from this type instead of re-listing the literals.
MediaType = Literal["image", "video"]
MEDIA_TYPES = frozenset(get_args(MediaType))


@dataclass(frozen=True, slots=True)
class RewardInferenceArtifact:
    """Stable artifact consumed by reward inference workers."""

    artifact_id: str
    path: str
    media_type: MediaType
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
            allowed = ", ".join(get_args(MediaType))
            raise ValueError(
                f"RewardInferenceArtifact.media_type must be one of {allowed}; "
                f"got {self.media_type!r}",
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
            "materialize the artifact before calling a file-based reward model",
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


@dataclass(frozen=True, slots=True)
class RewardMemoryParkingSpec:
    """Pre-construction contract for a verifiable reward parking runtime."""

    residual_bytes_limit: int = 0

    def __post_init__(self) -> None:
        if self.residual_bytes_limit < 0:
            raise ValueError("reward parking residual_bytes_limit must be >= 0")


def validate_reward_parking_residual(
    *,
    residual_bytes: int,
    baseline_bytes: int,
    limit_bytes: int,
    context: str,
) -> None:
    """One invariant behind every reward release check: residual <= baseline + limit."""

    if min(residual_bytes, baseline_bytes, limit_bytes) < 0:
        raise ValueError(f"{context} byte counts must be >= 0")
    if residual_bytes > baseline_bytes + limit_bytes:
        raise RuntimeError(
            f"incomplete {context}: residual={residual_bytes} "
            f"baseline={baseline_bytes} limit={limit_bytes}",
        )


@dataclass(frozen=True, slots=True)
class RewardMemoryReleaseProof:
    """Evidence that one reward request finished with GPU memory parked."""

    request_id: str
    released: bool
    baseline_gpu_used_bytes: int = 0
    residual_gpu_used_bytes: int = 0
    residual_bytes_limit: int = 0

    def validate(self, *, request_id: str) -> None:
        if self.request_id != request_id:
            raise RuntimeError(
                "reward memory release proof belongs to a different request: "
                f"expected={request_id!r}, actual={self.request_id!r}",
            )
        if not self.released:
            raise RuntimeError(
                f"reward GPU memory was not released for request {request_id!r}",
            )
        validate_reward_parking_residual(
            residual_bytes=self.residual_gpu_used_bytes,
            baseline_bytes=self.baseline_gpu_used_bytes,
            limit_bytes=self.residual_bytes_limit,
            context=f"reward memory parking for request {request_id!r}",
        )


@runtime_checkable
class RewardMemoryParkingRuntime(Protocol):
    """Runtime boundary for retryable, verifiable reward GPU parking."""

    @property
    def requires_memory_parking(self) -> bool: ...

    async def park_memory(self) -> RewardMemoryReleaseProof: ...


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
            f"reward inference result/artifact mismatch: missing={missing}, extra={extra}",
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
    "RewardMemoryParkingRuntime",
    "RewardMemoryReleaseProof",
    "score_artifacts_with_model",
    "validate_reward_parking_residual",
    "validate_reward_results",
]
