"""Reward inference contract.

The transport seam below ``RewardFunction``: ``RewardScorer`` is the reward
dual of the generation engine's Ray executor layer, with two implementations —
``InProcessRewardScorer`` (runtime.py) and ``HttpRewardScorer``
(service/client.py). This module owns the request/result dataclasses both
transports exchange; the service wire format derives its field sets from them,
so these dataclasses are the single schema source. It stays importable without
torch or aiohttp (torch loads lazily inside ``as_media``) because the wire
module, the artifact store, and the standalone service's launch parsing all
need it. ``MemoryParkingScorer`` is an isinstance-probed optional capability,
the reward twin of ``BatchSizeProbeExecutor``.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, get_args, runtime_checkable

if TYPE_CHECKING:
    from vrl.rewards.models.base import RewardModel

# Valid artifact media kinds. ``MediaType`` is the single source of truth; the
# disk artifact store (vrl.rewards.artifacts) imports MEDIA_TYPES, and both it
# and the runtime check derive from this type instead of re-listing the literals.
MediaType = Literal["image", "video"]
MEDIA_TYPES = frozenset(get_args(MediaType))


def sha256_file(path: str | Path) -> str:
    """Canonical ``RewardInferenceArtifact.sha256`` digest of one file.

    The artifact writer (vrl.rewards.artifacts) and the reward service's wire
    validator must hash identically or shared-filesystem integrity checks fail;
    this is the one implementation both sides use.
    """

    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@dataclass(frozen=True, slots=True)
class RewardWorkerLaunchContract:
    """Runtime-owned keys parsed once from a reward worker config.

    The reward twin of ``GenerationRuntimeLaunchContract``: these typed fields
    are the closed set the runtime itself branches on. The verbatim mapping
    still reaches the model factory as its open plugin bag (models read their
    own keys — model names, thresholds, debug dirs — which are genuinely
    unvalidated user input).
    """

    model_factory: str
    device: str
    sleep_offload: bool
    memory_parking_residual_bytes_limit: int
    reward_model_name: str
    reward_model_version: str
    worker_config: Mapping[str, Any]

    @classmethod
    def from_worker_config(
        cls,
        worker_config: Mapping[str, Any] | None,
    ) -> RewardWorkerLaunchContract:
        cfg = dict(worker_config or {})
        residual_limit = int(cfg.get("memory_parking_residual_bytes_limit", 0))
        if residual_limit < 0:
            raise ValueError("reward memory parking residual limit must be >= 0")
        return cls(
            model_factory=str(cfg.get("model_factory", "")).strip(),
            device=str(cfg.get("device", "")),
            sleep_offload=bool(cfg.get("sleep_offload", False)),
            memory_parking_residual_bytes_limit=residual_limit,
            reward_model_name=str(cfg.get("reward_model_name", "")).strip(),
            reward_model_version=str(cfg.get("reward_model_version", "")).strip(),
            worker_config=cfg,
        )


@dataclass(frozen=True, slots=True)
class RewardInferenceArtifact:
    """Stable artifact consumed by reward inference workers."""

    artifact_id: str
    path: str
    prompt: str = ""
    size_bytes: int | None = None
    sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Optional in-memory payload (e.g. an image/video tensor). The HTTP transport
    # ships file references; the in-process runtime can avoid materialization.
    media: Any = None

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("RewardInferenceArtifact.artifact_id is required")
        if not self.path and self.media is None:
            raise ValueError(
                "RewardInferenceArtifact requires a materialized path or in-memory media",
            )
        if (self.size_bytes is None) != (self.sha256 is None):
            raise ValueError(
                "RewardInferenceArtifact.size_bytes and sha256 must be set together",
            )
        if not self.path and self.size_bytes is not None:
            raise ValueError(
                "RewardInferenceArtifact in-memory media cannot declare file integrity",
            )
        if self.size_bytes is not None and int(self.size_bytes) < 0:
            raise ValueError("RewardInferenceArtifact.size_bytes must be >= 0")
        if self.sha256 is not None and (
            len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError(
                "RewardInferenceArtifact.sha256 must be a lowercase hex SHA-256 digest",
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


@dataclass(frozen=True, slots=True)
class RewardInferenceRequest:
    """A batch of stable artifacts to score."""

    request_id: str
    artifacts: tuple[RewardInferenceArtifact, ...]

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("RewardInferenceRequest.request_id is required")
        artifacts = tuple(self.artifacts)
        if not all(isinstance(artifact, RewardInferenceArtifact) for artifact in artifacts):
            raise TypeError(
                "RewardInferenceRequest.artifacts must contain RewardInferenceArtifact values",
            )
        artifact_ids = [artifact.artifact_id for artifact in artifacts]
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError(f"duplicate reward artifact ids: {artifact_ids}")
        object.__setattr__(self, "artifacts", artifacts)


@dataclass(frozen=True, slots=True)
class RewardInferenceResult:
    """One artifact's scored reward result."""

    artifact_id: str
    scores: dict[str, float]
    # display/provenance-only: which reward model scored this; wire + debug JSONL.
    reward_model_version: str | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("RewardInferenceResult.artifact_id is required")
        scores = {str(name): float(value) for name, value in self.scores.items()}
        if not all(math.isfinite(value) for value in scores.values()):
            raise ValueError(
                f"reward result {self.artifact_id!r} scores must be finite",
            )
        timing_ms = {str(name): float(value) for name, value in self.timing_ms.items()}
        for field_name, value in timing_ms.items():
            normalized = float(value)
            if not math.isfinite(normalized) or normalized < 0:
                raise ValueError(
                    f"RewardInferenceResult timing {field_name!r} must be finite and non-negative",
                )
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "timing_ms", timing_ms)


class RewardScorer(Protocol):
    """Runtime boundary for model-backed reward inference."""

    @property
    def scoring_is_nonblocking(self) -> bool:
        """Whether scoring yields the caller while model work runs elsewhere."""
        ...

    @property
    def external_accelerator_isolation_verified(self) -> bool:
        """Whether accelerator work outside the resource plan is isolated."""
        ...

    async def score_batch(
        self,
        request: RewardInferenceRequest,
    ) -> list[RewardInferenceResult]: ...

    async def shutdown(self) -> None: ...


@runtime_checkable
class MemoryParkingScorer(Protocol):
    """Runtime boundary for retryable, verifiable reward GPU parking."""

    @property
    def requires_memory_parking(self) -> bool: ...

    async def activate(self) -> None:
        """Build or wake the model at a GPU handoff (inverse of park_memory)."""
        ...

    async def park_memory(self) -> None: ...


def validate_reward_results(
    request: RewardInferenceRequest,
    results: list[RewardInferenceResult],
) -> list[RewardInferenceResult]:
    """Validate one finite result per request artifact in original order."""

    expected_ids = [artifact.artifact_id for artifact in request.artifacts]
    by_id: dict[str, RewardInferenceResult] = {}
    for result in results:
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
    model: RewardModel,
    request: RewardInferenceRequest,
    *,
    reward_model_version: str = "",
) -> list[RewardInferenceResult]:
    """Run a ``RewardModel`` over a request's artifacts and build result rows.

    The in-process runtime's scoring core (the former Ray reward worker shared
    it; that transport is gone). A model may expose a
    ``score_batch(artifacts) -> list[Mapping]`` hook; otherwise its
    per-artifact ``__call__`` is looped.
    """

    def build_result(
        artifact: RewardInferenceArtifact,
        raw_scores: Mapping[str, Any],
        inference_ms: float,
    ) -> RewardInferenceResult:
        if not isinstance(raw_scores, Mapping):
            raise TypeError("reward model must return a mapping of scores")
        scores = {str(key): float(value) for key, value in raw_scores.items()}
        return RewardInferenceResult(
            artifact_id=artifact.artifact_id,
            scores=scores,
            reward_model_version=str(reward_model_version) or None,
            timing_ms={"inference_ms": inference_ms},
        )

    batch_score = getattr(model, "score_batch", None)
    if callable(batch_score):
        started = time.perf_counter()
        score_maps = list(batch_score(request.artifacts))
        if len(score_maps) != len(request.artifacts):
            raise ValueError(
                "RewardModel.score_batch returned wrong number of score maps: "
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
        raw_scores = model(artifact)
        inference_ms = (time.perf_counter() - started) * 1000.0
        results.append(build_result(artifact, raw_scores, inference_ms))
    return results


__all__ = [
    "MEDIA_TYPES",
    "MemoryParkingScorer",
    "RewardInferenceArtifact",
    "RewardInferenceRequest",
    "RewardInferenceResult",
    "RewardScorer",
    "RewardWorkerLaunchContract",
    "score_artifacts_with_model",
    "sha256_file",
    "validate_reward_results",
]
