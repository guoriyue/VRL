"""Reward inference request/result schema.

The dataclasses both scorer transports exchange — ``InProcessRewardScorer``
(runtime.py) and ``HttpRewardScorer`` (service/client.py). The service wire
format derives its field sets from them, so these dataclasses are the single
schema source. Stays importable without torch or aiohttp (torch loads lazily
inside ``as_media``) because the wire module and the artifact store need it.
The scorer capability protocols live in protocols.py, the artifact stores in
artifacts.py, and the worker launch contract in launch_contract.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RewardInferenceArtifact:
    """Stable artifact consumed by reward inference workers."""

    artifact_id: str
    # Display/audit provenance: preserves the rollout identity across artifact
    # materialization and the remote reward-service boundary.
    sample_id: str
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
        if not isinstance(self.sample_id, str):
            raise TypeError("RewardInferenceArtifact.sample_id must be a str")
        if not self.sample_id:
            raise ValueError("RewardInferenceArtifact.sample_id must be non-empty")
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

    def validate_and_order_results(
        self,
        results: list[RewardInferenceResult],
    ) -> list[RewardInferenceResult]:
        """Validate one finite result per artifact and restore request order.

        Every transport (in-process, HTTP client, HTTP server) runs scored
        results through this before handing them back, so a scorer that drops,
        duplicates, invents, or mistypes a result fails at the boundary it
        crossed. The request owns the check because the request defines the
        expected identity set and order.
        """

        expected_ids = [artifact.artifact_id for artifact in self.artifacts]
        by_id: dict[str, RewardInferenceResult] = {}
        for result in results:
            if not isinstance(result, RewardInferenceResult):
                raise TypeError(
                    "reward scorer returned a non-RewardInferenceResult value: "
                    f"{type(result).__name__}",
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


__all__ = [
    "RewardInferenceArtifact",
    "RewardInferenceRequest",
    "RewardInferenceResult",
]
