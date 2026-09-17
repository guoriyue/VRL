"""Public protocol types for the standalone reward service.

This module is intentionally transport-agnostic: HTTP clients, servers, and
operators can inspect the same version, capability, identity, and error types
without importing either endpoint implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from vrl.utils.validation import require_int

# The one protocol identifier: trainer client and standalone service can be
# deployed at different versions, so a mismatched peer must fail loudly (426)
# instead of silently misreading fields. v3 dropped the protocol string, the
# capability array, and the artifact-transport field; v4 adds the original
# sample identity to artifact provenance; v5 adds safe typed tensor uploads
# and removes the obsolete managed-child launch token. Fixed facts of the
# service are guaranteed by this version, not advertised per request.
WIRE_VERSION = 5


# Keep the exported enum's historical ``str(member)`` representation; the wire
# contract serializes ``.value`` explicitly and does not need StrEnum semantics.
class RewardServiceErrorCode(str, Enum):  # noqa: UP042
    """Stable machine-readable failures returned by the service."""

    BAD_REQUEST = "bad_request"
    CANCELLED = "cancelled"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    INTERNAL_ERROR = "internal_error"
    METHOD_NOT_ALLOWED = "method_not_allowed"
    NOT_FOUND = "not_found"
    NOT_READY = "not_ready"
    OVERLOADED = "overloaded"
    PATH_NOT_ALLOWED = "path_not_allowed"
    REQUEST_COMPLETED = "request_completed"
    REQUEST_NOT_FOUND = "request_not_found"
    SCORING_FAILED = "scoring_failed"
    SERVICE_SHUTTING_DOWN = "service_shutting_down"
    TRANSPORT_ERROR = "transport_error"
    UNSUPPORTED_VERSION = "unsupported_version"


@dataclass(frozen=True, slots=True)
class RewardServiceInfo:
    """Discoverable service identity and scheduling facts."""

    model_name: str
    model_version: str
    # The one genuinely deployment-dependent fact: whether the operator proved
    # this service's accelerators are isolated from the training topology, so
    # the collector may overlap reward N with generation N+1.
    generation_overlap_safe: bool
    max_concurrency: int
    max_pending_requests: int
    # Whether this service takes the shared-GPU phase lease: POST /park yields
    # its model's GPU memory while the trainer/rollout own the card, and
    # POST /wake restores it through the configured parking backend. A service that
    # parks is by construction NOT generation_overlap_safe.
    memory_parking: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.memory_parking, bool):
            raise ValueError("reward service memory_parking must be a boolean")
        if self.memory_parking and self.generation_overlap_safe:
            raise ValueError(
                "a reward service that parks its GPU memory shares that GPU and cannot "
                "be generation_overlap_safe",
            )
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("reward service model_name must be a non-empty string")
        if not isinstance(self.model_version, str):
            raise ValueError("reward service model_version must be a string")
        if not isinstance(self.generation_overlap_safe, bool):
            # bool("false") is True; a stringly wire value must fail, not flip.
            raise ValueError("reward service generation_overlap_safe must be a boolean")
        require_int(self.max_concurrency, path="reward service max_concurrency", minimum=1)
        require_int(
            self.max_pending_requests, path="reward service max_pending_requests", minimum=1
        )
        if self.max_pending_requests < self.max_concurrency:
            raise ValueError(
                "reward service max_pending_requests must be >= max_concurrency",
            )


class RewardServiceProtocolError(ValueError):
    """A client request violated the versioned wire contract."""

    def __init__(
        self,
        code: RewardServiceErrorCode,
        message: str,
        *,
        status_code: int = 400,
        retryable: bool = False,
        request_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = int(status_code)
        self.retryable = bool(retryable)
        self.request_id = request_id
        self.details = dict(details or {})


class RemoteRewardServiceError(RuntimeError):
    """Typed remote failure surfaced by ``HttpRewardScorer``."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        request_id: str | None = None,
        details: dict[str, Any] | None = None,
        retain_reward_artifacts: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.status_code = status_code
        self.retryable = bool(retryable)
        self.request_id = request_id
        self.details = dict(details or {})
        # A transport failure can leave remote scoring alive after the client
        # coroutine exits. The artifact owner must not delete shared files
        # until cancellation or completion has been confirmed.
        self.retain_reward_artifacts = bool(retain_reward_artifacts)


__all__ = [
    "WIRE_VERSION",
    "RemoteRewardServiceError",
    "RewardServiceErrorCode",
    "RewardServiceInfo",
    "RewardServiceProtocolError",
]
