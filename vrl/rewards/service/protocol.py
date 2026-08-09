"""Public protocol types for the standalone reward service.

This module is intentionally transport-agnostic: HTTP clients, servers, and
operators can inspect the same version, capability, identity, and error types
without importing either endpoint implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

# These constants are protocol identifiers, not duplicated application data.
WIRE_PROTOCOL = "vrl.reward"
WIRE_VERSION = 1
SHARED_FILESYSTEM_ARTIFACT_TRANSPORT = "shared_filesystem_paths"
GENERATION_OVERLAP_SAFE_CAPABILITY = "generation_overlap_safe"


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
    """Discoverable service identity and scheduling capabilities."""

    model_name: str
    # Display/provenance-only until the typed client config gains a separate
    # expected_model_version field; expected_model currently gates model_name.
    model_version: str
    capabilities: tuple[str, ...]
    max_concurrency: int
    max_pending_requests: int
    artifact_transport: str = SHARED_FILESYSTEM_ARTIFACT_TRANSPORT
    protocol: str = WIRE_PROTOCOL
    wire_version: int = WIRE_VERSION

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("reward service model_name is required")
        if self.protocol != WIRE_PROTOCOL or self.wire_version != WIRE_VERSION:
            raise ValueError("reward service info protocol/version is inconsistent")
        if not self.capabilities or any(not capability for capability in self.capabilities):
            raise ValueError("reward service capabilities must be non-empty strings")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("reward service capabilities must not contain duplicates")
        if self.max_concurrency < 1:
            raise ValueError("reward service max_concurrency must be >= 1")
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
    """Typed remote failure surfaced by ``HttpRewardInferenceRuntime``."""

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
    "GENERATION_OVERLAP_SAFE_CAPABILITY",
    "SHARED_FILESYSTEM_ARTIFACT_TRANSPORT",
    "WIRE_PROTOCOL",
    "WIRE_VERSION",
    "RemoteRewardServiceError",
    "RewardServiceErrorCode",
    "RewardServiceInfo",
    "RewardServiceProtocolError",
]
