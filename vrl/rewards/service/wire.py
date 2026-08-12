"""Versioned JSON wire format for standalone reward scoring.

The single owner of the envelope encode/decode pair: client.py and server.py
both import only these functions, so the two endpoints cannot drift apart.
Field sets derive from the inference.py dataclasses via ``fields(...)``, which
keeps those dataclasses the one schema source — adding a field changes the
wire, and unknown keys are rejected rather than ignored. The envelope pins
``WIRE_PROTOCOL``/``WIRE_VERSION`` so a mismatched peer fails before any
scoring, and ``request_fingerprint`` canonicalizes a request for the server's
idempotency check. In-memory media never crosses this boundary: remote
scoring requires disk-materialized artifacts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields
from typing import Any

from vrl.rewards.inference import (
    RewardInferenceArtifact,
    RewardInferenceRequest,
    RewardInferenceResult,
)
from vrl.rewards.service.protocol import (
    WIRE_PROTOCOL,
    WIRE_VERSION,
    RemoteRewardServiceError,
    RewardServiceErrorCode,
    RewardServiceInfo,
    RewardServiceProtocolError,
)


def _wire_envelope(**payload: Any) -> dict[str, Any]:
    return {
        "protocol": WIRE_PROTOCOL,
        "version": WIRE_VERSION,
        **payload,
    }


def _require_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RewardServiceProtocolError(
            RewardServiceErrorCode.BAD_REQUEST,
            f"{context} must be a JSON object",
        )
    return value


def _validate_envelope(payload: Any, *, expected_keys: set[str]) -> Mapping[str, Any]:
    envelope = _require_mapping(payload, context="reward service payload")
    _reject_unknown_keys(
        envelope,
        {"protocol", "version", *expected_keys},
        context="reward envelope",
    )
    protocol = envelope.get("protocol")
    if protocol != WIRE_PROTOCOL:
        raise RewardServiceProtocolError(
            RewardServiceErrorCode.BAD_REQUEST,
            f"unsupported reward protocol {protocol!r}; expected {WIRE_PROTOCOL!r}",
        )
    version = envelope.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise RewardServiceProtocolError(
            RewardServiceErrorCode.BAD_REQUEST,
            "reward wire version must be an integer",
        )
    if version != WIRE_VERSION:
        raise RewardServiceProtocolError(
            RewardServiceErrorCode.UNSUPPORTED_VERSION,
            f"unsupported reward wire version {version}; supported={WIRE_VERSION}",
            status_code=426,
            details={"supported_versions": [WIRE_VERSION]},
        )
    return envelope


def _reject_unknown_keys(
    payload: Mapping[str, Any],
    allowed: set[str],
    *,
    context: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise RewardServiceProtocolError(
            RewardServiceErrorCode.BAD_REQUEST,
            f"unsupported {context} fields: {unknown}",
        )


# Artifact wire schema: every dataclass field except the in-memory payload,
# which never crosses the HTTP boundary (remote scoring reads disk files).
_ARTIFACT_WIRE_FIELDS = tuple(
    field.name for field in fields(RewardInferenceArtifact) if field.name != "media"
)


def request_to_wire(request: RewardInferenceRequest) -> dict[str, Any]:
    """Serialize a disk-artifact request into the current protocol envelope."""

    artifacts: list[dict[str, Any]] = []
    for artifact in request.artifacts:
        if not artifact.path:
            raise ValueError(
                "remote reward scoring requires disk-materialized artifacts, but "
                f"{artifact.artifact_id!r} carries only in-memory media. Use a "
                "disk-artifact reward or score inline.",
            )
        artifacts.append(
            {name: getattr(artifact, name) for name in _ARTIFACT_WIRE_FIELDS},
        )
    body = {
        field.name: getattr(request, field.name)
        for field in fields(RewardInferenceRequest)
        if field.name != "artifacts"
    }
    body["artifacts"] = artifacts
    envelope = _wire_envelope(request=body)
    # Fail locally before opening a connection when metadata is not JSON-safe.
    json.dumps(envelope, allow_nan=False, separators=(",", ":"))
    return envelope


def request_from_wire(payload: Any) -> RewardInferenceRequest:
    """Parse and validate one current-version request envelope."""

    envelope = _validate_envelope(payload, expected_keys={"request"})
    body = _require_mapping(envelope.get("request"), context="reward request")
    request_fields = {field.name for field in fields(RewardInferenceRequest)}
    _reject_unknown_keys(body, request_fields, context="reward request")

    request_id = body.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RewardServiceProtocolError(
            RewardServiceErrorCode.BAD_REQUEST,
            "reward request_id must be a non-empty string",
        )
    raw_artifacts = body.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise RewardServiceProtocolError(
            RewardServiceErrorCode.BAD_REQUEST,
            "reward request artifacts must be a JSON array",
            request_id=request_id,
        )

    artifacts: list[RewardInferenceArtifact] = []
    try:
        for index, value in enumerate(raw_artifacts):
            artifact = _require_mapping(
                value,
                context=f"reward artifact at index {index}",
            )
            _reject_unknown_keys(
                artifact,
                set(_ARTIFACT_WIRE_FIELDS),
                context=f"reward artifact at index {index}",
            )
            artifacts.append(RewardInferenceArtifact(**dict(artifact)))
        # Construct from the full validated body (unknown keys were rejected
        # above) so a future request field crosses the wire instead of being
        # silently dropped by a hand-written constructor call.
        request_kwargs = dict(body)
        request_kwargs["artifacts"] = tuple(artifacts)
        return RewardInferenceRequest(**request_kwargs)
    except RewardServiceProtocolError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise RewardServiceProtocolError(
            RewardServiceErrorCode.BAD_REQUEST,
            f"invalid reward request: {error}",
            request_id=request_id,
        ) from error


def request_fingerprint(request: RewardInferenceRequest) -> str:
    """Return a stable idempotency fingerprint for a normalized request."""

    canonical = json.dumps(
        request_to_wire(request),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def score_response_to_wire(
    request_id: str,
    results: Sequence[RewardInferenceResult],
) -> dict[str, Any]:
    return _wire_envelope(
        request_id=request_id,
        results=[asdict(result) for result in results],
    )


def score_response_from_wire(
    payload: Any,
    *,
    expected_request_id: str | None = None,
) -> list[RewardInferenceResult]:
    envelope = _validate_envelope(payload, expected_keys={"request_id", "results"})
    response_request_id = envelope.get("request_id")
    if not isinstance(response_request_id, str) or not response_request_id:
        raise RewardServiceProtocolError(
            RewardServiceErrorCode.BAD_REQUEST,
            "reward response request_id must be a non-empty string",
        )
    if expected_request_id is not None and response_request_id != expected_request_id:
        raise RewardServiceProtocolError(
            RewardServiceErrorCode.BAD_REQUEST,
            "reward response request_id mismatch: "
            f"expected={expected_request_id!r}, actual={response_request_id!r}",
        )
    rows = envelope.get("results")
    if not isinstance(rows, list):
        raise RewardServiceProtocolError(
            RewardServiceErrorCode.BAD_REQUEST,
            "reward response results must be a JSON array",
        )
    try:
        return [
            RewardInferenceResult(**dict(_require_mapping(row, context="reward result")))
            for row in rows
        ]
    except (TypeError, ValueError) as error:
        raise RewardServiceProtocolError(
            RewardServiceErrorCode.BAD_REQUEST,
            f"invalid reward response: {error}",
        ) from error


def error_to_wire(error: RewardServiceProtocolError) -> dict[str, Any]:
    body: dict[str, Any] = {
        "code": error.code.value,
        "message": str(error),
        "retryable": error.retryable,
    }
    if error.request_id is not None:
        body["request_id"] = error.request_id
    if error.details:
        body["details"] = error.details
    return _wire_envelope(error=body)


def error_from_wire(payload: Any, *, status_code: int) -> RemoteRewardServiceError:
    try:
        envelope = _validate_envelope(payload, expected_keys={"error"})
        body = _require_mapping(envelope.get("error"), context="reward error")
        _reject_unknown_keys(
            body,
            {"code", "message", "retryable", "request_id", "details"},
            context="reward error",
        )
        code = body.get("code")
        message = body.get("message")
        if not isinstance(code, str) or not isinstance(message, str):
            raise ValueError("reward error requires string code and message")
        retryable = body.get("retryable", False)
        if not isinstance(retryable, bool):
            # bool("false") is True; a stringly flag must fail, not flip.
            raise ValueError("reward error retryable must be a JSON boolean")
        request_id = body.get("request_id")
        if request_id is not None and not isinstance(request_id, str):
            raise ValueError("reward error request_id must be a string")
        details = body.get("details") or {}
        if not isinstance(details, Mapping):
            raise ValueError("reward error details must be an object")
        return RemoteRewardServiceError(
            code,
            message,
            status_code=status_code,
            retryable=retryable,
            request_id=request_id,
            details=dict(details),
        )
    except (RewardServiceProtocolError, TypeError, ValueError) as error:
        return RemoteRewardServiceError(
            RewardServiceErrorCode.TRANSPORT_ERROR.value,
            f"reward service returned an invalid error response: {error}",
            status_code=status_code,
            retryable=status_code >= 500,
        )


def info_to_wire(info: RewardServiceInfo) -> dict[str, Any]:
    return _wire_envelope(info=asdict(info))


def info_from_wire(payload: Any) -> RewardServiceInfo:
    envelope = _validate_envelope(payload, expected_keys={"info"})
    body = _require_mapping(envelope.get("info"), context="reward service info")
    try:
        values = dict(body)
        raw_capabilities = values.get("capabilities") or ()
        if isinstance(raw_capabilities, str) or not isinstance(raw_capabilities, Sequence):
            # tuple("score_batch") would explode a string into characters.
            raise ValueError("reward service capabilities must be an array of strings")
        if not all(isinstance(capability, str) for capability in raw_capabilities):
            raise ValueError("reward service capabilities must be an array of strings")
        values["capabilities"] = tuple(raw_capabilities)
        return RewardServiceInfo(**values)
    except (TypeError, ValueError) as error:
        raise RewardServiceProtocolError(
            RewardServiceErrorCode.BAD_REQUEST,
            f"invalid reward service info: {error}",
        ) from error


def status_to_wire(status: str) -> dict[str, Any]:
    return _wire_envelope(status=status)


def status_from_wire(payload: Any) -> str:
    envelope = _validate_envelope(payload, expected_keys={"status"})
    status = envelope.get("status")
    if not isinstance(status, str):
        raise RewardServiceProtocolError(
            RewardServiceErrorCode.BAD_REQUEST,
            "reward service status must be a string",
        )
    return status


__all__ = [
    "error_from_wire",
    "error_to_wire",
    "info_from_wire",
    "info_to_wire",
    "request_fingerprint",
    "request_from_wire",
    "request_to_wire",
    "score_response_from_wire",
    "score_response_to_wire",
    "status_from_wire",
    "status_to_wire",
]
