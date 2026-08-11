"""Versioned JSON wire format for standalone reward scoring."""

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


def _validate_envelope(payload: Any) -> Mapping[str, Any]:
    envelope = _require_mapping(payload, context="reward service payload")
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


def request_to_wire(request: RewardInferenceRequest) -> dict[str, Any]:
    """Serialize a disk-artifact request into the current protocol envelope."""

    artifact_field_names = tuple(
        field.name for field in fields(RewardInferenceArtifact) if field.name != "media"
    )
    artifacts: list[dict[str, Any]] = []
    for artifact in request.artifacts:
        if not artifact.path:
            raise ValueError(
                "remote reward scoring requires disk-materialized artifacts, but "
                f"{artifact.artifact_id!r} carries only in-memory media. Use a "
                "disk-artifact reward or score inline.",
            )
        artifacts.append(
            {name: getattr(artifact, name) for name in artifact_field_names},
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

    envelope = _validate_envelope(payload)
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

    artifact_fields = {
        field.name for field in fields(RewardInferenceArtifact) if field.name != "media"
    }
    artifacts: list[RewardInferenceArtifact] = []
    try:
        for index, value in enumerate(raw_artifacts):
            artifact = _require_mapping(
                value,
                context=f"reward artifact at index {index}",
            )
            _reject_unknown_keys(
                artifact,
                artifact_fields,
                context=f"reward artifact at index {index}",
            )
            artifacts.append(RewardInferenceArtifact(**dict(artifact)))
        return RewardInferenceRequest(
            request_id=request_id,
            artifacts=tuple(artifacts),
        )
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
    envelope = _validate_envelope(payload)
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
        envelope = _validate_envelope(payload)
        body = _require_mapping(envelope.get("error"), context="reward error")
        code = body.get("code")
        message = body.get("message")
        if not isinstance(code, str) or not isinstance(message, str):
            raise ValueError("reward error requires string code and message")
        details = body.get("details") or {}
        if not isinstance(details, Mapping):
            raise ValueError("reward error details must be an object")
        return RemoteRewardServiceError(
            code,
            message,
            status_code=status_code,
            retryable=bool(body.get("retryable", False)),
            request_id=body.get("request_id"),
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
    envelope = _validate_envelope(payload)
    body = _require_mapping(envelope.get("info"), context="reward service info")
    try:
        values = dict(body)
        values["capabilities"] = tuple(values.get("capabilities") or ())
        return RewardServiceInfo(**values)
    except (TypeError, ValueError) as error:
        raise RewardServiceProtocolError(
            RewardServiceErrorCode.BAD_REQUEST,
            f"invalid reward service info: {error}",
        ) from error


def status_to_wire(status: str) -> dict[str, Any]:
    return _wire_envelope(status=status)


def status_from_wire(payload: Any) -> str:
    envelope = _validate_envelope(payload)
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
