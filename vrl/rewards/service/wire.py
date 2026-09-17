"""Versioned JSON wire format for standalone reward scoring.

The single owner of the envelope encode/decode pair: client.py and server.py
both import only these functions, so the two endpoints cannot drift apart.
Field sets derive from the inference.py dataclasses via ``fields(...)``, which
keeps those dataclasses the one schema source — adding a field changes the
wire, and unknown keys are rejected rather than ignored. The envelope pins
``WIRE_VERSION`` so a mismatched peer fails before any scoring, and
``request_fingerprint`` canonicalizes a request for the server's idempotency
check. Image/video tensors cross as explicit typed, checksummed numeric bytes,
never Python pickle. Shared filesystem paths remain an optional compatibility
transport for operators who explicitly configure allowed roots.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any

from vrl.rewards.inference import (
    RewardInferenceArtifact,
    RewardInferenceRequest,
    RewardInferenceResult,
)
from vrl.rewards.service.protocol import (
    WIRE_VERSION,
    RemoteRewardServiceError,
    RewardServiceErrorCode,
    RewardServiceInfo,
    RewardServiceProtocolError,
)
from vrl.utils.json_files import canonical_json_sha256


def _wire_envelope(**payload: Any) -> dict[str, Any]:
    return {
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
    # Version first: a cross-version peer must get the explicit 426, not a
    # confusing unknown-field complaint about an envelope key that moved.
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
    _reject_unknown_keys(
        envelope,
        {"version", *expected_keys},
        context="reward envelope",
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


@dataclass(frozen=True, slots=True)
class _TensorMedia:
    """Closed raw-tensor wire schema; dtype uses an explicit little-endian codec."""

    encoding: str
    dtype: str
    shape: list[int]
    data: str
    sha256: str


def _media_dtype(name: str) -> Any:
    import numpy as np

    # Deliberately bounded protocol vocabulary, not arbitrary numpy dtype input.
    codecs = {"uint8": "u1", "float16": "<f2", "float32": "<f4", "float64": "<f8"}
    if not isinstance(name, str) or name not in codecs:
        raise ValueError(f"unsupported reward media dtype: {name!r}")
    return np.dtype(codecs[name])


def _media_to_wire(media: Any) -> dict[str, Any]:
    import torch

    if not isinstance(media, torch.Tensor):
        raise ValueError("uploaded reward media must be an image/video tensor")
    if media.ndim not in {3, 4} or any(size < 1 for size in media.shape):
        raise ValueError("uploaded reward media requires non-empty [C,H,W] or [C,T,H,W]")
    media = media.detach().cpu()
    # NumPy has no portable bfloat16 dtype; conversion preserves every value.
    if media.dtype == torch.bfloat16:
        media = media.float()
    dtype = str(media.dtype).removeprefix("torch.")
    raw = media.numpy().astype(_media_dtype(dtype), copy=False).tobytes(order="C")
    return asdict(
        _TensorMedia(
            encoding="tensor-base64",
            dtype=dtype,
            shape=list(media.shape),
            data=base64.b64encode(raw).decode("ascii"),
            sha256=hashlib.sha256(raw).hexdigest(),
        )
    )


def _media_from_wire(value: Any) -> Any:
    import numpy as np
    import torch

    body = _require_mapping(value, context="reward media")
    _reject_unknown_keys(
        body, {field.name for field in fields(_TensorMedia)}, context="reward media"
    )
    media = _TensorMedia(**dict(body))
    if media.encoding != "tensor-base64":
        raise ValueError("reward media encoding must be tensor-base64")
    dtype = _media_dtype(media.dtype)
    if (
        not isinstance(media.shape, list)
        or len(media.shape) not in {3, 4}
        or any(type(size) is not int or size < 1 for size in media.shape)
    ):
        raise ValueError("reward media shape requires positive [C,H,W] or [C,T,H,W] dimensions")
    expected_bytes = math.prod(media.shape) * dtype.itemsize
    if not isinstance(media.data, str) or len(media.data) != 4 * ((expected_bytes + 2) // 3):
        raise ValueError("reward media encoded size disagrees with shape and dtype")
    # Check declared shape before decoding or allocating a tensor. No compression
    # or executable serialization can expand an adversarial payload here.
    raw = base64.b64decode(media.data, validate=True)
    if len(raw) != expected_bytes:
        raise ValueError("reward media byte size disagrees with shape and dtype")
    if not isinstance(media.sha256, str) or hashlib.sha256(raw).hexdigest() != media.sha256:
        raise ValueError("reward media SHA-256 mismatch")
    array = np.frombuffer(raw, dtype=dtype).reshape(media.shape)
    return torch.from_numpy(array.astype(dtype.newbyteorder("="), copy=True))


def request_to_wire(request: RewardInferenceRequest) -> dict[str, Any]:
    """Serialize uploaded tensors or explicitly shared artifact paths."""

    request = request.resolve_media()
    artifacts: list[dict[str, Any]] = []
    for artifact in request.artifacts:
        row = {
            field.name: getattr(artifact, field.name)
            for field in fields(RewardInferenceArtifact)
            if field.name != "media"
        }
        if artifact.media is not None:
            row.update(path="", size_bytes=None, sha256=None, media=_media_to_wire(artifact.media))
        artifacts.append(row)
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
                {field.name for field in fields(RewardInferenceArtifact)},
                context=f"reward artifact at index {index}",
            )
            artifact_kwargs = dict(artifact)
            if artifact_kwargs.get("media") is not None:
                if artifact_kwargs.get("path"):
                    raise ValueError("reward artifact must choose uploaded media or a shared path")
                artifact_kwargs["media"] = _media_from_wire(artifact_kwargs["media"])
            artifacts.append(RewardInferenceArtifact(**artifact_kwargs))
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

    return canonical_json_sha256(request_to_wire(request), allow_nan=False)


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
        details = body.get("details", {})
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
        return RewardServiceInfo(**dict(body))
    except (TypeError, ValueError) as error:
        raise RewardServiceProtocolError(
            RewardServiceErrorCode.BAD_REQUEST,
            f"invalid reward service info: {error}",
        ) from error


def status_to_wire(status: str) -> dict[str, Any]:
    return _wire_envelope(status=status)


def park_to_wire(*, residual_bytes: int) -> dict[str, Any]:
    """``POST /park`` reply: the service's own physical CUDA bytes after release."""

    return _wire_envelope(status="parked", residual_bytes=int(residual_bytes))


def park_from_wire(payload: Any) -> int:
    envelope = _validate_envelope(payload, expected_keys={"status", "residual_bytes"})
    if envelope.get("status") != "parked":
        raise RewardServiceProtocolError(
            RewardServiceErrorCode.BAD_REQUEST,
            f"reward service park status must be 'parked', got {envelope.get('status')!r}",
        )
    residual = envelope.get("residual_bytes")
    if isinstance(residual, bool) or not isinstance(residual, int) or residual < 0:
        raise RewardServiceProtocolError(
            RewardServiceErrorCode.BAD_REQUEST,
            "reward service park residual_bytes must be a non-negative integer",
        )
    return residual


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
    "park_from_wire",
    "park_to_wire",
    "request_fingerprint",
    "request_from_wire",
    "request_to_wire",
    "score_response_from_wire",
    "score_response_to_wire",
    "status_from_wire",
    "status_to_wire",
]
