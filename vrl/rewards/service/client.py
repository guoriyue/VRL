"""True-async client runtime for the standalone reward service.

``HttpRewardScorer`` is the remote ``RewardScorer`` transport — the twin of
the in-process ``InProcessRewardScorer`` (vrl/rewards/runtime.py). It exists
so the trainer process holds no reward model weights: scoring crosses to an
operator-owned service, which is why ``scoring_is_nonblocking`` is True and
why accelerator isolation is *verified* against the service's advertised
``generation_overlap_safe`` fact rather than assumed. Sessions are loop-affine (aiohttp binds a
pool to its creation loop), so preflight on the trainer loop hands off only
validated identity state and the scoring owner builds its own pool. An
ambiguous POST outcome is settled via explicit request-id cancellation; when
still unknown, the raised error tells the artifact owner to retain shared
files a live remote scorer might read.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import suppress
from dataclasses import replace
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import aiohttp

from vrl.rewards.inference import (
    RewardInferenceRequest,
    RewardInferenceResult,
)
from vrl.rewards.service.protocol import (
    RemoteRewardServiceError,
    RewardServiceErrorCode,
    RewardServiceInfo,
    RewardServiceProtocolError,
)
from vrl.rewards.service.wire import (
    error_from_wire,
    info_from_wire,
    request_to_wire,
    score_response_from_wire,
    status_from_wire,
)

if TYPE_CHECKING:
    from vrl.config.reward_inference import RewardInferenceConfig


class HttpRewardScorer:
    """Score against an operator-owned reward service over async HTTP."""

    scoring_is_nonblocking = True

    def __init__(
        self,
        service: str | RewardInferenceConfig,
        *,
        timeout_s: float | None = None,
        expected_model: str | None = None,
    ) -> None:
        from vrl.config.reward_inference import (
            RewardInferenceConfig,
            validate_http_origin,
        )

        if isinstance(service, RewardInferenceConfig):
            if service.kind != "http":
                raise ValueError(
                    "HttpRewardScorer requires inference.kind=http",
                )
            if timeout_s is not None or expected_model is not None:
                raise ValueError(
                    "timeout_s/expected_model are owned by the RewardInferenceConfig; "
                    "do not also pass them as keyword arguments",
                )
            # Validated and normalized by the config's own __post_init__.
            service_url = service.endpoint
            timeout_s = service.timeout_s
            expected_model = service.expected_model
        else:
            service_url = validate_http_origin(
                str(service),
                context="reward service endpoint",
            )
            timeout_s = 1800.0 if timeout_s is None else timeout_s
            expected_model = "" if expected_model is None else expected_model
        if timeout_s <= 0:
            raise ValueError("reward service timeout_s must be > 0")

        self._base_url = service_url
        self._timeout = aiohttp.ClientTimeout(total=float(timeout_s))
        self._expected_model = str(expected_model).strip()
        self._session: aiohttp.ClientSession | None = None
        self._session_loop: asyncio.AbstractEventLoop | None = None
        self._session_lock = asyncio.Lock()
        self._identity_lock = asyncio.Lock()
        self._identity_checked = False
        self._external_accelerator_isolation_verified = False
        self._closed = False

    @property
    def external_accelerator_isolation_verified(self) -> bool:
        """Whether preflight proved the service safe beside generation."""

        return self._external_accelerator_isolation_verified

    async def score_batch(
        self,
        request: RewardInferenceRequest,
    ) -> list[RewardInferenceResult]:
        await self._ensure_identity()
        payload = request_to_wire(request)
        roundtrip_started = time.perf_counter()
        try:
            body, status = await self._request_json("POST", "/score", json_body=payload)
        except asyncio.CancelledError:
            # HTTP disconnect does not reliably cancel model work. Make the
            # request-id cancellation explicit before preserving local cancel.
            with suppress(Exception, asyncio.CancelledError):
                await asyncio.shield(
                    asyncio.wait_for(self.cancel(request.request_id), timeout=2.0),
                )
            raise
        except RemoteRewardServiceError as error:
            if (
                error.code == RewardServiceErrorCode.TRANSPORT_ERROR.value
                and not await self._settle_ambiguous_request(request.request_id)
            ):
                error.retain_reward_artifacts = True
            raise
        if status >= 400:
            raise error_from_wire(body, status_code=status)
        try:
            results = score_response_from_wire(
                body,
                expected_request_id=request.request_id,
            )
            validated = request.validate_and_order_results(results)
            roundtrip_ms = (time.perf_counter() - roundtrip_started) * 1000.0
            return [
                replace(
                    result,
                    timing_ms={
                        **result.timing_ms,
                        "http_roundtrip_ms": roundtrip_ms,
                    },
                )
                for result in validated
            ]
        except RewardServiceProtocolError as error:
            raise RemoteRewardServiceError(
                RewardServiceErrorCode.TRANSPORT_ERROR.value,
                f"invalid reward score response: {error}",
                status_code=status,
            ) from error

    async def ready(self) -> bool:
        body, status = await self._request_json("GET", "/ready")
        if status == 503:
            return False
        if status >= 400:
            raise error_from_wire(body, status_code=status)
        return self._parse_status(body, status_code=status) == "ready"

    async def ensure_ready(self) -> None:
        """Startup preflight: readiness plus model identity, before any scoring.

        ``score_batch`` still re-checks identity lazily; this exists so the
        trainer fails at launch rather than after the first generation batch.
        """

        try:
            if not await self.ready():
                raise RemoteRewardServiceError(
                    RewardServiceErrorCode.NOT_READY.value,
                    f"reward service at {self._base_url} is not accepting requests",
                    status_code=503,
                    retryable=True,
                )
            await self._ensure_identity()
        finally:
            # Online preflight runs on the trainer loop, while continuous rollout
            # owns scoring on a dedicated loop/thread. aiohttp sessions are bound
            # to their creation loop, so preflight must hand off only validated
            # identity state and let the scoring owner create its own connection
            # pool. Strict scheduling recreates the pool on the same loop.
            await self._close_session()

    async def info(self) -> RewardServiceInfo:
        body, status = await self._request_json("GET", "/info")
        if status >= 400:
            raise error_from_wire(body, status_code=status)
        try:
            return info_from_wire(body)
        except RewardServiceProtocolError as error:
            raise RemoteRewardServiceError(
                RewardServiceErrorCode.TRANSPORT_ERROR.value,
                f"invalid reward service info: {error}",
                status_code=status,
            ) from error

    async def cancel(self, request_id: str) -> str:
        if not request_id:
            raise ValueError("reward request_id is required for cancellation")
        path = f"/requests/{quote(request_id, safe='')}"
        body, status = await self._request_json("DELETE", path)
        if status >= 400:
            raise error_from_wire(body, status_code=status)
        return self._parse_status(body, status_code=status)

    async def shutdown(self) -> None:
        """Close the client connection pool; the operator owns the service."""

        async with self._session_lock:
            if self._closed:
                return
            self._closed = True
            session = self._session
            self._session = None
            self._session_loop = None
        if session is not None:
            await session.close()

    async def _close_session(self) -> None:
        """Close the current-loop pool without shutting down the runtime."""

        async with self._session_lock:
            session = self._session
            self._session = None
            self._session_loop = None
        if session is not None:
            await session.close()

    async def _ensure_identity(self) -> None:
        if self._identity_checked:
            return
        async with self._identity_lock:
            if self._identity_checked:
                return
            info = await self.info()
            if self._expected_model and info.model_name != self._expected_model:
                raise RemoteRewardServiceError(
                    RewardServiceErrorCode.BAD_REQUEST.value,
                    "reward service model identity mismatch: "
                    f"expected={self._expected_model!r}, actual={info.model_name!r}",
                    retryable=False,
                    details={
                        "expected_model": self._expected_model,
                        "actual_model": info.model_name,
                        "actual_version": info.model_version,
                    },
                )
            self._external_accelerator_isolation_verified = info.generation_overlap_safe
            self._identity_checked = True

    async def _get_session(self) -> aiohttp.ClientSession:
        current_loop = asyncio.get_running_loop()
        async with self._session_lock:
            if self._closed:
                raise RuntimeError("reward HTTP runtime is shut down")
            if self._session is not None and self._session_loop is not current_loop:
                raise RuntimeError(
                    "reward HTTP session cannot cross event loops; close the owning "
                    "pipeline before transferring runtime ownership",
                )
            if self._session is None:
                self._session = aiohttp.ClientSession(
                    timeout=self._timeout,
                    headers={"Accept": "application/json"},
                )
                self._session_loop = current_loop
            return self._session

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> tuple[Any, int]:
        session = await self._get_session()
        try:
            async with session.request(
                method,
                f"{self._base_url}{path}",
                json=json_body,
            ) as response:
                status = response.status
                try:
                    body = await response.json(content_type=None)
                except (aiohttp.ContentTypeError, json.JSONDecodeError) as error:
                    raise RemoteRewardServiceError(
                        RewardServiceErrorCode.TRANSPORT_ERROR.value,
                        "reward service returned a non-JSON response",
                        status_code=status,
                        retryable=status >= 500,
                    ) from error
                return body, status
        except asyncio.CancelledError:
            raise
        except RemoteRewardServiceError:
            raise
        except (aiohttp.ClientError, TimeoutError) as error:
            raise RemoteRewardServiceError(
                RewardServiceErrorCode.TRANSPORT_ERROR.value,
                f"reward service request failed: {type(error).__name__}: {error}",
                retryable=True,
            ) from error

    async def _settle_ambiguous_request(self, request_id: str) -> bool:
        """Best-effort cancel after a POST whose remote state is unknown.

        DELETE waits for the server-side task to stop, including a
        non-cooperative model call. A timeout or missing request is still
        ambiguous, so callers must retain shared artifacts in that case.
        """

        try:
            await asyncio.shield(
                asyncio.wait_for(self.cancel(request_id), timeout=2.0),
            )
        except RemoteRewardServiceError as error:
            return error.code in {
                RewardServiceErrorCode.CANCELLED.value,
                RewardServiceErrorCode.REQUEST_COMPLETED.value,
            }
        except Exception:
            return False
        return True

    @staticmethod
    def _parse_status(body: Any, *, status_code: int) -> str:
        try:
            return status_from_wire(body)
        except RewardServiceProtocolError as error:
            raise RemoteRewardServiceError(
                RewardServiceErrorCode.TRANSPORT_ERROR.value,
                f"invalid reward service status: {error}",
                status_code=status_code,
            ) from error


__all__ = ["HttpRewardScorer", "RemoteRewardServiceError"]
