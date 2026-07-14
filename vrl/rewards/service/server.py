"""Async standalone reward scoring service and CLI lifecycle."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

from vrl.rewards.inference import sha256_file, validate_reward_results
from vrl.rewards.service.owner import RewardRuntimeOwner
from vrl.rewards.service.protocol import (
    GENERATION_OVERLAP_SAFE_CAPABILITY,
    RewardServiceErrorCode,
    RewardServiceInfo,
    RewardServiceProtocolError,
)
from vrl.rewards.service.wire import (
    error_to_wire,
    info_to_wire,
    request_fingerprint,
    request_from_wire,
    score_response_to_wire,
    status_to_wire,
)
from vrl.utils.logging import init_logger

if TYPE_CHECKING:
    from vrl.rewards.inference import RewardInferenceRequest, RewardInferenceRuntime

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class _Reply:
    status: int
    body: dict[str, Any]


@dataclass(slots=True)
class _RequestRecord:
    fingerprint: str
    task: asyncio.Task[_Reply]


@dataclass(frozen=True, slots=True)
class RewardServiceConfig:
    """Typed CLI config; ``worker_config`` remains model-factory-owned."""

    host: str = "127.0.0.1"
    port: int = 8300
    model_name: str = ""
    model_version: str = ""
    artifact_roots: tuple[str, ...] = ()
    worker_config: dict[str, Any] = field(default_factory=dict)
    max_concurrency: int = 1
    max_pending_requests: int = 8
    max_cached_requests: int = 1024
    max_request_bytes: int = 16 * 1024 * 1024
    # Operator attestation for GPU services. CPU services are inferred safe by
    # _load_service because they execute no accelerator work beside generation.
    generation_overlap_safe: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RewardServiceConfig:
        payload = dict(value)
        allowed = {config_field.name for config_field in fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unsupported reward service config keys: {unknown}")
        roots = payload.get("artifact_roots") or ()
        if isinstance(roots, str):
            roots = (roots,)
        elif isinstance(roots, Sequence):
            roots = tuple(str(root) for root in roots)
        else:
            raise TypeError("reward service artifact_roots must be a list of paths")
        worker_config = payload.get("worker_config") or {}
        if not isinstance(worker_config, Mapping):
            raise TypeError("reward service worker_config must be a mapping")
        generation_overlap_safe = payload.get("generation_overlap_safe", False)
        if not isinstance(generation_overlap_safe, bool):
            raise TypeError("reward service generation_overlap_safe must be a boolean")
        payload["artifact_roots"] = roots
        payload["worker_config"] = dict(worker_config)
        return cls(**payload)


class RewardService:
    """Async HTTP owner for one runtime and one model identity."""

    def __init__(
        self,
        runtime: RewardInferenceRuntime,
        *,
        artifact_roots: Sequence[str | Path],
        host: str = "127.0.0.1",
        port: int = 8300,
        model_name: str = "",
        model_version: str = "",
        max_concurrency: int = 1,
        max_pending_requests: int = 8,
        max_cached_requests: int = 1024,
        max_request_bytes: int = 16 * 1024 * 1024,
        generation_overlap_safe: bool = False,
    ) -> None:
        if not host:
            raise ValueError("reward service host is required")
        if not 0 <= int(port) <= 65535:
            raise ValueError("reward service port must be between 0 and 65535")
        if max_cached_requests < 0:
            raise ValueError("reward service max_cached_requests must be >= 0")
        if max_request_bytes < 1:
            raise ValueError("reward service max_request_bytes must be >= 1")
        if not isinstance(generation_overlap_safe, bool):
            raise TypeError("reward service generation_overlap_safe must be a boolean")

        roots: list[Path] = []
        for value in artifact_roots:
            root = Path(value).expanduser()
            if not root.is_absolute():
                raise ValueError(f"reward artifact root must be absolute: {value!r}")
            try:
                root = root.resolve(strict=True)
            except OSError as error:
                raise ValueError(
                    f"reward artifact root does not exist: {value!r}",
                ) from error
            if not root.is_dir():
                raise ValueError(f"reward artifact root is not a directory: {root}")
            roots.append(root)
        if not roots:
            raise ValueError(
                "reward service requires at least one artifact_root; unrestricted "
                "client-supplied filesystem paths are not allowed",
            )

        self._host = host
        self._port = int(port)
        self._artifact_roots = tuple(roots)
        capabilities = ["cancel", "idempotency", "score_batch"]
        if generation_overlap_safe:
            capabilities.append(GENERATION_OVERLAP_SAFE_CAPABILITY)
        self._info = RewardServiceInfo(
            model_name=str(model_name).strip() or type(runtime).__name__,
            model_version=str(model_version).strip(),
            capabilities=tuple(capabilities),
            max_concurrency=max_concurrency,
            max_pending_requests=max_pending_requests,
        )
        self._max_cached_requests = max_cached_requests
        self._concurrency = asyncio.Semaphore(self._info.max_concurrency)
        self._records: OrderedDict[str, _RequestRecord] = OrderedDict()
        self._active_requests = 0
        self._records_lock = asyncio.Lock()
        self._accepting = False
        self._runner: web.AppRunner | None = None
        self._address: tuple[str, int] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._started = threading.Event()
        self._shutdown_task: asyncio.Task[None] | None = None

        @web.middleware
        async def typed_errors(
            request: web.Request,
            handler: Any,
        ) -> web.StreamResponse:
            try:
                return await handler(request)
            except RewardServiceProtocolError as error:
                return self._error_response(error)
            except web.HTTPRequestEntityTooLarge as error:
                return self._error_response(
                    RewardServiceProtocolError(
                        RewardServiceErrorCode.BAD_REQUEST,
                        str(error),
                        status_code=413,
                    ),
                )
            except web.HTTPMethodNotAllowed as error:
                return self._error_response(
                    RewardServiceProtocolError(
                        RewardServiceErrorCode.METHOD_NOT_ALLOWED,
                        str(error),
                        status_code=405,
                    ),
                )
            except web.HTTPNotFound:
                return self._error_response(
                    RewardServiceProtocolError(
                        RewardServiceErrorCode.NOT_FOUND,
                        f"unknown reward service path {request.path!r}",
                        status_code=404,
                    ),
                )
            except Exception as error:
                logger.exception("reward_service: unhandled request failure")
                return self._error_response(
                    RewardServiceProtocolError(
                        RewardServiceErrorCode.INTERNAL_ERROR,
                        f"{type(error).__name__}: {error}",
                        status_code=500,
                        retryable=True,
                    ),
                )

        self._app = web.Application(
            client_max_size=max_request_bytes,
            middlewares=[typed_errors],
        )
        self._app.router.add_get("/live", self._handle_live)
        self._app.router.add_get("/ready", self._handle_ready)
        self._app.router.add_get("/info", self._handle_info)
        self._app.router.add_post("/score", self._handle_score)
        self._app.router.add_delete(
            "/requests/{request_id:.*}",
            self._handle_cancel,
        )
        # Start the owner thread only after every fallible configuration and
        # aiohttp setup step has completed, so constructor errors cannot leak a
        # runtime thread that the caller never receives a handle to shut down.
        self._owner = RewardRuntimeOwner(runtime)

    @property
    def address(self) -> tuple[str, int]:
        if self._address is None and self._port == 0:
            self._started.wait(timeout=10)
        if self._address is None:
            if self._port:
                return self._host, self._port
            raise RuntimeError("reward service has not started")
        return self._address

    async def start(self) -> None:
        """Bind the listener and begin accepting requests."""

        if self._runner is not None:
            return
        if self._shutdown_task is not None:
            raise RuntimeError("reward service cannot restart after shutdown")
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError("reward service belongs to a different event loop")
        self._loop = loop
        runner = web.AppRunner(self._app, access_log=logger)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        try:
            await site.start()
            addresses = runner.addresses
            if not addresses:
                raise RuntimeError("reward service listener started without a socket")
        except BaseException:
            await runner.cleanup()
            await self.shutdown_async()
            raise
        host, port = addresses[0][:2]
        self._runner = runner
        self._address = str(host), int(port)
        self._accepting = True
        self._started.set()
        logger.info("reward_service: serving on http://%s:%s", host, port)

    async def shutdown_async(self) -> None:
        """Stop admission and release the owned runtime exactly once."""

        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(
                self._shutdown_impl(),
                name="reward-service-shutdown",
            )
        await asyncio.shield(self._shutdown_task)

    async def _handle_live(self, request: web.Request) -> web.Response:
        del request
        return self._json_response(200, status_to_wire("live"))

    async def _handle_ready(self, request: web.Request) -> web.Response:
        del request
        if self._accepting and self._owner.alive:
            return self._json_response(200, status_to_wire("ready"))
        return self._error_response(
            RewardServiceProtocolError(
                RewardServiceErrorCode.NOT_READY,
                "reward service is not accepting requests",
                status_code=503,
                retryable=True,
            ),
        )

    async def _handle_info(self, request: web.Request) -> web.Response:
        del request
        return self._json_response(200, info_to_wire(self._info))

    async def _handle_score(self, request: web.Request) -> web.Response:
        if not self._accepting:
            raise RewardServiceProtocolError(
                RewardServiceErrorCode.SERVICE_SHUTTING_DOWN,
                "reward service is shutting down",
                status_code=503,
                retryable=True,
            )
        try:
            payload = await request.json(loads=json.loads)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            raise RewardServiceProtocolError(
                RewardServiceErrorCode.BAD_REQUEST,
                f"request body must be valid JSON: {error}",
            ) from error
        inference_request = request_from_wire(payload)
        fingerprint = request_fingerprint(inference_request)

        async with self._records_lock:
            record = self._records.get(inference_request.request_id)
            if record is not None:
                if record.fingerprint != fingerprint:
                    raise RewardServiceProtocolError(
                        RewardServiceErrorCode.IDEMPOTENCY_CONFLICT,
                        "reward request_id was reused with a different payload",
                        status_code=409,
                        request_id=inference_request.request_id,
                    )
                self._records.move_to_end(inference_request.request_id)
                if record.task.done() and record.task.result().status == 200:
                    if self._active_requests >= self._info.max_pending_requests:
                        raise self._overloaded_error(inference_request.request_id)
                    self._active_requests += 1
                    cached_reply = record.task.result()
                    record.task = asyncio.create_task(
                        self._revalidate_cached(inference_request, cached_reply),
                        name=f"reward-revalidate:{inference_request.request_id}",
                    )
                elif record.task.done() and record.task.result().status >= 500:
                    if self._active_requests >= self._info.max_pending_requests:
                        raise self._overloaded_error(inference_request.request_id)
                    self._active_requests += 1
                    record.task = asyncio.create_task(
                        self._execute(inference_request),
                        name=f"reward-score:{inference_request.request_id}",
                    )
            else:
                if self._active_requests >= self._info.max_pending_requests:
                    raise self._overloaded_error(inference_request.request_id)
                self._active_requests += 1
                task = asyncio.create_task(
                    self._execute(inference_request),
                    name=f"reward-score:{inference_request.request_id}",
                )
                record = _RequestRecord(fingerprint=fingerprint, task=task)
                self._records[inference_request.request_id] = record

        reply = await asyncio.shield(record.task)
        return self._json_response(reply.status, reply.body)

    async def _handle_cancel(self, request: web.Request) -> web.Response:
        request_id = request.match_info["request_id"]
        if not request_id:
            raise RewardServiceProtocolError(
                RewardServiceErrorCode.BAD_REQUEST,
                "reward request_id is required for cancellation",
            )
        async with self._records_lock:
            record = self._records.get(request_id)
            if record is None:
                raise RewardServiceProtocolError(
                    RewardServiceErrorCode.REQUEST_NOT_FOUND,
                    f"reward request {request_id!r} was not found",
                    status_code=404,
                    request_id=request_id,
                )
            if record.task.done():
                raise RewardServiceProtocolError(
                    RewardServiceErrorCode.REQUEST_COMPLETED,
                    f"reward request {request_id!r} already completed",
                    status_code=409,
                    request_id=request_id,
                )
            record.task.cancel()
        await asyncio.shield(record.task)
        return self._json_response(200, status_to_wire("cancelled"))

    async def _execute(self, request: RewardInferenceRequest) -> _Reply:
        try:
            validation_started = time.perf_counter()
            request = await self._validate_artifact_paths(request)
            artifact_validation_ms = (time.perf_counter() - validation_started) * 1000.0
            queued_at = time.perf_counter()
            async with self._concurrency:
                service_queue_wait_ms = (time.perf_counter() - queued_at) * 1000.0
                inference_started = time.perf_counter()
                results = list(await self._owner.score_batch(request))
                service_inference_wall_ms = (time.perf_counter() - inference_started) * 1000.0
                results = validate_reward_results(request, results)
                results = [
                    replace(
                        result,
                        queue_wait_ms=(float(result.queue_wait_ms or 0.0) + service_queue_wait_ms),
                        latency_ms=(
                            float(result.latency_ms)
                            if result.latency_ms is not None
                            else float(result.queue_wait_ms or 0.0)
                            + float(result.inference_ms or 0.0)
                        )
                        + service_queue_wait_ms,
                        metadata={
                            **result.metadata,
                            "service_artifact_validation_ms": artifact_validation_ms,
                            "service_inference_wall_ms": service_inference_wall_ms,
                        },
                    )
                    for result in results
                ]
            return _Reply(
                status=200,
                body=score_response_to_wire(request.request_id, results),
            )
        except RewardServiceProtocolError as error:
            return _Reply(error.status_code, error_to_wire(error))
        except asyncio.CancelledError:
            error = RewardServiceProtocolError(
                RewardServiceErrorCode.CANCELLED,
                f"reward request {request.request_id!r} was cancelled",
                status_code=499,
                request_id=request.request_id,
            )
            return _Reply(error.status_code, error_to_wire(error))
        except Exception as error:
            logger.exception(
                "reward_service: scoring failed request_id=%s",
                request.request_id,
            )
            service_error = RewardServiceProtocolError(
                RewardServiceErrorCode.SCORING_FAILED,
                f"{type(error).__name__}: {error}",
                status_code=500,
                retryable=True,
                request_id=request.request_id,
            )
            return _Reply(service_error.status_code, error_to_wire(service_error))
        finally:
            async with self._records_lock:
                self._active_requests -= 1
                self._evict_completed_records(completing=request.request_id)

    async def _revalidate_cached(
        self,
        request: RewardInferenceRequest,
        cached_reply: _Reply,
    ) -> _Reply:
        try:
            await self._validate_artifact_paths(request)
            return cached_reply
        except RewardServiceProtocolError as error:
            return _Reply(error.status_code, error_to_wire(error))
        except asyncio.CancelledError:
            error = RewardServiceProtocolError(
                RewardServiceErrorCode.CANCELLED,
                f"reward request {request.request_id!r} was cancelled",
                status_code=499,
                request_id=request.request_id,
            )
            return _Reply(error.status_code, error_to_wire(error))
        except Exception as error:
            logger.exception(
                "reward_service: artifact revalidation failed request_id=%s",
                request.request_id,
            )
            service_error = RewardServiceProtocolError(
                RewardServiceErrorCode.INTERNAL_ERROR,
                f"{type(error).__name__}: {error}",
                status_code=500,
                retryable=True,
                request_id=request.request_id,
            )
            return _Reply(service_error.status_code, error_to_wire(service_error))
        finally:
            async with self._records_lock:
                self._active_requests -= 1
                self._evict_completed_records(completing=request.request_id)

    def _overloaded_error(self, request_id: str) -> RewardServiceProtocolError:
        return RewardServiceProtocolError(
            RewardServiceErrorCode.OVERLOADED,
            "reward service admission queue is full",
            status_code=429,
            retryable=True,
            request_id=request_id,
            details={
                "active_requests": self._active_requests,
                "max_pending_requests": self._info.max_pending_requests,
            },
        )

    async def _validate_artifact_paths(
        self,
        request: RewardInferenceRequest,
    ) -> RewardInferenceRequest:
        # Video artifacts can be large; hash them outside the HTTP loop so
        # liveness, cancellation, and admission responses remain responsive.
        validation = asyncio.create_task(
            asyncio.to_thread(self._validate_artifact_paths_sync, request),
            name=f"reward-validate:{request.request_id}",
        )
        try:
            return await asyncio.shield(validation)
        except asyncio.CancelledError:
            # A Python file read cannot be preempted safely. Keep the admission
            # slot until it has stopped touching the artifact, then publish the
            # request cancellation.
            with suppress(Exception, asyncio.CancelledError):
                await asyncio.shield(validation)
            raise

    def _validate_artifact_paths_sync(
        self,
        request: RewardInferenceRequest,
    ) -> RewardInferenceRequest:
        artifacts = []
        for artifact in request.artifacts:
            path = Path(artifact.path).expanduser()
            if not path.is_absolute():
                raise RewardServiceProtocolError(
                    RewardServiceErrorCode.PATH_NOT_ALLOWED,
                    f"reward artifact path must be absolute: {artifact.path!r}",
                    request_id=request.request_id,
                )
            try:
                resolved = path.resolve(strict=True)
            except OSError as error:
                raise RewardServiceProtocolError(
                    RewardServiceErrorCode.PATH_NOT_ALLOWED,
                    f"reward artifact path does not exist: {artifact.path!r}",
                    request_id=request.request_id,
                ) from error
            if not resolved.is_file() or not any(
                resolved.is_relative_to(root) for root in self._artifact_roots
            ):
                raise RewardServiceProtocolError(
                    RewardServiceErrorCode.PATH_NOT_ALLOWED,
                    f"reward artifact path is outside configured roots: {artifact.path!r}",
                    request_id=request.request_id,
                )
            if artifact.size_bytes is None or artifact.sha256 is None:
                raise RewardServiceProtocolError(
                    RewardServiceErrorCode.BAD_REQUEST,
                    "remote reward artifacts require size_bytes and sha256 integrity fields",
                    request_id=request.request_id,
                )
            try:
                actual_size = resolved.stat().st_size
            except OSError as error:
                raise RewardServiceProtocolError(
                    RewardServiceErrorCode.PATH_NOT_ALLOWED,
                    f"reward artifact disappeared during validation: {artifact.path!r}",
                    request_id=request.request_id,
                ) from error
            if actual_size != artifact.size_bytes:
                raise RewardServiceProtocolError(
                    RewardServiceErrorCode.PATH_NOT_ALLOWED,
                    "reward artifact size mismatch: "
                    f"path={artifact.path!r}, expected={artifact.size_bytes}, "
                    f"actual={actual_size}",
                    request_id=request.request_id,
                )
            try:
                actual_sha256 = sha256_file(resolved)
            except OSError as error:
                raise RewardServiceProtocolError(
                    RewardServiceErrorCode.PATH_NOT_ALLOWED,
                    f"reward artifact could not be read: {artifact.path!r}",
                    request_id=request.request_id,
                ) from error
            if actual_sha256 != artifact.sha256:
                raise RewardServiceProtocolError(
                    RewardServiceErrorCode.PATH_NOT_ALLOWED,
                    "reward artifact SHA-256 mismatch: "
                    f"path={artifact.path!r}, expected={artifact.sha256}, "
                    f"actual={actual_sha256}",
                    request_id=request.request_id,
                )
            artifacts.append(replace(artifact, path=str(resolved)))
        return replace(request, artifacts=tuple(artifacts))

    def _evict_completed_records(self, *, completing: str) -> None:
        completed = sum(
            request_id == completing or record.task.done()
            for request_id, record in self._records.items()
        )
        if completed <= self._max_cached_requests:
            return
        for request_id, record in tuple(self._records.items()):
            if request_id != completing and not record.task.done():
                continue
            del self._records[request_id]
            completed -= 1
            if completed <= self._max_cached_requests:
                break

    async def _shutdown_impl(self) -> None:
        self._accepting = False
        errors: list[BaseException] = []
        try:
            async with self._records_lock:
                active = [
                    record.task for record in self._records.values() if not record.task.done()
                ]
                for task in active:
                    task.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)
            runner = self._runner
            self._runner = None
            if runner is not None:
                try:
                    await runner.cleanup()
                except BaseException as error:
                    errors.append(error)
            try:
                await self._owner.close()
            except BaseException as error:
                errors.append(error)
        finally:
            self._started.set()
        if errors:
            summary = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
            raise RuntimeError(f"reward service shutdown failed: {summary}") from errors[0]

    @staticmethod
    def _json_response(
        status: int,
        body: dict[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> web.Response:
        return web.json_response(
            body,
            status=status,
            headers=headers,
            dumps=lambda value: json.dumps(value, allow_nan=False, separators=(",", ":")),
        )

    def _error_response(self, error: RewardServiceProtocolError) -> web.Response:
        headers = {"Retry-After": "1"} if error.code is RewardServiceErrorCode.OVERLOADED else None
        return self._json_response(error.status_code, error_to_wire(error), headers=headers)


def _load_service(config_path: Path) -> RewardService:
    from omegaconf import OmegaConf

    from vrl.rewards.runtime import InProcessRewardRuntime

    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, Mapping):
        raise TypeError("reward service config must be a mapping")
    cfg = RewardServiceConfig.from_mapping(raw)
    worker_config = cfg.worker_config
    if worker_config.get("sleep_offload"):
        raise ValueError(
            "reward service owns its device for its whole lifetime; drop "
            "worker_config.sleep_offload because parking is colocated-only",
        )
    roots = [
        path if Path(path).is_absolute() else config_path.parent / path
        for path in cfg.artifact_roots
    ]
    configured_device = str(worker_config.get("device", "")).strip().lower()
    runs_on_cpu = configured_device == "cpu" or configured_device.startswith("cpu:")
    return RewardService(
        InProcessRewardRuntime(worker_config),
        artifact_roots=roots,
        host=str(cfg.host),
        port=int(cfg.port),
        model_name=str(
            cfg.model_name
            or worker_config.get("reward_model_name")
            or worker_config.get("model_factory")
            or ""
        ),
        model_version=str(cfg.model_version or worker_config.get("reward_model_version") or ""),
        max_concurrency=int(cfg.max_concurrency),
        max_pending_requests=int(cfg.max_pending_requests),
        max_cached_requests=int(cfg.max_cached_requests),
        max_request_bytes=int(cfg.max_request_bytes),
        generation_overlap_safe=bool(cfg.generation_overlap_safe or runs_on_cpu),
    )


async def _run_cli(service: RewardService) -> None:
    loop = asyncio.get_running_loop()
    stop_requested = asyncio.Event()
    installed: list[signal.Signals] = []

    def request_stop(signum: signal.Signals) -> None:
        logger.info("reward_service: received %s", signum.name)
        stop_requested.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, request_stop, signum)
            installed.append(signum)
        except (NotImplementedError, RuntimeError):
            signal.signal(
                signum,
                lambda _number, _frame, signum=signum: loop.call_soon_threadsafe(
                    request_stop,
                    signum,
                ),
            )
    try:
        await service.start()
        await stop_requested.wait()
    finally:
        await service.shutdown_async()
        for signum in installed:
            loop.remove_signal_handler(signum)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="VRL standalone reward service")
    parser.add_argument("--config", required=True, help="service YAML config path")
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser().resolve(strict=True)
    service = _load_service(config_path)
    asyncio.run(_run_cli(service))


if __name__ == "__main__":
    main()


__all__ = ["RewardService", "RewardServiceConfig", "main"]
