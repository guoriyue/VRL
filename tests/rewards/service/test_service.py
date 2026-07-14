"""Standalone reward service wire, transport, and lifecycle contracts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import aiohttp
import pytest

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.inference import (
    RewardInferenceArtifact,
    RewardInferenceRequest,
    RewardInferenceResult,
)
from vrl.rewards.runtime import build_reward_runtime
from vrl.rewards.service.client import HttpRewardRuntime
from vrl.rewards.service.protocol import (
    GENERATION_OVERLAP_SAFE_CAPABILITY,
    SHARED_FILESYSTEM_ARTIFACT_TRANSPORT,
    WIRE_PROTOCOL,
    WIRE_VERSION,
    RemoteRewardServiceError,
    RewardServiceErrorCode,
    RewardServiceInfo,
)
from vrl.rewards.service.server import RewardService, RewardServiceConfig
from vrl.rewards.service.wire import request_from_wire, request_to_wire


def _request(
    path: str,
    *,
    request_id: str = "req-1",
    reward_name: str = "unit",
) -> RewardInferenceRequest:
    artifact_path = Path(path)
    payload = artifact_path.read_bytes()
    return RewardInferenceRequest(
        request_id=request_id,
        artifacts=(
            RewardInferenceArtifact(
                artifact_id="a0",
                path=path,
                media_type="video",
                prompt="a dancer spins",
                sample_id="sample-0",
                policy_version=3,
                size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
                metadata={"target_text": "HELLO"},
            ),
        ),
        reward_name=reward_name,
        score_key="overall",
    )


def _results(request: RewardInferenceRequest) -> list[RewardInferenceResult]:
    return [
        RewardInferenceResult(
            artifact_id=artifact.artifact_id,
            scores={"overall": 0.75},
            selected_score=0.75,
            reward_name=request.reward_name,
            score_key=request.score_key,
            policy_version=artifact.policy_version,
            source_request_id=artifact.source_request_id,
            sample_id=artifact.sample_id,
            group_id=artifact.group_id,
            trajectory_id=artifact.trajectory_id,
            reward_model_version="unit-v1",
            worker_id="service",
        )
        for artifact in request.artifacts
    ]


class _FakeRuntime:
    def __init__(self) -> None:
        self.requests: list[RewardInferenceRequest] = []
        self.shutdown_calls = 0
        self.thread_ids: list[int] = []

    async def score_batch(self, request):
        self.thread_ids.append(threading.get_ident())
        self.requests.append(request)
        return _results(request)

    async def shutdown(self) -> None:
        self.thread_ids.append(threading.get_ident())
        self.shutdown_calls += 1


@asynccontextmanager
async def _running_service(
    runtime,
    artifact_root: Path,
    **kwargs,
):
    service = RewardService(
        runtime,
        host="127.0.0.1",
        port=0,
        artifact_roots=[artifact_root],
        model_name="unit-model",
        model_version="unit-v1",
        **kwargs,
    )
    await service.start()
    host, port = service.address
    client = HttpRewardRuntime(
        RewardInferenceConfig(
            kind="http",
            endpoint=f"http://{host}:{port}",
            timeout_s=5,
            expected_model="unit-model",
        ),
    )
    try:
        yield service, client
    finally:
        await client.shutdown()
        await service.shutdown_async()


def test_wire_roundtrip_is_versioned_and_preserves_request(tmp_path) -> None:
    artifact_file = tmp_path / "a0.mp4"
    artifact_file.write_bytes(b"x")
    request = _request(str(artifact_file))

    payload = request_to_wire(request)
    restored = request_from_wire(payload)

    assert payload["protocol"] == WIRE_PROTOCOL
    assert payload["version"] == WIRE_VERSION
    assert restored.request_id == request.request_id
    assert restored.reward_name == request.reward_name
    assert restored.artifacts[0].path == str(artifact_file)
    assert restored.artifacts[0].policy_version == 3
    assert restored.artifacts[0].size_bytes == 1
    assert restored.artifacts[0].sha256 == hashlib.sha256(b"x").hexdigest()
    assert restored.artifacts[0].metadata == {"target_text": "HELLO"}


def test_wire_rejects_inmemory_media() -> None:
    request = RewardInferenceRequest(
        request_id="req-2",
        artifacts=(
            RewardInferenceArtifact(
                artifact_id="a0",
                path="",
                media_type="image",
                media=object(),
            ),
        ),
        reward_name="unit",
        score_key="overall",
    )
    with pytest.raises(ValueError, match="disk-materialized"):
        request_to_wire(request)


def test_wire_rejects_unknown_fields_and_versions(tmp_path) -> None:
    artifact_file = tmp_path / "a0.mp4"
    artifact_file.write_bytes(b"x")
    payload = request_to_wire(_request(str(artifact_file)))
    payload["request"]["surprise"] = True
    with pytest.raises(ValueError, match="unsupported reward request fields"):
        request_from_wire(payload)

    payload = request_to_wire(_request(str(artifact_file)))
    payload["version"] = WIRE_VERSION + 1
    with pytest.raises(ValueError, match="unsupported reward wire version"):
        request_from_wire(payload)


@pytest.mark.asyncio
async def test_client_scores_through_async_server_and_validates_identity(tmp_path) -> None:
    artifact_file = tmp_path / "a0.mp4"
    artifact_file.write_bytes(b"x")
    runtime = _FakeRuntime()
    main_thread = threading.get_ident()

    async with _running_service(runtime, tmp_path) as (service, client):
        assert client.scoring_is_nonblocking is True
        assert client.external_accelerator_isolation_verified is False
        host, port = service.address
        # /live stays operator-facing (process supervisors probe it directly);
        # the trainer client only consumes readiness + identity via preflight.
        async with (
            aiohttp.ClientSession() as probe,
            probe.get(f"http://{host}:{port}/live") as live_response,
        ):
            assert live_response.status == 200
            assert (await live_response.json())["status"] == "live"
        assert await client.ready()
        await client.ensure_ready()
        assert client.external_accelerator_isolation_verified is False
        info = await client.info()
        results = await client.score_batch(_request(str(artifact_file)))

    assert info.model_name == "unit-model"
    assert info.model_version == "unit-v1"
    assert set(info.capabilities) == {"cancel", "idempotency", "score_batch"}
    assert [result.selected_score for result in results] == [0.75]
    assert results[0].metadata["service_artifact_validation_ms"] >= 0.0
    assert results[0].metadata["service_inference_wall_ms"] >= 0.0
    assert results[0].metadata["http_roundtrip_ms"] >= 0.0
    assert runtime.requests[0].artifacts[0].path == str(artifact_file.resolve())
    assert runtime.requests[0].artifacts[0].metadata == {"target_text": "HELLO"}
    assert runtime.shutdown_calls == 1
    assert len(set(runtime.thread_ids)) == 1
    assert runtime.thread_ids[0] != main_thread


@pytest.mark.asyncio
async def test_client_verifies_isolation_only_after_safe_service_preflight(tmp_path) -> None:
    runtime = _FakeRuntime()

    async with _running_service(
        runtime,
        tmp_path,
        generation_overlap_safe=True,
    ) as (_service, client):
        assert client.scoring_is_nonblocking is True
        assert client.external_accelerator_isolation_verified is False

        await client.ensure_ready()
        info = await client.info()

        assert GENERATION_OVERLAP_SAFE_CAPABILITY in info.capabilities
        assert client.external_accelerator_isolation_verified is True


@pytest.mark.asyncio
async def test_expected_model_mismatch_fails_before_scoring(tmp_path) -> None:
    artifact_file = tmp_path / "a0.mp4"
    artifact_file.write_bytes(b"x")
    runtime = _FakeRuntime()
    service = RewardService(
        runtime,
        artifact_roots=[tmp_path],
        host="127.0.0.1",
        port=0,
        model_name="actual-model",
    )
    await service.start()
    host, port = service.address
    client = HttpRewardRuntime(
        RewardInferenceConfig(
            kind="http",
            endpoint=f"http://{host}:{port}",
            expected_model="wrong-model",
        ),
    )
    try:
        with pytest.raises(RemoteRewardServiceError, match="identity mismatch") as caught:
            await client.score_batch(_request(str(artifact_file)))
    finally:
        await client.shutdown()
        await service.shutdown_async()

    assert caught.value.code == RewardServiceErrorCode.BAD_REQUEST.value
    assert runtime.requests == []


@pytest.mark.asyncio
async def test_ensure_ready_fails_fast_when_service_is_unreachable() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        _, unbound_port = listener.getsockname()
    client = HttpRewardRuntime(f"http://127.0.0.1:{unbound_port}")
    try:
        with pytest.raises(RemoteRewardServiceError) as caught:
            await client.ensure_ready()
    finally:
        await client.shutdown()

    assert caught.value.code == RewardServiceErrorCode.TRANSPORT_ERROR.value
    assert caught.value.retryable


@pytest.mark.asyncio
async def test_ensure_ready_fails_fast_on_identity_mismatch(tmp_path) -> None:
    runtime = _FakeRuntime()
    service = RewardService(
        runtime,
        artifact_roots=[tmp_path],
        host="127.0.0.1",
        port=0,
        model_name="actual-model",
    )
    await service.start()
    host, port = service.address
    client = HttpRewardRuntime(
        RewardInferenceConfig(
            kind="http",
            endpoint=f"http://{host}:{port}",
            expected_model="wrong-model",
        ),
    )
    try:
        with pytest.raises(RemoteRewardServiceError, match="identity mismatch"):
            await client.ensure_ready()
    finally:
        await client.shutdown()
        await service.shutdown_async()

    assert runtime.requests == []
    assert client.external_accelerator_isolation_verified is False


@pytest.mark.asyncio
async def test_client_rejects_unknown_artifact_transport_before_scoring(tmp_path) -> None:
    artifact_file = tmp_path / "a0.mp4"
    artifact_file.write_bytes(b"x")
    client = HttpRewardRuntime("http://127.0.0.1:1", expected_model="unit-model")

    async def incompatible_info() -> RewardServiceInfo:
        return RewardServiceInfo(
            model_name="unit-model",
            model_version="unit-v1",
            capabilities=("score_batch",),
            max_concurrency=1,
            max_pending_requests=1,
            artifact_transport="object_store_blobs",
        )

    client.info = incompatible_info
    try:
        with pytest.raises(RemoteRewardServiceError, match="transport is unsupported") as caught:
            await client.score_batch(_request(str(artifact_file)))
    finally:
        await client.shutdown()

    assert caught.value.code == RewardServiceErrorCode.BAD_REQUEST.value
    assert caught.value.details["supported_artifact_transport"] == (
        SHARED_FILESYSTEM_ARTIFACT_TRANSPORT
    )


@pytest.mark.asyncio
async def test_service_scoring_error_is_typed(tmp_path) -> None:
    class _FailingRuntime(_FakeRuntime):
        async def score_batch(self, request):
            raise KeyError("missing score keys: overall")

    artifact_file = tmp_path / "a0.mp4"
    artifact_file.write_bytes(b"x")
    async with _running_service(_FailingRuntime(), tmp_path) as (_service, client):
        with pytest.raises(RemoteRewardServiceError, match="missing score keys") as caught:
            await client.score_batch(_request(str(artifact_file)))

    assert caught.value.code == RewardServiceErrorCode.SCORING_FAILED.value
    assert caught.value.status_code == 500
    assert caught.value.retryable
    assert caught.value.request_id == "req-1"


@pytest.mark.asyncio
async def test_server_rejects_unsupported_wire_version_with_typed_error(tmp_path) -> None:
    async with _running_service(_FakeRuntime(), tmp_path) as (service, _client):
        host, port = service.address
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"http://{host}:{port}/score",
                json={"protocol": WIRE_PROTOCOL, "version": 999, "request": {}},
            ) as response,
        ):
            body = await response.json()

    assert response.status == 426
    assert body["error"]["code"] == RewardServiceErrorCode.UNSUPPORTED_VERSION.value
    assert body["error"]["details"]["supported_versions"] == [WIRE_VERSION]


@pytest.mark.asyncio
async def test_server_rejects_paths_outside_root_and_symlink_escapes(tmp_path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"x")
    escape = allowed / "escape.mp4"
    escape.symlink_to(outside)

    async with _running_service(_FakeRuntime(), allowed) as (_service, client):
        for path in (outside, escape):
            with pytest.raises(RemoteRewardServiceError) as caught:
                await client.score_batch(
                    _request(str(path), request_id=f"req-{path.name}"),
                )
            assert caught.value.code == RewardServiceErrorCode.PATH_NOT_ALLOWED.value
            assert caught.value.status_code == 400


@pytest.mark.asyncio
async def test_server_requires_and_verifies_artifact_integrity(tmp_path) -> None:
    artifact_file = tmp_path / "a0.mp4"
    artifact_file.write_bytes(b"original")
    valid = _request(str(artifact_file))
    artifact = valid.artifacts[0]
    missing = replace(
        valid,
        request_id="missing-integrity",
        artifacts=(replace(artifact, size_bytes=None, sha256=None),),
    )
    wrong_size = replace(
        valid,
        request_id="wrong-size",
        artifacts=(replace(artifact, size_bytes=artifact.size_bytes + 1),),
    )
    wrong_digest = replace(
        valid,
        request_id="wrong-digest",
        artifacts=(replace(artifact, sha256="0" * 64),),
    )

    async with _running_service(_FakeRuntime(), tmp_path) as (_service, client):
        for request, expected_code in (
            (missing, RewardServiceErrorCode.BAD_REQUEST.value),
            (wrong_size, RewardServiceErrorCode.PATH_NOT_ALLOWED.value),
            (wrong_digest, RewardServiceErrorCode.PATH_NOT_ALLOWED.value),
        ):
            with pytest.raises(RemoteRewardServiceError) as caught:
                await client.score_batch(request)
            assert caught.value.code == expected_code


@pytest.mark.asyncio
async def test_cached_idempotent_request_revalidates_replaced_file(tmp_path) -> None:
    artifact_file = tmp_path / "a0.mp4"
    artifact_file.write_bytes(b"first")
    request = _request(str(artifact_file), request_id="stable-artifact")
    runtime = _FakeRuntime()

    async with _running_service(runtime, tmp_path) as (_service, client):
        await client.score_batch(request)
        artifact_file.write_bytes(b"other")
        with pytest.raises(RemoteRewardServiceError) as caught:
            await client.score_batch(request)

    assert caught.value.code == RewardServiceErrorCode.PATH_NOT_ALLOWED.value
    assert len(runtime.requests) == 1


@pytest.mark.asyncio
async def test_bounded_admission_returns_retryable_backpressure(tmp_path) -> None:
    class _BlockingRuntime(_FakeRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        async def score_batch(self, request):
            self.requests.append(request)
            self.started.set()
            await asyncio.to_thread(self.release.wait)
            return _results(request)

    first_file = tmp_path / "first.mp4"
    second_file = tmp_path / "second.mp4"
    first_file.write_bytes(b"x")
    second_file.write_bytes(b"x")
    runtime = _BlockingRuntime()
    async with _running_service(
        runtime,
        tmp_path,
        max_pending_requests=1,
    ) as (_service, client):
        first = asyncio.create_task(
            client.score_batch(_request(str(first_file), request_id="req-first")),
        )
        assert await asyncio.to_thread(runtime.started.wait, 2)
        with pytest.raises(RemoteRewardServiceError) as caught:
            await client.score_batch(
                _request(str(second_file), request_id="req-second"),
            )
        runtime.release.set()
        await first

    assert caught.value.code == RewardServiceErrorCode.OVERLOADED.value
    assert caught.value.status_code == 429
    assert caught.value.retryable


@pytest.mark.asyncio
async def test_service_reports_actual_semaphore_queue_wait(tmp_path) -> None:
    class _BlockingRuntime(_FakeRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        async def score_batch(self, request):
            self.requests.append(request)
            self.started.set()
            await asyncio.to_thread(self.release.wait)
            return _results(request)

    class _ObservedSemaphore(asyncio.Semaphore):
        def __init__(self, value: int) -> None:
            super().__init__(value)
            self.waiting = asyncio.Event()

        async def acquire(self) -> bool:
            if self.locked():
                self.waiting.set()
            return await super().acquire()

    first_file = tmp_path / "first.mp4"
    second_file = tmp_path / "second.mp4"
    first_file.write_bytes(b"x")
    second_file.write_bytes(b"x")
    runtime = _BlockingRuntime()
    async with _running_service(
        runtime,
        tmp_path,
        max_concurrency=1,
        max_pending_requests=2,
    ) as (service, client):
        observed = _ObservedSemaphore(1)
        service._concurrency = observed
        first = asyncio.create_task(
            client.score_batch(_request(str(first_file), request_id="req-first")),
        )
        assert await asyncio.to_thread(runtime.started.wait, 2)
        second = asyncio.create_task(
            client.score_batch(_request(str(second_file), request_id="req-second")),
        )
        await asyncio.wait_for(observed.waiting.wait(), 2)
        runtime.release.set()
        first_results, second_results = await asyncio.gather(first, second)

    first_result = first_results[0]
    second_result = second_results[0]
    assert first_result.queue_wait_ms is not None
    assert first_result.queue_wait_ms >= 0
    assert second_result.queue_wait_ms is not None
    assert second_result.queue_wait_ms > first_result.queue_wait_ms
    assert second_result.latency_ms is not None
    assert second_result.latency_ms >= second_result.queue_wait_ms


@pytest.mark.asyncio
async def test_idempotency_joins_inflight_replays_cache_and_rejects_conflict(tmp_path) -> None:
    class _BlockingRuntime(_FakeRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        async def score_batch(self, request):
            self.requests.append(request)
            self.started.set()
            await asyncio.to_thread(self.release.wait)
            return _results(request)

    artifact_file = tmp_path / "a0.mp4"
    artifact_file.write_bytes(b"x")
    runtime = _BlockingRuntime()
    request = _request(str(artifact_file), request_id="stable-id")
    async with _running_service(runtime, tmp_path) as (_service, client):
        first = asyncio.create_task(client.score_batch(request))
        assert await asyncio.to_thread(runtime.started.wait, 2)
        duplicate = asyncio.create_task(client.score_batch(request))
        runtime.release.set()
        first_results, duplicate_results = await asyncio.gather(first, duplicate)
        replay_results = await client.score_batch(request)
        with pytest.raises(RemoteRewardServiceError) as caught:
            await client.score_batch(
                _request(
                    str(artifact_file),
                    request_id="stable-id",
                    reward_name="different",
                ),
            )

    assert len(runtime.requests) == 1

    # The idempotent service result is stable; client-local roundtrip time is a
    # per-attempt observation and is intentionally allowed to differ.
    def semantic_results(results):
        return [
            replace(
                result,
                metadata={
                    key: value
                    for key, value in result.metadata.items()
                    if key != "http_roundtrip_ms"
                },
            )
            for result in results
        ]

    assert (
        semantic_results(first_results)
        == semantic_results(duplicate_results)
        == semantic_results(replay_results)
    )
    assert all(
        result.metadata["http_roundtrip_ms"] >= 0.0
        for results in (first_results, duplicate_results, replay_results)
        for result in results
    )
    assert caught.value.code == RewardServiceErrorCode.IDEMPOTENCY_CONFLICT.value
    assert caught.value.status_code == 409


@pytest.mark.asyncio
async def test_client_cancellation_propagates_by_request_id(tmp_path) -> None:
    class _CancellableRuntime(_FakeRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.cancelled = threading.Event()

        async def score_batch(self, request):
            self.requests.append(request)
            self.started.set()
            try:
                await asyncio.Future()
            finally:
                self.cancelled.set()

    artifact_file = tmp_path / "a0.mp4"
    artifact_file.write_bytes(b"x")
    runtime = _CancellableRuntime()
    request = _request(str(artifact_file), request_id="cancel-me")
    async with _running_service(runtime, tmp_path) as (_service, client):
        scoring = asyncio.create_task(client.score_batch(request))
        assert await asyncio.to_thread(runtime.started.wait, 2)
        scoring.cancel()
        with pytest.raises(asyncio.CancelledError):
            await scoring
        assert await asyncio.to_thread(runtime.cancelled.wait, 2)
        with pytest.raises(RemoteRewardServiceError) as caught:
            await client.score_batch(request)

    assert caught.value.code == RewardServiceErrorCode.CANCELLED.value
    assert caught.value.status_code == 499
    assert len(runtime.requests) == 1


@pytest.mark.asyncio
async def test_ambiguous_score_transport_failure_requires_artifact_retention(
    tmp_path,
) -> None:
    artifact_file = tmp_path / "a0.mp4"
    artifact_file.write_bytes(b"x")
    client = HttpRewardRuntime("http://127.0.0.1:8300")
    client._identity_checked = True

    async def fail_request(*args, **kwargs):
        raise RemoteRewardServiceError(
            RewardServiceErrorCode.TRANSPORT_ERROR.value,
            "score response was lost",
            retryable=True,
        )

    async def fail_cancel(request_id):
        raise RemoteRewardServiceError(
            RewardServiceErrorCode.REQUEST_NOT_FOUND.value,
            f"request {request_id} was not found",
            status_code=404,
        )

    client._request_json = fail_request
    client.cancel = fail_cancel
    try:
        with pytest.raises(RemoteRewardServiceError) as caught:
            await client.score_batch(_request(str(artifact_file)))
    finally:
        await client.shutdown()

    assert caught.value.retain_reward_artifacts is True


def test_invalid_concurrency_does_not_start_runtime_owner(tmp_path, monkeypatch) -> None:
    runtime = _FakeRuntime()
    owner_started = False

    class _UnexpectedOwner:
        def __init__(self, runtime) -> None:
            nonlocal owner_started
            owner_started = True

    monkeypatch.setattr(
        "vrl.rewards.service.server.RewardRuntimeOwner",
        _UnexpectedOwner,
    )

    with pytest.raises(ValueError, match="max_concurrency"):
        RewardService(
            runtime,
            artifact_roots=[tmp_path],
            max_concurrency=0,
        )

    assert owner_started is False
    assert runtime.shutdown_calls == 0


@pytest.mark.asyncio
async def test_noncooperative_cancel_holds_admission_until_runtime_stops(tmp_path) -> None:
    class _BlockingRuntime(_FakeRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        async def score_batch(self, request):
            self.requests.append(request)
            self.started.set()
            # Represents a synchronous model/kernel that cannot observe task
            # cancellation until it returns control to its owner loop.
            self.release.wait()
            return _results(request)

    first_file = tmp_path / "first.mp4"
    second_file = tmp_path / "second.mp4"
    first_file.write_bytes(b"x")
    second_file.write_bytes(b"x")
    runtime = _BlockingRuntime()
    async with _running_service(
        runtime,
        tmp_path,
        max_pending_requests=1,
    ) as (_service, client):
        scoring = asyncio.create_task(
            client.score_batch(_request(str(first_file), request_id="blocking")),
        )
        assert await asyncio.to_thread(runtime.started.wait, 2)
        cancellation = asyncio.create_task(client.cancel("blocking"))
        await asyncio.sleep(0)
        with pytest.raises(RemoteRewardServiceError) as overloaded:
            await client.score_batch(
                _request(str(second_file), request_id="still-bounded"),
            )
        runtime.release.set()
        assert await cancellation == "cancelled"
        with pytest.raises(RemoteRewardServiceError) as cancelled:
            await scoring

    assert overloaded.value.code == RewardServiceErrorCode.OVERLOADED.value
    assert cancelled.value.code == RewardServiceErrorCode.CANCELLED.value


@pytest.mark.asyncio
async def test_retryable_scoring_failure_can_retry_same_request_id(tmp_path) -> None:
    class _FlakyRuntime(_FakeRuntime):
        async def score_batch(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                raise RuntimeError("transient failure")
            return _results(request)

    artifact_file = tmp_path / "a0.mp4"
    artifact_file.write_bytes(b"x")
    runtime = _FlakyRuntime()
    request = _request(str(artifact_file), request_id="retry-me")
    async with _running_service(runtime, tmp_path) as (_service, client):
        with pytest.raises(RemoteRewardServiceError) as first:
            await client.score_batch(request)
        results = await client.score_batch(request)

    assert first.value.code == RewardServiceErrorCode.SCORING_FAILED.value
    assert first.value.retryable
    assert [result.selected_score for result in results] == [0.75]
    assert len(runtime.requests) == 2


@pytest.mark.asyncio
async def test_invalid_timing_is_retryable_instead_of_cached_as_success(tmp_path) -> None:
    class _InvalidTimingRuntime(_FakeRuntime):
        async def score_batch(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return [replace(_results(request)[0], queue_wait_ms=float("nan"))]
            return _results(request)

    artifact_file = tmp_path / "a0.mp4"
    artifact_file.write_bytes(b"x")
    runtime = _InvalidTimingRuntime()
    request = _request(str(artifact_file), request_id="retry-invalid-timing")
    async with _running_service(runtime, tmp_path) as (_service, client):
        with pytest.raises(RemoteRewardServiceError) as first:
            await client.score_batch(request)
        results = await client.score_batch(request)

    assert first.value.code == RewardServiceErrorCode.SCORING_FAILED.value
    assert first.value.retryable
    assert [result.selected_score for result in results] == [0.75]
    assert len(runtime.requests) == 2


@pytest.mark.asyncio
async def test_service_shutdown_cancels_work_and_shuts_runtime_exactly_once(tmp_path) -> None:
    class _CancellableRuntime(_FakeRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.cancelled = threading.Event()

        async def score_batch(self, request):
            self.started.set()
            try:
                await asyncio.Future()
            finally:
                self.cancelled.set()

    artifact_file = tmp_path / "a0.mp4"
    artifact_file.write_bytes(b"x")
    runtime = _CancellableRuntime()
    service = RewardService(
        runtime,
        artifact_roots=[tmp_path],
        host="127.0.0.1",
        port=0,
        model_name="unit-model",
    )
    await service.start()
    host, port = service.address
    client = HttpRewardRuntime(f"http://{host}:{port}")
    scoring = asyncio.create_task(client.score_batch(_request(str(artifact_file))))
    assert await asyncio.to_thread(runtime.started.wait, 2)

    await service.shutdown_async()
    await service.shutdown_async()
    with pytest.raises(RemoteRewardServiceError) as caught:
        await scoring
    await client.shutdown()

    assert caught.value.code == RewardServiceErrorCode.CANCELLED.value
    assert runtime.cancelled.is_set()
    assert runtime.shutdown_calls == 1


def test_build_reward_runtime_accepts_typed_http_config() -> None:
    runtime = build_reward_runtime(
        {},
        inference=RewardInferenceConfig(
            kind="http",
            endpoint="http://127.0.0.1:8300",
            timeout_s=12,
            expected_model="unit-model",
        ),
    )
    assert isinstance(runtime, HttpRewardRuntime)
    assert runtime.scoring_is_nonblocking is True
    assert runtime.external_accelerator_isolation_verified is False


def test_service_requires_explicit_existing_absolute_artifact_roots(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least one artifact_root"):
        RewardService(_FakeRuntime(), artifact_roots=[])
    with pytest.raises(ValueError, match="must be absolute"):
        RewardService(_FakeRuntime(), artifact_roots=[Path("relative")])
    with pytest.raises(ValueError, match="does not exist"):
        RewardService(_FakeRuntime(), artifact_roots=[tmp_path / "missing"])


def test_cli_config_rejects_unknown_top_level_keys_but_keeps_worker_config_open() -> None:
    with pytest.raises(ValueError, match="unsupported reward service config keys"):
        RewardServiceConfig.from_mapping({"artifact_roots": ["/tmp"], "typo": True})
    parsed = RewardServiceConfig.from_mapping(
        {
            "artifact_roots": ["/tmp"],
            "worker_config": {"model_specific_knob": 7},
        },
    )
    assert parsed.worker_config == {"model_specific_knob": 7}
    assert parsed.generation_overlap_safe is False
    with pytest.raises(TypeError, match="must be a boolean"):
        RewardServiceConfig.from_mapping(
            {
                "artifact_roots": ["/tmp"],
                "generation_overlap_safe": "true",
            },
        )


def test_module_cli_handles_sigterm_without_eager_import_warning(tmp_path) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = tmp_path / "service.yaml"
    config.write_text(
        "\n".join(
            [
                "host: 127.0.0.1",
                f"port: {port}",
                "model_name: cli-model",
                "artifact_roots:",
                f"  - {tmp_path}",
                "worker_config:",
                "  model_factory: builtins:dict",
                "  device: cpu",
            ],
        ),
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vrl.rewards.service.server",
            "--config",
            str(config),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 15
        while True:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/live",
                    timeout=0.2,
                ) as response:
                    assert response.status == 200
                    break
            except OSError:
                if process.poll() is not None or time.monotonic() >= deadline:
                    stdout, stderr = process.communicate(timeout=2)
                    pytest.fail(
                        f"reward service CLI failed to start: stdout={stdout!r} stderr={stderr!r}",
                    )
                time.sleep(0.05)
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/info",
            timeout=1.0,
        ) as response:
            info = json.load(response)["info"]
        assert GENERATION_OVERLAP_SAFE_CAPABILITY in info["capabilities"]
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=15)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    assert process.returncode == 0, (stdout, stderr)
    assert "found in sys.modules" not in stderr
