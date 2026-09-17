"""Real HTTP uploads need no shared filesystem and never deserialize pickle."""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from dataclasses import replace

import aiohttp
import pytest
import torch

from vrl.rewards.inference import (
    RewardInferenceArtifact,
    RewardInferenceRequest,
    RewardInferenceResult,
)
from vrl.rewards.service.client import HttpRewardScorer
from vrl.rewards.service.protocol import RemoteRewardServiceError, RewardServiceProtocolError
from vrl.rewards.service.server import RewardService
from vrl.rewards.service.wire import request_fingerprint, request_from_wire, request_to_wire


def _request(media, *, request_id="upload"):
    return RewardInferenceRequest(
        request_id=request_id,
        artifacts=(
            RewardInferenceArtifact(
                artifact_id="frame-0",
                sample_id="sample-0",
                path="",
                media=media,
                prompt="the dancer",
                metadata={"fps": 8.0},
            ),
        ),
    )


class _MediaScorer:
    def __init__(self, *, blocking=False):
        self.requests = []
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.shutdown_calls = 0
        self.blocking = blocking

    async def score_batch(self, request):
        self.requests.append(request)
        self.started.set()
        try:
            if self.blocking:
                await asyncio.Future()
            return [
                RewardInferenceResult(
                    artifact_id=artifact.artifact_id,
                    scores={"sum": float(artifact.media.float().sum())},
                )
                for artifact in request.artifacts
            ]
        finally:
            self.stopped.set()

    async def shutdown(self):
        self.shutdown_calls += 1


@asynccontextmanager
async def _service(runtime, *, timeout_s=5, **kwargs):
    service = RewardService(
        runtime, port=0, artifact_roots=(), model_name="upload-model", model_version="1", **kwargs
    )
    await service.start()
    host, port = service.address
    endpoint = f"http://{host}:{port}"
    client = HttpRewardScorer(
        endpoint, timeout_s=timeout_s, expected_model="upload-model", expected_model_version="1"
    )
    try:
        yield service, client, endpoint
    finally:
        await client.shutdown()
        await service.shutdown_async()


@pytest.mark.parametrize("shape", [(3, 4, 5), (3, 2, 4, 5)])
@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.float16, torch.float32, torch.float64, torch.bfloat16]
)
def test_wire_tensor_values_shape_dtype_and_digest_roundtrip(shape, dtype):
    media = torch.arange(torch.tensor(shape).prod().item()).reshape(shape).to(dtype)
    request = _request(media)
    wire = request_to_wire(request)
    decoded = request_from_wire(wire)
    actual = decoded.artifacts[0]
    assert actual.path == ""
    assert actual.size_bytes is None and actual.sha256 is None
    assert actual.sample_id == "sample-0"
    assert actual.prompt == "the dancer" and actual.metadata == {"fps": 8.0}
    assert actual.media.dtype == (torch.float32 if dtype == torch.bfloat16 else dtype)
    assert torch.equal(actual.media, media.float() if dtype == torch.bfloat16 else media)
    assert request_fingerprint(decoded) == request_fingerprint(request)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("encoding", "pickle", "encoding"),
        ("dtype", "object", "dtype"),
        ("shape", [3, 10**30, 4], "size"),
        ("shape", [True, 4, 4], "shape"),
        ("data", "!" * 64, "base64"),
        ("sha256", "0" * 64, "SHA-256"),
        ("extra", 1, "unsupported reward media fields"),
    ],
)
def test_wire_rejects_malformed_media_before_scoring(field, value, match):
    wire = request_to_wire(_request(torch.zeros(3, 4, 4, dtype=torch.uint8)))
    wire["request"]["artifacts"][0]["media"][field] = value
    with pytest.raises(RewardServiceProtocolError, match=match):
        request_from_wire(wire)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", [(3, 4, 5), (3, 2, 4, 5)])
async def test_rootless_http_upload_roundtrip_idempotency_and_shutdown(tmp_path, shape):
    missing_shared_root = tmp_path / "not-mounted-on-service"
    media = torch.arange(torch.tensor(shape).prod().item()).reshape(shape).float() / 100
    request = _request(media)
    runtime = _MediaScorer()
    async with _service(runtime) as (service, client, endpoint):
        await client.ensure_ready()
        assert await client.ready()
        assert (await client.info()).model_name == "upload-model"
        async with (
            aiohttp.ClientSession() as probe,
            probe.get(f"{endpoint}/info") as response,
        ):
            assert response.status == 200
            assert "launch_token" not in (await response.json())["info"]
        result = await client.score_batch(request)
        replay = await client.score_batch(request)
        assert result[0].scores == replay[0].scores == {"sum": float(media.sum())}
        assert len(runtime.requests) == 1
        delivered = runtime.requests[0].artifacts[0]
        assert delivered.path == "" and torch.equal(delivered.media, media)
        changed = replace(request, artifacts=(replace(request.artifacts[0], media=media + 1),))
        with pytest.raises(RemoteRewardServiceError) as conflict:
            await client.score_batch(changed)
        assert conflict.value.status_code == 409
        assert service._active_requests == 0
        async with (
            aiohttp.ClientSession() as probe,
            probe.get(f"{endpoint}/live") as response,
        ):
            assert response.status == 200
    assert not missing_shared_root.exists()
    assert runtime.shutdown_calls == 1
    assert not service._owner.alive


@pytest.mark.asyncio
async def test_rootless_service_rejects_shared_paths_and_oversize_uploads(tmp_path):
    local_file = tmp_path / "client-only.pt"
    local_file.write_bytes(b"not shared")
    runtime = _MediaScorer()
    async with _service(runtime, max_request_bytes=1024) as (service, client, _endpoint):
        path_request = RewardInferenceRequest(
            request_id="path",
            artifacts=(
                RewardInferenceArtifact(
                    artifact_id="a",
                    sample_id="s",
                    path=str(local_file),
                ),
            ),
        )
        with pytest.raises(RemoteRewardServiceError) as denied:
            await client.score_batch(path_request)
        assert denied.value.code == "path_not_allowed"
        with pytest.raises(RemoteRewardServiceError) as oversized:
            await client.score_batch(_request(torch.zeros(3, 32, 32)))
        assert oversized.value.status_code == 413
        assert runtime.requests == [] and service._active_requests == 0
        assert await client.ready()


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["cancel", "timeout", "shutdown"])
async def test_uploaded_request_lifecycle_drains_remote_work(termination):
    runtime = _MediaScorer(blocking=True)
    async with _service(runtime, timeout_s=0.2 if termination == "timeout" else 5) as (
        service,
        client,
        _endpoint,
    ):
        await client.ensure_ready()
        task = asyncio.create_task(client.score_batch(_request(torch.ones(3, 4, 4))))
        assert await asyncio.to_thread(runtime.started.wait, 2)
        if termination == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            if termination == "shutdown":
                await service.shutdown_async()
            with pytest.raises(RemoteRewardServiceError) as caught:
                await task
            if termination == "timeout":
                assert not caught.value.retain_reward_artifacts
        assert await asyncio.to_thread(runtime.stopped.wait, 2)
        assert service._active_requests == 0
    assert runtime.shutdown_calls == 1
