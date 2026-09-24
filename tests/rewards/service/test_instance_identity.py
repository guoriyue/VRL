"""A cached service handshake cannot authorize a replacement server at the same URL."""

from dataclasses import replace

import aiohttp
import pytest
from PIL import Image

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.inference import (
    RewardInferenceArtifact,
    RewardInferenceRequest,
    RewardInferenceResult,
)
from vrl.rewards.service.client import HttpRewardScorer
from vrl.rewards.service.protocol import RemoteRewardServiceError
from vrl.rewards.service.server import RewardService
from vrl.utils.artifacts import sha256_file


@pytest.mark.asyncio
async def test_replacement_server_rejects_old_client_before_scoring(tmp_path):
    media = tmp_path / "input.png"
    Image.new("RGB", (8, 8), "white").save(media)

    class Runtime:
        def __init__(self, score):
            self.score = score
            self.requests = []

        async def score_batch(self, request):
            self.requests.append(request)
            return [
                RewardInferenceResult(
                    artifact_id=artifact.artifact_id, scores={"quality": self.score}
                )
                for artifact in request.artifacts
            ]

        async def shutdown(self):
            pass

    old_runtime, new_runtime = Runtime(0.25), Runtime(0.75)
    first = RewardService(
        old_runtime,
        artifact_roots=[tmp_path],
        port=0,
        model_name="unit",
        model_version="v1",
        generation_overlap_safe=True,
    )
    await first.start()
    host, port = first.address
    client = HttpRewardScorer(
        RewardInferenceConfig(
            kind="http",
            endpoint=f"http://{host}:{port}",
            expected_model="unit",
            expected_model_version="v1",
        )
    )
    second = None
    request = RewardInferenceRequest(
        "first",
        (
            RewardInferenceArtifact(
                artifact_id="image",
                sample_id="sample",
                path=str(media),
                prompt="white",
                size_bytes=media.stat().st_size,
                sha256=sha256_file(media),
            ),
        ),
    )
    try:
        assert (await client.score_batch(request))[0].scores["quality"] == 0.25
        assert client.external_accelerator_isolation_verified
        await first.shutdown_async()
        # Establish a new TCP connection while preserving the client's cached
        # handshake, as happens when a long-lived client reconnects.
        await client._close_session()
        second = RewardService(
            new_runtime,
            artifact_roots=[tmp_path],
            host=host,
            port=port,
            model_name="unit",
            # Matching model labels do not prove that scheduling capabilities
            # and in-flight request ownership survived the replacement.
            model_version="v1",
            generation_overlap_safe=False,
        )
        await second.start()
        with pytest.raises(RemoteRewardServiceError) as caught:
            await client.score_batch(replace(request, request_id="after-restart"))
        assert caught.value.code == "service_identity_changed"
        assert not caught.value.retryable
        assert not new_runtime.requests
        fresh = HttpRewardScorer(
            client._base_url, expected_model="unit", expected_model_version="v1"
        )
        try:
            await fresh.ensure_ready()
            assert not fresh.external_accelerator_isolation_verified
            assert (await fresh.score_batch(request))[0].scores["quality"] == 0.75
        finally:
            await fresh.shutdown()
    finally:
        await client.shutdown()
        await first.shutdown_async()
        if second is not None:
            await second.shutdown_async()


@pytest.mark.asyncio
async def test_replacement_rejects_old_phase_lease_and_cancel_before_owner_calls(tmp_path):
    class Runtime:
        requires_memory_parking = True

        def __init__(self):
            self.events = []

        async def activate(self):
            self.events.append("wake")

        async def park_memory(self):
            self.events.append("park")

        async def shutdown(self):
            pass

    old_runtime, new_runtime = Runtime(), Runtime()
    first = RewardService(old_runtime, artifact_roots=[tmp_path], port=0)
    await first.start()
    host, port = first.address
    endpoint = f"http://{host}:{port}"
    client = HttpRewardScorer(endpoint)
    second = None
    try:
        await client.ensure_ready()
        await client.activate()
        assert old_runtime.events == ["wake"]
        await client._close_session()
        await first.shutdown_async()
        second = RewardService(new_runtime, artifact_roots=[tmp_path], host=host, port=port)
        await second.start()
        for operation in (client.activate, client.park_memory):
            with pytest.raises(RemoteRewardServiceError) as caught:
                await operation()
            assert caught.value.code == "service_identity_changed"
            assert not caught.value.retryable
        with pytest.raises(RemoteRewardServiceError) as caught:
            await client.cancel("request-from-old-instance")
        assert caught.value.code == "service_identity_changed"
        # Cancellation against a replacement cannot settle work owned by the
        # previous instance, so shared artifacts must remain retained.
        assert not await client._settle_ambiguous_request("request-from-old-instance")
        async with (
            aiohttp.ClientSession() as session,
            session.post(f"{endpoint}/wake") as response,
        ):
            assert response.status == 409
            assert (await response.json())["error"]["code"] == "service_identity_changed"
        assert new_runtime.events == []
        fresh = HttpRewardScorer(endpoint)
        try:
            await fresh.activate()
            await fresh.park_memory()
            assert new_runtime.events == ["wake", "park"]
        finally:
            await fresh.shutdown()
    finally:
        await client.shutdown()
        await first.shutdown_async()
        if second is not None:
            await second.shutdown_async()
