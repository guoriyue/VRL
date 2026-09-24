"""Cached HTTP rewards must revalidate their actual reference and verifier files."""

import asyncio
import json
import threading
from dataclasses import replace

import pytest
import torch
from PIL import Image

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.service.client import HttpRewardScorer
from vrl.rewards.service.protocol import RemoteRewardServiceError
from vrl.rewards.service.server import RewardService
from vrl.utils.artifacts import sha256_file


@pytest.mark.asyncio
async def test_cached_reward_rejects_changed_or_outside_target(tmp_path, monkeypatch):
    from vrl.rewards.models.image_sharpness import ImageSharpnessRewardModel

    root = tmp_path / "allowed"
    root.mkdir()
    candidate, target = root / "candidate.png", root / "target.png"
    image = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    image.paste((200, 30, 80, 255), (4, 4, 12, 12))
    image.save(candidate)
    image.save(target)
    config = root / "server.yaml"
    config.write_text(
        json.dumps(
            {
                "host": "127.0.0.1",
                "port": 0,
                "model_name": "aux-integrity",
                "model_version": "test",
                "artifact_roots": [str(root)],
                "worker_config": {
                    "model_factory": "vrl.rewards.models.image_sharpness:ImageSharpnessRewardModel",
                    "device": "cpu",
                    "data_root": str(root),
                },
            }
        )
    )
    service = RewardService.from_yaml(config)
    await service.start()
    host, port = service.address
    client = HttpRewardScorer(
        RewardInferenceConfig(
            kind="http",
            endpoint=f"http://{host}:{port}",
            expected_model="aux-integrity",
            expected_model_version="test",
            timeout_s=5,
        )
    )
    artifact = RewardInferenceArtifact(
        artifact_id="candidate",
        sample_id="candidate",
        path=str(candidate),
        size_bytes=candidate.stat().st_size,
        sha256=sha256_file(candidate),
        metadata={"target_image": str(target)},
    )
    request = RewardInferenceRequest(request_id="same-request", artifacts=(artifact,))
    try:
        first = await client.score_batch(request)
        repeated = await client.score_batch(request)
        assert repeated[0].scores == first[0].scores
        image.paste((30, 200, 80, 255), (4, 4, 12, 12))
        image.save(target)
        with pytest.raises(RemoteRewardServiceError) as changed:
            await client.score_batch(request)
        assert changed.value.code == "path_not_allowed"
        # A new request re-pins the changed target instead of reusing the cached digest.
        fresh = await client.score_batch(replace(request, request_id="new-target-version"))
        assert fresh[0].scores == first[0].scores
        outside = tmp_path / "outside.png"
        image.save(outside)
        outside_request = RewardInferenceRequest(
            request_id="outside",
            artifacts=(replace(artifact, metadata={"target_image": str(outside)}),),
        )
        with pytest.raises(RemoteRewardServiceError) as rejected:
            await client.score_batch(outside_request)
        assert rejected.value.code == "path_not_allowed"
        # Uploading the candidate does not authorize unvalidated shared targets.
        uploaded = replace(
            artifact,
            path="",
            size_bytes=None,
            sha256=None,
            media=torch.zeros(4, 1, 16, 16, dtype=torch.uint8),
            metadata={"target_image": str(outside)},
        )
        with pytest.raises(RemoteRewardServiceError) as uploaded_rejected:
            await client.score_batch(
                RewardInferenceRequest(request_id="uploaded-outside", artifacts=(uploaded,))
            )
        assert uploaded_rejected.value.code == "path_not_allowed"
        original_score = ImageSharpnessRewardModel._score_one

        def mutate_after_scoring(model, item):
            result = original_score(model, item)
            image.paste((30, 80, 200, 255), (4, 4, 12, 12))
            image.save(target)
            return result

        monkeypatch.setattr(ImageSharpnessRewardModel, "_score_one", mutate_after_scoring)
        with pytest.raises(RemoteRewardServiceError) as during_scoring:
            await client.score_batch(replace(request, request_id="mid-score-mutation"))
        assert during_scoring.value.code == "path_not_allowed"
    finally:
        await client.shutdown()
        await service.shutdown_async()


@pytest.mark.asyncio
async def test_repeated_cancel_during_file_hash_keeps_admission_until_reader_finishes(
    tmp_path, monkeypatch
):
    from vrl.rewards.service import server

    started, release = threading.Event(), threading.Event()
    image = tmp_path / "candidate.png"
    Image.new("RGBA", (4, 4), (20, 30, 40, 255)).save(image)
    digest = sha256_file(image)

    def slow_hash(path):
        started.set()
        if not release.wait(5):
            raise TimeoutError("test reader was not released")
        return sha256_file(path)

    monkeypatch.setattr(server, "sha256_file", slow_hash)

    class Runtime:
        async def score_batch(self, request):
            raise AssertionError("cancelled validation must not enter model execution")

        async def shutdown(self):
            pass

    service = RewardService(
        Runtime(),
        host="127.0.0.1",
        port=0,
        artifact_roots=[tmp_path],
        model_name="reader",
        model_version="test",
        max_pending_requests=1,
    )
    await service.start()
    host, port = service.address
    client = HttpRewardScorer(f"http://{host}:{port}", timeout_s=5)
    request = RewardInferenceRequest(
        request_id="reading",
        artifacts=(
            RewardInferenceArtifact(
                artifact_id="one",
                sample_id="one",
                path=str(image),
                size_bytes=image.stat().st_size,
                sha256=digest,
            ),
        ),
    )
    scoring = asyncio.create_task(client.score_batch(request))
    cancels = []
    try:
        assert await asyncio.to_thread(started.wait, 2)
        cancels.append(asyncio.create_task(client.cancel("reading")))
        await asyncio.sleep(0.02)
        cancels.append(asyncio.create_task(client.cancel("reading")))
        await asyncio.sleep(0.05)
        assert not any(task.done() for task in cancels), (
            "cancellation acknowledged while a file reader remained live"
        )
        with pytest.raises(RemoteRewardServiceError) as full:
            await client.score_batch(replace(request, request_id="another"))
        assert full.value.code == "overloaded"
        release.set()
        await asyncio.gather(*cancels)
        with pytest.raises(RemoteRewardServiceError) as cancelled:
            await scoring
        assert cancelled.value.code == "cancelled"
    finally:
        release.set()
        await asyncio.gather(scoring, *cancels, return_exceptions=True)
        await client.shutdown()
        await service.shutdown_async()
