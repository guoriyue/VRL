"""Existing-media scoring preserves evidence across resume and real HTTP."""

from __future__ import annotations

import json

import pytest
from PIL import Image, UnidentifiedImageError

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.evaluation import ScoringConfig, evaluation_access, rescore_media
from vrl.rewards.runtime import build_reward_scorer
from vrl.rewards.service.server import RewardService
from vrl.utils.artifacts import sha256_file


@pytest.fixture
def scoring_input(tmp_path):
    Image.new("RGB", (12, 12), "white").save(tmp_path / "image.png")
    manifest = tmp_path / "media.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "image-1",
                "prompt_id": "prompt-1",
                "prompt": "white square",
                "path": "image.png",
            }
        )
        + "\n"
    )
    config = ScoringConfig(
        name="sharpness",
        revision="test-v1",
        preprocessing_revision="native-v1",
        rubric_revision="laplacian-v1",
        worker_config={
            "model_factory": "vrl.rewards.models.image_sharpness:ImageSharpnessRewardModel",
            "device": "cpu",
            "reward_model_version": "test-v1",
        },
    )
    return manifest, config, tmp_path / "scores"


@pytest.mark.asyncio
async def test_real_cpu_scoring_resumes_without_constructing_model(scoring_input, monkeypatch):
    manifest, config, output = scoring_input
    first = await rescore_media(manifest, config, output)
    row = json.loads(next((output / "samples").glob("*.json")).read_text())
    assert row["result"]["scores"] == {"image_sharpness": 0.0}
    assert row["input"]["sha256"]
    assert row["result"]["reward_model_version"] == "test-v1"

    def unexpected(*args, **kwargs):
        raise AssertionError("cached evaluation must not construct a scorer")

    monkeypatch.setattr("vrl.rewards.runtime.build_reward_scorer", unexpected)
    second = await rescore_media(manifest, config, output, resume=True)
    assert first["scored"] == 1
    assert second["reused"] == 1
    assert second["scored"] == 0
    with pytest.raises(FileExistsError):
        await rescore_media(manifest, config, output)


@pytest.mark.asyncio
async def test_producer_digest_preserves_resume_identity_and_rejects_replaced_media(
    scoring_input, monkeypatch
):
    manifest, config, output = scoring_input
    first = await rescore_media(manifest, config, output)
    row = json.loads(manifest.read_text())
    row["sha256"] = sha256_file(manifest.parent / row["path"])
    manifest.write_text(json.dumps(row) + "\n")

    def unexpected(*args, **kwargs):
        raise AssertionError("resume or digest rejection must not construct a scorer")

    monkeypatch.setattr("vrl.rewards.runtime.build_reward_scorer", unexpected)
    resumed = await rescore_media(manifest, config, output, resume=True)
    assert resumed["run_id"] == first["run_id"]
    assert resumed["reused"] == 1 and resumed["scored"] == 0
    Image.new("RGB", (12, 12), "black").save(manifest.parent / row["path"])
    fresh = output.with_name("rejected")
    with pytest.raises(ValueError, match="declared media SHA-256 mismatch"):
        await rescore_media(manifest, config, fresh)
    assert not fresh.exists()
    with pytest.raises(ValueError, match="declared media SHA-256 mismatch"):
        await rescore_media(manifest, config, output, resume=True)


@pytest.mark.asyncio
async def test_resume_rejects_changed_media_and_reward_recipe(scoring_input):
    manifest, config, output = scoring_input
    await rescore_media(manifest, config, output)
    changed = config.model_copy(update={"rubric_revision": "new"})
    with pytest.raises(ValueError, match="configuration changed"):
        await rescore_media(manifest, changed, output, resume=True)
    Image.new("RGB", (12, 12), "black").save(manifest.parent / "image.png")
    with pytest.raises(ValueError, match="configuration changed"):
        await rescore_media(manifest, config, output, resume=True)
    with evaluation_access(output, writing=True):
        pass


@pytest.mark.asyncio
async def test_failed_decode_is_recorded_without_a_fabricated_reward(scoring_input):
    manifest, config, output = scoring_input
    (manifest.parent / "image.png").write_bytes(b"invalid image")
    with pytest.raises(UnidentifiedImageError):
        await rescore_media(manifest, config, output)
    row = json.loads(next((output / "samples").glob("*.json")).read_text())
    assert row["status"] == "error"
    assert "result" not in row
    assert row["error"]["type"] == "UnidentifiedImageError"
    assert not (output / "summary.json").exists()
    with evaluation_access(output, writing=True):
        pass


@pytest.mark.asyncio
async def test_http_uploaded_media_matches_local_scoring(scoring_input):
    manifest, local, output = scoring_input
    service = RewardService(
        build_reward_scorer(local.worker_config),
        port=0,
        artifact_roots=(),
        model_name="sharpness",
        model_version="test-v1",
    )
    await service.start()
    host, port = service.address
    config = ScoringConfig(
        name="sharpness-http",
        revision="test-v1",
        preprocessing_revision="rgb-v1",
        rubric_revision="laplacian-v1",
        media_mode="tensor",
        inference=RewardInferenceConfig(
            kind="http",
            endpoint=f"http://{host}:{port}",
            expected_model="sharpness",
            expected_model_version="test-v1",
            timeout_s=10,
        ),
    )
    try:
        await rescore_media(manifest, config, output)
    finally:
        await service.shutdown_async()
    row = json.loads(next((output / "samples").glob("*.json")).read_text())
    assert row["status"] == "success"
    assert row["result"]["scores"] == {"image_sharpness": 0.0}


@pytest.mark.asyncio
async def test_writer_lock_prevents_concurrent_overwrite(scoring_input):
    manifest, config, output = scoring_input
    output.mkdir()
    lock = output / ".writer.lock"
    lock.write_text("another writer")
    with pytest.raises(FileExistsError):
        await rescore_media(manifest, config, output)
    assert lock.read_text() == "another writer"


@pytest.mark.asyncio
async def test_reference_assets_participate_in_resume_identity(scoring_input):
    manifest, config, output = scoring_input
    reference = manifest.parent / "reference.png"
    Image.new("RGB", (12, 12), "red").save(reference)
    row = json.loads(manifest.read_text())
    row["assets"] = {"reference_image": "reference.png"}
    manifest.write_text(json.dumps(row) + "\n")
    await rescore_media(manifest, config, output)
    Image.new("RGB", (12, 12), "blue").save(reference)
    with pytest.raises(ValueError, match="configuration changed"):
        await rescore_media(manifest, config, output, resume=True)
