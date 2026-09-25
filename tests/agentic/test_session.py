"""Chain sessions admit CPU verification and reject unsafe reward placement before models."""

import json
from argparse import Namespace
from unittest.mock import Mock

import pytest

import vrl.run
from agentic.scripts.session import open_session


@pytest.mark.asyncio
async def test_visual_reward_recipe_requires_explicit_remote_identity_and_cpu_local_placement(
    tmp_path, monkeypatch
):
    recipe = tmp_path / "reward.yaml"
    output = tmp_path / "output"
    output.mkdir()
    args = Namespace(
        reward_config=str(recipe),
        reward_revision="test-pixel-v1",
        reward_model=None,
        resolution=256,
        steps=8,
        output_mode="rgba",
        device="cpu",
        path="unused",
        revision="test",
    )
    allocate = Mock(side_effect=RuntimeError("reached editor allocation"))
    monkeypatch.setattr(vrl.run, "resolve_model", allocate)
    recipe.write_text("components: {image_sharpness: 1.0}\n")
    with pytest.raises(RuntimeError, match="reached editor allocation"):
        async with open_session(args, output=output):
            pytest.fail("fixture stops before allocating the editor")
    allocate.assert_called_once()
    assert json.loads((output / "reward_recipe.json").read_text())["recipe"]["components"] == {
        "image_sharpness": 1.0
    }
    allocate.reset_mock()
    recipe.write_text(
        "components: {image_sharpness: 1.0}\ninference: {image_sharpness: {kind: ray}}\n"
    )
    with pytest.raises(ValueError, match="CPU or operator-owned HTTP"):
        async with open_session(args, output=output):
            pytest.fail("invalid placement admitted")
    recipe.write_text(
        "components: {editreward: 1.0}\ninference:\n  editreward:\n    kind: http\n    endpoint: http://127.0.0.1:18315\n    expected_model: test\n"
    )
    with pytest.raises(ValueError, match="expected_model_version"):
        async with open_session(args, output=output):
            pytest.fail("unpinned service admitted")
    allocate.assert_not_called()

    # A calibration reference cannot silently fall back to weighted raw scores.
    recipe.write_text(
        json.dumps(
            {
                "components": {"image_sharpness": 1},
                "inference": {
                    "image_sharpness": {
                        "kind": "http",
                        "endpoint": "http://127.0.0.1:1",
                        "expected_model": "test",
                        "expected_model_version": "v1",
                    }
                },
                "calibration": {"deployment_path": str(tmp_path / "missing-receipt.json")},
            }
        )
    )
    with pytest.raises(FileNotFoundError, match="missing-receipt"):
        async with open_session(args, output=output):
            pytest.fail("missing calibration receipt admitted")
    allocate.assert_not_called()


@pytest.mark.asyncio
async def test_live_cpu_http_judge_needs_no_gpu_lease_but_unverified_service_is_rejected(
    tmp_path, monkeypatch
):
    from vrl.config.reward_inference import RewardInferenceConfig
    from vrl.rewards.runtime import build_reward_scorer
    from vrl.rewards.service.server import RewardService

    worker = {
        "model_factory": "vrl.rewards.models.image_sharpness:ImageSharpnessRewardModel",
        "device": "cpu",
    }
    service_config = tmp_path / "service.yaml"
    service_config.write_text(
        json.dumps(
            {
                "host": "127.0.0.1",
                "port": 0,
                "model_name": "cpu-sharpness",
                "model_version": "v1",
                "worker_config": worker,
                "artifact_roots": [str(tmp_path)],
            }
        )
    )
    service = RewardService.from_yaml(service_config)
    await service.start()
    recipe = tmp_path / "recipe.yaml"
    output = tmp_path / "output"
    output.mkdir()
    args = Namespace(
        reward_config=str(recipe),
        reward_revision="test-v1",
        reward_model=None,
        resolution=256,
        steps=8,
        output_mode="rgba",
        device="cpu",
        path="unused",
        revision="test",
    )
    allocate = Mock(side_effect=RuntimeError("reached editor allocation"))
    monkeypatch.setattr(vrl.run, "resolve_model", allocate)
    try:
        host, port = service.address
        recipe.write_text(
            json.dumps(
                {
                    "components": {"image_sharpness": 1.0},
                    "inference": {
                        "image_sharpness": {
                            "kind": "http",
                            "endpoint": f"http://{host}:{port}",
                            "expected_model": "cpu-sharpness",
                            "expected_model_version": "v1",
                        }
                    },
                }
            )
        )
        with pytest.raises(RuntimeError, match="reached editor allocation"):
            async with open_session(args, output=output):
                pytest.fail("fixture stops before model allocation")
        allocate.assert_called_once()
    finally:
        await service.shutdown_async()

    # A service without either advertised guarantee must fail, even though this
    # fixture happens to be implemented on CPU. The client cannot infer that.
    unverified = RewardService(
        build_reward_scorer(worker, inference=RewardInferenceConfig(kind="in_process")),
        host="127.0.0.1",
        port=0,
        artifact_roots=[tmp_path],
        model_name="unknown-placement",
        model_version="v1",
        generation_overlap_safe=False,
    )
    await unverified.start()
    allocate.reset_mock()
    try:
        host, port = unverified.address
        recipe.write_text(
            json.dumps(
                {
                    "components": {"image_sharpness": 1.0},
                    "inference": {
                        "image_sharpness": {
                            "kind": "http",
                            "endpoint": f"http://{host}:{port}",
                            "expected_model": "unknown-placement",
                            "expected_model_version": "v1",
                        }
                    },
                }
            )
        )
        with pytest.raises(ValueError, match="prove accelerator isolation or memory parking"):
            async with open_session(args, output=output):
                pytest.fail("unverified reward placement admitted")
        allocate.assert_not_called()
    finally:
        await unverified.shutdown_async()
