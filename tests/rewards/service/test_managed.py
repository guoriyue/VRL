"""Managed reward service: the driver-launched subprocess transport."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.service.managed import ManagedRewardScorer
from vrl.utils.artifacts import sha256_file

FAKE_FACTORY = "tests.rewards.service._fake_reward_model:FakeRewardModel"


def _scorer(tmp_path: Path, **overrides) -> ManagedRewardScorer:
    return ManagedRewardScorer(
        RewardInferenceConfig(kind="service", timeout_s=120),
        worker_config={"model_factory": FAKE_FACTORY, "device": "cpu", "scale": 2.0, **overrides},
        artifact_dir=tmp_path / "run" / "reward_artifacts",
        component_name="fake",
    )


def test_service_kind_allows_an_unset_endpoint_and_identity() -> None:
    cfg = RewardInferenceConfig(kind="service")
    assert cfg.endpoint == "" and cfg.expected_model == ""
    pinned = RewardInferenceConfig(kind="service", endpoint="http://127.0.0.1:8399")
    assert pinned.endpoint == "http://127.0.0.1:8399"
    with pytest.raises(ValueError, match="requires expected_model"):
        RewardInferenceConfig(kind="http", endpoint="http://127.0.0.1:8399")


def test_managed_scorer_writes_the_service_config_it_launches(tmp_path: Path) -> None:
    scorer = _scorer(tmp_path)
    config = scorer.service_config()
    host, port = scorer.host_port
    assert (host, port) == ("127.0.0.1", port) and port > 0
    assert config["worker_config"] == {
        "model_factory": FAKE_FACTORY,
        "device": "cpu",
        "scale": 2.0,
        "reward_model_version": FAKE_FACTORY,
    }
    assert config["artifact_roots"] == [str((tmp_path / "run" / "reward_artifacts").resolve())]
    assert config["model_name"] == f"fake:{FAKE_FACTORY}"
    assert scorer.deployment.expected_model == config["model_name"]
    assert scorer.config_path.parent == (tmp_path / "run").resolve()
    assert scorer.pid is None
    # A shared-GPU topology hands the parking contract to the service.
    parking = _scorer(tmp_path, sleep_offload=True, device="cuda:0")
    assert parking.service_config()["worker_config"]["sleep_offload"] is True
    assert parking.service_config()["generation_overlap_safe"] is False


@pytest.mark.slow_test
@pytest.mark.asyncio
async def test_managed_scorer_launches_scores_and_terminates_the_subprocess(
    tmp_path: Path,
) -> None:
    scorer = _scorer(tmp_path)
    artifact_dir = scorer.artifact_dir
    artifact_dir.mkdir(parents=True)
    path = artifact_dir / "s0.pt"
    torch.save(torch.full((3, 4, 4), 0.25), path)
    request = RewardInferenceRequest(
        request_id="managed-test",
        artifacts=[
            RewardInferenceArtifact(
                artifact_id="a0",
                sample_id="s0",
                path=str(path),
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )
        ],
    )
    try:
        await scorer.ensure_ready()
        pid = scorer.pid
        assert pid is not None and scorer.config_path.exists()
        results = await scorer.score_batch(request)
        assert [round(r.scores["overall"], 6) for r in results] == [0.5]
        assert results[0].reward_model_version == scorer.model_version
    finally:
        await scorer.shutdown()
    assert scorer.pid is None
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert scorer.log_path.exists()
