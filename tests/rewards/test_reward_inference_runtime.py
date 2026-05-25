"""Tests for reward inference runtime helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from vrl.rewards.functions.video_reward import VideoReward
from vrl.rewards.inference import (
    RewardInferenceArtifact,
    RewardInferenceRequest,
    shard_reward_request,
)
from vrl.rewards.ray import RayRewardRuntime
from vrl.rewards.ray.worker import RewardModelWorker


def _request(count: int = 3) -> RewardInferenceRequest:
    return RewardInferenceRequest(
        request_id="req",
        artifacts=tuple(
            RewardInferenceArtifact(
                artifact_id=f"a{i}",
                path=f"/tmp/a{i}.pt",
                media_type="video",
            )
            for i in range(count)
        ),
        reward_name="reward",
        score_key="overall_reward",
    )


def test_shard_reward_request_balances_artifacts() -> None:
    shards = shard_reward_request(_request(5), num_shards=2)

    assert [len(shard.artifacts) for shard in shards] == [3, 2]
    assert [artifact.artifact_id for shard in shards for artifact in shard.artifacts] == [
        "a0",
        "a2",
        "a4",
        "a1",
        "a3",
    ]


def test_runtime_factory_rejects_missing_worker_config() -> None:
    with pytest.raises(ValueError, match="worker_config"):
        RayRewardRuntime({"inference_runtime": "ray"})


def test_reward_name_without_worker_config_is_not_a_worker_loader() -> None:
    with pytest.raises(ValueError, match="worker_config"):
        RayRewardRuntime(
            {
                "inference_runtime": "ray",
                "reward_name": "KlingTeam/VideoReward@main",
                "score_key": "overall_reward",
            },
        )


def test_video_reward_derives_internal_model_factory_from_reward_name(tmp_path) -> None:
    reward = VideoReward(
        reward_name="KlingTeam/VideoReward@main",
        score_key="overall_reward",
        artifact_dir=str(tmp_path),
        worker_config={"model_path": "", "dtype": "bfloat16"},
    )

    assert reward._actor_runtime.worker_config == {
        "model_path": "",
        "dtype": "bfloat16",
        "model_factory": "vrl.rewards.models.kling_video_reward:KlingVideoRewardModel",
        "reward_model_name": "KlingTeam/VideoReward@main",
        "reward_model_version": "KlingTeam/VideoReward@main",
    }


def test_worker_loads_reward_model_via_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeRewardModel:
        def __init__(self, worker_config):
            self.worker_config = worker_config

        def __call__(self, *, artifact, request):
            assert artifact.prompt == "prompt"
            assert request.score_key == "overall_reward"
            assert self.worker_config["reward_model_name"] == "KlingTeam/VideoReward@main"
            return {"overall_reward": 3.0, "motion_quality": 1.0}

    def _build(worker_config):
        return _FakeRewardModel(worker_config)

    monkeypatch.setattr(
        "vrl.rewards.ray.worker.import_from_path",
        lambda path: _build,
    )
    worker = RewardModelWorker(
        "reward-0",
        {
            "model_factory": "fake.module:build_model",
            "reward_model_name": "KlingTeam/VideoReward@main",
            "reward_model_version": "KlingTeam/VideoReward@main",
        },
    )
    request = RewardInferenceRequest(
        request_id="req",
        artifacts=(
            RewardInferenceArtifact(
                artifact_id="a0",
                path="/tmp/a0.mp4",
                media_type="video",
                prompt="prompt",
            ),
        ),
        reward_name="video_reward",
        score_key="overall_reward",
    )

    results = worker.score_batch(request)

    assert results[0].selected_score == pytest.approx(3.0)
    assert results[0].reward_model_version == "KlingTeam/VideoReward@main"


def test_worker_requires_explicit_model_factory_even_with_reward_model_name() -> None:
    with pytest.raises(ValueError, match="model_factory"):
        RewardModelWorker("reward-0", {})
    with pytest.raises(ValueError, match="model_factory"):
        RewardModelWorker(
            "reward-0",
            {"reward_model_name": "KlingTeam/VideoReward@main"},
        )


def test_kling_video_reward_snapshot_root_keeps_model_config_at_repo_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vrl.rewards.models.kling_video_reward import _resolve_model_root

    root = _video_reward_root(tmp_path)
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: str(root))

    resolved = _resolve_model_root({"reward_model_name": "KlingTeam/VideoReward@main"})

    assert resolved == root


def test_kling_video_reward_checkpoint_path_resolves_to_repo_root(tmp_path: Path) -> None:
    from vrl.rewards.models.kling_video_reward import _resolve_model_root

    root = _video_reward_root(tmp_path)

    resolved = _resolve_model_root({"model_path": str(root / "checkpoint-11352")})

    assert resolved == root


def test_kling_video_reward_requires_materialized_artifact_path() -> None:
    from vrl.rewards.models.kling_video_reward import KlingVideoRewardModel

    def _reward(*args, **kwargs):
        raise AssertionError("file-path validation should run before model scoring")

    model = KlingVideoRewardModel.__new__(KlingVideoRewardModel)
    model._reward = _reward
    model.use_norm = True
    artifact = RewardInferenceArtifact(
        artifact_id="a0",
        path="",
        media_type="video",
        media=object(),
        prompt="prompt",
    )
    request = RewardInferenceRequest(
        request_id="req",
        artifacts=(artifact,),
        reward_name="video_reward",
        score_key="overall_reward",
    )

    with pytest.raises(ValueError, match="no materialized path"):
        model(artifact=artifact, request=request)


def test_kling_video_reward_builds_repo_owned_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vrl.rewards.models import kling_video_reward as kling_reward
    from vrl.rewards.models.kling_video_reward import KlingVideoRewardModel

    captured = {}

    class _FakeModel:
        def eval(self):
            captured["model_eval"] = True

        def to(self, device):
            captured["model_device"] = device
            return self

    def _fake_create_model_and_processor(model_config, peft_config, *, dtype, disable_flash_attn2):
        del model_config, peft_config
        captured.update(
            {
                "dtype": dtype,
                "disable_flash_attn2": disable_flash_attn2,
            },
        )
        return _FakeModel(), object()

    def _fake_load_checkpoint(model, checkpoint_dir, checkpoint_step):
        captured.update(
            {
                "checkpoint_dir": checkpoint_dir,
                "checkpoint_step": checkpoint_step,
            },
        )
        return model, "final"

    root = _video_reward_root(tmp_path)
    monkeypatch.setattr(kling_reward, "_create_model_and_processor", _fake_create_model_and_processor)
    monkeypatch.setattr(kling_reward, "load_kling_video_reward_checkpoint", _fake_load_checkpoint)

    KlingVideoRewardModel(
        {
            "model_path": str(root),
            "device": "cpu",
            "dtype": "bfloat16",
            "disable_flash_attn2": True,
        },
    )

    assert captured == {
        "dtype": kling_reward.torch.bfloat16,
        "disable_flash_attn2": True,
        "checkpoint_dir": root,
        "checkpoint_step": -1,
        "model_eval": True,
        "model_device": "cpu",
    }


def test_kling_video_reward_remaps_qwen2vl_checkpoint_keys() -> None:
    from vrl.rewards.models.kling_video_reward import _remap_qwen2vl_key

    assert _remap_qwen2vl_key(
        "base_model.model.visual.patch_embed.proj.weight",
    ) == "base_model.model.model.visual.patch_embed.proj.weight"
    assert _remap_qwen2vl_key(
        "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight",
    ) == (
        "base_model.model.model.language_model.layers.0.self_attn.q_proj."
        "base_layer.weight"
    )
    assert _remap_qwen2vl_key(
        "base_model.model.model.embed_tokens.weight",
    ) == "base_model.model.model.language_model.embed_tokens.weight"
    assert _remap_qwen2vl_key(
        "base_model.model.lm_head.weight",
    ) == "base_model.model.lm_head.weight"


def _video_reward_root(tmp_path: Path) -> Path:
    root = tmp_path / "VideoReward"
    checkpoint = root / "checkpoint-11352"
    checkpoint.mkdir(parents=True)
    (root / "model_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.pth").write_bytes(b"")
    return root
