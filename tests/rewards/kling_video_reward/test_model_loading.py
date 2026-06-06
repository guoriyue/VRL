"""Tests for KlingVideoReward model loading: repo-root resolution, checkpoint
paths, materialized-artifact validation, repo-owned model build, and Qwen2VL
checkpoint key remapping."""

from __future__ import annotations

from pathlib import Path

import pytest

from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest


def _video_reward_root(tmp_path: Path) -> Path:
    root = tmp_path / "VideoReward"
    checkpoint = root / "checkpoint-11352"
    checkpoint.mkdir(parents=True)
    (root / "model_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.pth").write_bytes(b"")
    return root


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
        reward_name="kling_video_reward",
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
