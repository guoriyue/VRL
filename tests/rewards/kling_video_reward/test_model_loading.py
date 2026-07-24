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
    """Checks Kling video reward snapshot root keeps model config at repo root."""
    from vrl.rewards.models.kling_video_reward import _resolve_model_root

    root = _video_reward_root(tmp_path)
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: str(root))

    resolved = _resolve_model_root({"reward_model_name": "KlingTeam/VideoReward@main"})

    assert resolved == root


def test_kling_video_reward_checkpoint_path_resolves_to_repo_root(tmp_path: Path) -> None:
    """Checks Kling video reward checkpoint path resolves to repo root."""
    from vrl.rewards.models.kling_video_reward import _resolve_model_root

    root = _video_reward_root(tmp_path)

    resolved = _resolve_model_root({"model_path": str(root / "checkpoint-11352")})

    assert resolved == root


def test_kling_video_reward_requires_materialized_artifact_path() -> None:
    """Checks Kling video reward requires materialized artifact path."""
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
    """Checks Kling video reward builds repo owned model."""
    from vrl.rewards.models import kling_video_reward as kling_reward
    from vrl.rewards.models.kling_video_reward import KlingVideoRewardModel

    captured = {}

    class _FakeModel:
        def eval(self):
            captured["model_eval"] = True

        def to(self, device):
            captured["model_device"] = device
            return self

    def _fake_create_model_and_processor(
        model_config,
        peft_config,
        *,
        dtype,
        disable_flash_attn2,
        local_files_only,
    ):
        del model_config, peft_config
        captured.update(
            {
                "dtype": dtype,
                "disable_flash_attn2": disable_flash_attn2,
                "local_files_only": local_files_only,
            },
        )
        return _FakeModel(), object()

    def _fake_load_checkpoint(model, checkpoint_dir):
        captured["checkpoint_dir"] = checkpoint_dir
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
            "local_files_only": True,
        },
    )

    assert captured == {
        "dtype": kling_reward.torch.bfloat16,
        "disable_flash_attn2": True,
        "local_files_only": True,
        "checkpoint_dir": root,
        "model_eval": True,
        "model_device": "cpu",
    }


def test_kling_video_reward_parses_frame_pixel_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks min/max frame pixel bounds parse from worker_config."""
    from vrl.rewards.models import kling_video_reward as kling_reward
    from vrl.rewards.models.kling_video_reward import KlingVideoRewardModel

    class _FakeModel:
        def eval(self):
            return None

        def to(self, device):
            del device
            return self

    monkeypatch.setattr(
        kling_reward,
        "_create_model_and_processor",
        lambda *args, **kwargs: (_FakeModel(), object()),
    )
    monkeypatch.setattr(
        kling_reward,
        "load_kling_video_reward_checkpoint",
        lambda model, checkpoint_dir: (model, "final"),
    )

    root = _video_reward_root(tmp_path)
    model = KlingVideoRewardModel(
        {
            "model_path": str(root),
            "device": "cpu",
            "disable_flash_attn2": True,
            "min_frame_pixels": 200704,
        },
    )

    assert model.min_frame_pixels == 200704
    assert model.max_frame_pixels is None  # checkpoint budget stays in charge


def test_kling_video_reward_snapshot_download_honors_local_files_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks Kling video reward passes local-only mode to snapshot download."""
    from vrl.rewards.models.kling_video_reward import _resolve_model_root

    root = _video_reward_root(tmp_path)
    captured = {}

    def _fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(root)

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot_download)

    resolved = _resolve_model_root(
        {
            "reward_model_name": "KlingTeam/VideoReward@main",
            "local_files_only": True,
        },
    )

    assert resolved == root
    assert captured["repo_id"] == "KlingTeam/VideoReward"
    assert captured["revision"] == "main"
    assert captured["local_files_only"] is True


def test_kling_video_reward_base_loader_honors_local_files_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks Kling video reward keeps base model loading offline when requested."""
    from vrl.rewards.models import kling_video_reward as kling_reward

    captured = {}

    class _FakeTokenizer:
        padding_side = "right"
        pad_token_id = 0

    class _FakeProcessor:
        tokenizer = _FakeTokenizer()

    class _FakeModelConfig:
        pass

    class _FakeModel:
        config = _FakeModelConfig()

        def to(self, dtype):
            captured["to_dtype"] = dtype
            return self

    class _FakeAutoProcessor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            captured["processor"] = (args, kwargs)
            return _FakeProcessor()

    def _fake_from_pretrained(*args, **kwargs):
        captured["model"] = (args, kwargs)
        return _FakeModel()

    monkeypatch.setattr(
        kling_reward.KlingQwen2VLRewardModel,
        "from_pretrained",
        staticmethod(_fake_from_pretrained),
    )
    monkeypatch.setattr(
        "transformers.AutoProcessor",
        _FakeAutoProcessor,
    )

    model_config = kling_reward._ModelConfig(
        model_name_or_path="Qwen/Qwen2-VL-2B-Instruct",
        model_revision="main",
        torch_dtype="bfloat16",
    )
    peft_config = kling_reward._PeftLoraConfig(lora_enable=False)

    model, processor = kling_reward._create_model_and_processor(
        model_config,
        peft_config,
        dtype=kling_reward.torch.bfloat16,
        disable_flash_attn2=True,
        local_files_only=True,
    )

    assert isinstance(model, _FakeModel)
    assert isinstance(processor, _FakeProcessor)
    assert captured["processor"][1]["local_files_only"] is True
    assert captured["processor"][1]["revision"] == "main"
    assert captured["model"][1]["local_files_only"] is True
    assert captured["model"][1]["revision"] == "main"


def test_kling_video_reward_remaps_qwen2vl_checkpoint_keys() -> None:
    """Checks Kling video reward remaps Qwen2vl checkpoint keys."""
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


def test_kling_normalize_scores_renames_drops_missing_and_never_leaks() -> None:
    """_normalize_scores renames raw VQ/MQ/TA/Overall to public aliases, drops a
    public key when its raw source is absent, and never leaks raw/undocumented keys.

    Expectations are written as literal dicts (not derived from ``_SCORE_KEY_MAP``)
    so the test fails if the mapping or the ``if model_key in raw`` filter regresses.
    """
    from vrl.rewards.models.kling_video_reward import _normalize_scores

    # Full raw set -> all four public aliases, values carried by the rename.
    assert _normalize_scores({"VQ": 1.0, "MQ": 2.0, "TA": 3.0, "Overall": 6.0}) == {
        "overall_reward": 6.0,
        "visual_quality": 1.0,
        "motion_quality": 2.0,
        "text_alignment": 3.0,
    }

    # Missing raw source -> the corresponding public key is dropped, not defaulted.
    assert _normalize_scores({"VQ": 1.0, "Overall": 4.0}) == {
        "visual_quality": 1.0,
        "overall_reward": 4.0,
    }

    # Undocumented raw key -> never crosses into the public scoring contract.
    assert _normalize_scores({"VQ": 1.0, "BOGUS": 9.0}) == {"visual_quality": 1.0}
