"""CPU preflight tests for the offline SFT target encoder."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from vrl.scripts.denoise import encode_targets
from vrl.scripts.denoise.encode_targets import (
    _resolve_clean_targets,
    _target_at_sampling_geometry,
    _weight_map_at_latent_geometry,
)
from vrl.trainers.data.prompts import PromptExample


def test_target_preflight_uses_artifact_root_and_target_identity(tmp_path) -> None:
    root = tmp_path / "artifacts"
    (root / "targets").mkdir(parents=True)
    (root / "targets/a.mp4").touch()
    (root / "targets/b.mp4").touch()
    examples = [
        PromptExample(prompt="same instruction", target_video="targets/a.mp4"),
        PromptExample(prompt="same instruction", target_video="targets/b.mp4"),
    ]

    targets = _resolve_clean_targets(examples, data_root=root)

    assert [key for key, _, _, _ in targets] == ["targets/a.mp4", "targets/b.mp4"]
    assert [path for _, path, _, _ in targets] == [
        str(root / "targets/a.mp4"),
        str(root / "targets/b.mp4"),
    ]
    assert [kind for _, _, kind, _ in targets] == ["target_video", "target_video"]
    assert [mask for _, _, _, mask in targets] == [None, None]


def test_target_preflight_resolves_weight_mask_beside_target(tmp_path) -> None:
    root = tmp_path / "artifacts"
    (root / "targets").mkdir(parents=True)
    (root / "targets/a.png").touch()
    (root / "targets/a_mask.png").touch()
    examples = [
        PromptExample(
            prompt="complete the cabinet",
            target_image="targets/a.png",
            metadata={"sft_weight_mask": "targets/a_mask.png"},
        ),
    ]

    [(key, _path, _kind, mask)] = _resolve_clean_targets(examples, data_root=root)

    assert key == "targets/a.png"
    assert mask == str(root / "targets/a_mask.png")


def test_target_preflight_rejects_missing_weight_mask(tmp_path) -> None:
    (tmp_path / "a.png").touch()
    example = PromptExample(
        prompt="p", target_image="a.png", metadata={"sft_weight_mask": "nope.png"}
    )
    with pytest.raises(FileNotFoundError, match="sft_weight_mask does not exist"):
        _resolve_clean_targets([example], data_root=tmp_path)


def test_weight_map_pools_to_latent_grid_and_normalizes_to_mean_one(monkeypatch) -> None:
    # 4x4 mask: bright top-left quadrant, black elsewhere -> 2x2 latent grid.
    mask = torch.zeros(1, 4, 4, 3)
    mask[:, :2, :2, :] = 1.0
    monkeypatch.setattr("vrl.utils.media.read_image_as_frames", lambda _path: mask)

    weights = _weight_map_at_latent_geometry("mask.png", (16, 3, 2, 2))

    assert weights.shape == (1, 1, 2, 2)
    torch.testing.assert_close(weights[0, 0], torch.tensor([[4.0, 0.0], [0.0, 0.0]]))
    assert float(weights.mean()) == 1.0


def test_weight_map_rejects_all_black_mask(monkeypatch) -> None:
    monkeypatch.setattr(
        "vrl.utils.media.read_image_as_frames", lambda _path: torch.zeros(1, 4, 4, 3)
    )
    with pytest.raises(ValueError, match="all black"):
        _weight_map_at_latent_geometry("mask.png", (4, 2, 2))


def test_target_preflight_rejects_duplicate_target_identity(tmp_path) -> None:
    (tmp_path / "target.mp4").touch()
    examples = [
        PromptExample(prompt="first", target_video="target.mp4"),
        PromptExample(prompt="second", target_video="target.mp4"),
    ]

    with pytest.raises(ValueError, match="repeats clean target"):
        _resolve_clean_targets(examples, data_root=tmp_path)


def test_target_preflight_rejects_missing_file(tmp_path) -> None:
    example = PromptExample(prompt="missing", target_video="targets/missing.mp4")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _resolve_clean_targets([example], data_root=tmp_path)


def test_video_geometry_rejects_short_target(monkeypatch) -> None:
    monkeypatch.setattr(
        "vrl.utils.media.read_video_frames",
        lambda _path, *, num_frames: torch.zeros(num_frames - 1, 4, 4, 3),
    )

    with pytest.raises(ValueError, match="requires 3"):
        _target_at_sampling_geometry(
            "short.mp4",
            media_type="target_video",
            height=4,
            width=4,
            num_frames=3,
        )


def test_video_geometry_resizes_without_changing_frame_count(monkeypatch) -> None:
    monkeypatch.setattr(
        "vrl.utils.media.read_video_frames",
        lambda _path, *, num_frames: torch.zeros(num_frames, 2, 3, 3),
    )

    video = _target_at_sampling_geometry(
        "target.mp4",
        media_type="target_video",
        height=4,
        width=6,
        num_frames=3,
    )

    assert video.shape == (1, 3, 3, 4, 6)


def test_image_target_becomes_one_frame_video(monkeypatch) -> None:
    monkeypatch.setattr(
        "vrl.utils.media.read_image_as_frames",
        lambda _path: torch.zeros(1, 2, 3, 3),
    )

    video = _target_at_sampling_geometry(
        "target.png",
        media_type="target_image",
        height=4,
        width=6,
        num_frames=1,
    )

    assert video.shape == (1, 3, 1, 4, 6)


def test_entrypoint_rejects_malformed_model_before_registry_model_build(
    monkeypatch,
    tmp_path,
) -> None:
    import vrl.models.families.registry as family_registry
    from vrl.config import loading as config_loading

    monkeypatch.setattr(
        config_loading,
        "load_config",
        lambda *_args, **_kwargs: OmegaConf.create(
            {"model": ["not", "a", "mapping"]},
        ),
    )
    monkeypatch.setattr(
        family_registry,
        "get_model_family_entry",
        lambda _family: pytest.fail("registry model build started before config validation"),
    )

    with pytest.raises(ValueError, match="model must be a mapping"):
        encode_targets.main(
            [
                "--experiment",
                "invalid",
                "--out",
                str(tmp_path / "latents.pt"),
            ],
        )


def test_entrypoint_forwards_composition_overrides(monkeypatch, tmp_path) -> None:
    from vrl.config import loading as config_loading

    expected_overrides = [
        "+reward=wd_tagger",
        "+reward=pickscore",
        "+dataset=anime_craft",
        "model.use_lora=false",
        "sampling.num_steps=40",
    ]

    def capture_load_config(path, *, overrides):
        assert path == "experiment/sd3_5/online_grpo_pickscore"
        assert overrides == expected_overrides
        raise RuntimeError("configuration arguments captured before model loading")

    monkeypatch.setattr(config_loading, "load_config", capture_load_config)

    with pytest.raises(RuntimeError, match="configuration arguments captured"):
        encode_targets.main(
            [
                "--experiment",
                "sd3_5/online_grpo_pickscore",
                "--out",
                str(tmp_path / "latents.pt"),
                *expected_overrides,
            ],
        )
