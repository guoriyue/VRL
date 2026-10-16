"""CPU tests for the layered scene encoder: cropping, frame stacking and the cache it writes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vrl.scripts.denoise import encode_layered_scenes
from vrl.scripts.denoise.encode_layered_scenes import crop_photo, encode_scene, scene_frames

_SIZE = 64
_TOKENS = (_SIZE // 16) ** 2


class _FakeLayeredModel:
    """Records what the encoder asks of the family model and answers with shaped zeros."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def encode_prompt(self, prompt, negative_prompt=None, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return {
            "prompt_embeds": torch.zeros(1, 3, 5),
            "prompt_embeds_mask": None,
            "condition_latents": torch.zeros(1, _TOKENS, 8, dtype=torch.bfloat16),
        }

    def encode_video_to_latents(self, video):
        self.calls.append({"video": tuple(video.shape)})
        return torch.zeros(1, video.shape[2] * _TOKENS, 8)


def _rgba(path: Path, box, color=(200, 30, 30)) -> None:
    image = np.zeros((_SIZE, _SIZE, 4), dtype=np.uint8)
    top, left, bottom, right = box
    image[top:bottom, left:right, :3] = color
    image[top:bottom, left:right, 3] = 255
    Image.fromarray(image, "RGBA").save(path)


def _scene(root: Path, *, inferable: bool) -> Path:
    scene = root / "scene"
    scene.mkdir()
    Image.new("RGB", (_SIZE, _SIZE), (10, 20, 30)).save(scene / "source.png")
    _rgba(scene / "layer_0.png", (0, 0, _SIZE, _SIZE), (10, 20, 30))
    _rgba(scene / "layer_1.png", (16, 16, 48, 48))
    if inferable:
        mask = np.zeros((_SIZE, _SIZE), dtype=np.uint8)
        mask[24:40, 24:40] = 255  # hidden under the object on layer 0
        Image.fromarray(mask, "L").save(scene / "inferable_mask.png")
    return scene


def test_crop_photo_keeps_the_center_at_the_sampling_aspect(tmp_path) -> None:
    photo = np.zeros((60, 100, 3), dtype=np.uint8)
    photo[:, 20:80] = 255  # the central 60x60 is white
    Image.fromarray(photo, "RGB").save(tmp_path / "photo.png")

    row = crop_photo(tmp_path / "photo.png", tmp_path / "scene", height=32, width=32)

    cropped = np.asarray(Image.open(tmp_path / "scene" / "source.png"))
    assert cropped.shape == (32, 32, 3) and cropped.min() == 255
    assert row == {
        "scene": "scene",
        "photo": str(tmp_path / "photo.png"),
        "original_size": [100, 60],
    }


def test_scene_frames_put_the_opaque_source_before_the_layers(tmp_path) -> None:
    scene = _scene(tmp_path, inferable=False)

    frames = scene_frames(scene, 2)

    assert frames.shape == (1, 4, 3, _SIZE, _SIZE)
    assert frames[0, 3, 0].min() == 1.0  # the composite frame is opaque
    assert frames[0, 3, 2].mean() == pytest.approx((32 * 32) / (_SIZE * _SIZE))
    assert frames[0, 0, 2, 20, 20] == pytest.approx(200 / 255)


def test_encode_scene_caches_conditioning_targets_and_raised_weights(tmp_path) -> None:
    scene = _scene(tmp_path, inferable=True)
    model = _FakeLayeredModel()

    cache = encode_scene(model, scene, prompt="a clean image", height=_SIZE, width=_SIZE)

    assert model.calls[0]["prompt"] == "a clean image"
    assert (model.calls[0]["height"], model.calls[0]["width"]) == (_SIZE, _SIZE)
    assert model.calls[1]["video"] == (1, 4, 3, _SIZE, _SIZE)
    assert cache["layers"] == 2
    assert cache["target_latents"].shape == (1, 3 * _TOKENS, 8)
    assert cache["weights_weighted"].shape == (1, 3 * _TOKENS, 1)
    assert cache["weights_weighted"].max() > 1.0  # the hidden inferable tokens are raised
    torch.testing.assert_close(cache["weights_control"], torch.ones(1, 3 * _TOKENS, 1))
    assert cache["weight_stats"]["inferable_share"] == pytest.approx(0.25)
    assert cache["condition_latents"].dtype == torch.bfloat16 and len(cache["source_sha256"]) == 64


def test_scene_without_inferable_mask_gets_uniform_weights(tmp_path) -> None:
    cache = encode_scene(
        _FakeLayeredModel(),
        _scene(tmp_path, inferable=False),
        prompt="p",
        height=_SIZE,
        width=_SIZE,
    )

    torch.testing.assert_close(cache["weights_weighted"], torch.ones(1, 3 * _TOKENS, 1))
    assert cache["weight_stats"] == {}


def test_truth_free_photo_needs_a_layer_budget_and_caches_no_target(tmp_path) -> None:
    scene = tmp_path / "scene_r00"
    scene.mkdir()
    Image.new("RGB", (_SIZE, _SIZE)).save(scene / "source.png")
    model = _FakeLayeredModel()

    with pytest.raises(ValueError, match="--layers"):
        encode_scene(model, scene, prompt="p", height=_SIZE, width=_SIZE)
    cache = encode_scene(model, scene, prompt="p", height=_SIZE, width=_SIZE, layers=6)

    assert cache["layers"] == 6 and "target_latents" not in cache
    assert len(model.calls) == 1  # one prompt encode, no target encode


def test_source_at_another_size_is_refused(tmp_path) -> None:
    scene = _scene(tmp_path, inferable=False)
    with pytest.raises(ValueError, match="sampling size"):
        encode_scene(_FakeLayeredModel(), scene, prompt="p", height=32, width=32)


def test_entrypoint_rejects_ambiguous_inputs_before_loading_anything(monkeypatch) -> None:
    monkeypatch.setattr(
        encode_layered_scenes,
        "build_parser",
        lambda: SimpleNamespace(
            parse_args=lambda argv: SimpleNamespace(
                root=Path("scenes"), photos=[Path("a.jpg")], out=None, layers=None
            )
        ),
    )
    with pytest.raises(ValueError, match="exactly one of --root or --photos"):
        encode_layered_scenes.main([])
