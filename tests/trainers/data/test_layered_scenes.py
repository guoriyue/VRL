"""Cached layered scenes: token weights, spare layers, and the dataset over the cache contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from vrl.trainers.data.layered_scenes import (
    EMPTY_FRAME_FILE,
    LAYERED_SCENE_CACHE,
    LAYERED_SCENE_SCHEMA_VERSION,
    LayeredSceneDataset,
    append_spare_layers,
    collate_layered_scenes,
    token_weights,
)

_SIZE = 640
_TOKENS = (_SIZE // 16) ** 2


def _layer(path: Path, box) -> None:
    alpha = np.zeros((_SIZE, _SIZE), dtype=np.uint8)
    top, left, bottom, right = box
    alpha[top:bottom, left:right] = 255
    rgba = np.zeros((_SIZE, _SIZE, 4), dtype=np.uint8)
    rgba[..., 3] = alpha
    Image.fromarray(rgba, "RGBA").save(path)


def _scene(tmp_path: Path, *, inferable_box) -> Path:
    """A full-frame room layer behind one square object."""

    _layer(tmp_path / "layer_0.png", (0, 0, _SIZE, _SIZE))
    _layer(tmp_path / "layer_1.png", (160, 160, 480, 480))
    inferable = np.zeros((_SIZE, _SIZE), dtype=np.uint8)
    if inferable_box is not None:
        top, left, bottom, right = inferable_box
        inferable[top:bottom, left:right] = 255
    Image.fromarray(inferable, "L").save(tmp_path / "inferable_mask.png")
    return tmp_path


def test_inferable_hidden_tokens_carry_the_requested_share(tmp_path) -> None:
    # The inferable region lies inside the object, so it is hidden on the room layer.
    scene = _scene(tmp_path, inferable_box=(240, 240, 400, 400))

    weights, stats = token_weights(scene, 2, height=_SIZE, width=_SIZE, share=0.25)

    frames = weights.reshape(3, _TOKENS)
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))
    baseline = frames[0, 0]
    # Composite and the front object frame are never raised.
    torch.testing.assert_close(frames[0], torch.full_like(frames[0], baseline))
    torch.testing.assert_close(frames[2], torch.full_like(frames[2], baseline))
    raised = frames[1] - baseline
    assert raised.max() > 0
    # The raised part of the room layer is exactly the requested share of the loss.
    torch.testing.assert_close(raised.sum() / weights.sum(), torch.tensor(0.25), rtol=0, atol=1e-5)
    assert abs(stats["inferable_share"] - 0.25) < 1e-5


def test_visible_inferable_pixels_are_not_raised(tmp_path) -> None:
    # Inferable only where nothing covers the room layer: nothing is hidden there.
    scene = _scene(tmp_path, inferable_box=(0, 0, 80, 80))

    weights, stats = token_weights(scene, 2, height=_SIZE, width=_SIZE)

    torch.testing.assert_close(weights, torch.ones_like(weights))
    assert stats["k"] == 0.0


def test_spare_layers_append_transparent_targets_and_keep_mean_one() -> None:
    clean = torch.randn(1, 6, 4)
    raw = torch.tensor([1.0, 1.0, 3.0, 3.0, 1.0, 1.0]).reshape(1, 6, 1)
    weights = raw / raw.mean()  # the cache stores mean-one weights
    empty = torch.full((1, 2, 4), -0.5)

    stacked, stacked_weights = append_spare_layers(clean, weights, empty, 2)

    assert stacked.shape == (1, 10, 4)
    torch.testing.assert_close(stacked[:, :6], clean)
    torch.testing.assert_close(stacked[:, 6:], empty.repeat(1, 2, 1))
    torch.testing.assert_close(stacked_weights.mean(), torch.tensor(1.0))
    # Relative weights are preserved: the raised scene tokens stay three times the rest.
    torch.testing.assert_close(
        stacked_weights[0, 2, 0] / stacked_weights[0, 0, 0], torch.tensor(3.0)
    )
    # A spare token weighs as much as an average scene token.
    torch.testing.assert_close(stacked_weights[0, 6:], stacked_weights[0, :6].mean().expand(4, 1))


def test_no_spare_layers_leaves_the_scene_untouched() -> None:
    clean, weights, empty = torch.randn(1, 6, 4), torch.rand(1, 6, 1), torch.zeros(1, 2, 4)

    stacked, stacked_weights = append_spare_layers(clean, weights, empty, 0)

    assert stacked is clean and stacked_weights is weights


# ── The dataset over cache.pt ─────────────────────────────────────────────────

_T = 4  # tokens per frame of the toy caches


def _write_cache(root: Path, name: str, *, layers: int, revision: str = "rev") -> Path:
    scene = root / "train" / name
    scene.mkdir(parents=True)
    frames = layers + 1
    torch.save(
        {
            "schema": LAYERED_SCENE_SCHEMA_VERSION,
            "family": "qwen_image_layered",
            "model_revision": revision,
            "condition_latents": torch.full((1, _T, 8), float(layers)),
            "prompt_embeds": torch.zeros(1, 3, 5),
            "prompt_embeds_mask": None,
            "height": 32,
            "width": 32,
            "layers": layers,
            "target_latents": torch.arange(frames * _T * 8, dtype=torch.float32).reshape(
                1, frames * _T, 8
            ),
            "weights_weighted": torch.full((1, frames * _T, 1), 1.0),
            "weights_control": torch.full((1, frames * _T, 1), 1.0),
        },
        scene / LAYERED_SCENE_CACHE,
    )
    return scene


def _write_empty(root: Path, *, revision: str = "rev") -> None:
    torch.save(
        {"latent": torch.full((1, _T, 8), -1.0), "model_revision": revision},
        root / EMPTY_FRAME_FILE,
    )


def test_dataset_reads_targets_weights_and_conditioning(tmp_path) -> None:
    _write_cache(tmp_path, "scene_a", layers=2)
    _write_cache(tmp_path, "scene_b", layers=3)
    dataset = LayeredSceneDataset(
        {str(tmp_path): None}, split="train", token_weights="weighted", model_revision="rev"
    )

    assert len(dataset) == 2
    sample = dataset[1]
    assert sample.latents.shape == (1, 4 * _T, 8)
    assert sample.weights.shape == (1, 4 * _T, 1)
    assert sample.conditioning["layers"] == 3
    assert sample.conditioning["scene"] == "scene_b"
    torch.testing.assert_close(
        sample.conditioning["condition_latents"], torch.full((1, _T, 8), 3.0)
    )


def test_root_count_limits_the_scenes_drawn_from_it(tmp_path) -> None:
    for name in ("scene_a", "scene_b", "scene_c"):
        _write_cache(tmp_path, name, layers=2)

    assert len(LayeredSceneDataset({str(tmp_path): 2}, split="train")) == 2


def test_spare_layers_extend_the_budget_with_the_empty_frame(tmp_path) -> None:
    _write_cache(tmp_path, "scene_a", layers=2)
    _write_empty(tmp_path)
    dataset = LayeredSceneDataset({str(tmp_path): None}, split="train", spare_layers_max=1)

    torch.manual_seed(0)
    samples = [dataset[0] for _ in range(12)]

    spares = {sample.conditioning["layers"] - 2 for sample in samples}
    assert spares == {0, 1}
    with_spare = next(sample for sample in samples if sample.conditioning["layers"] == 3)
    assert with_spare.latents.shape == (1, 4 * _T, 8)
    torch.testing.assert_close(with_spare.latents[:, 3 * _T :], torch.full((1, _T, 8), -1.0))
    torch.testing.assert_close(with_spare.weights.mean(), torch.tensor(1.0))


def test_spare_layers_need_the_empty_frame(tmp_path) -> None:
    _write_cache(tmp_path, "scene_a", layers=2)
    with pytest.raises(FileNotFoundError, match=EMPTY_FRAME_FILE):
        LayeredSceneDataset({str(tmp_path): None}, split="train", spare_layers_max=1)


def test_cache_from_another_model_revision_is_refused(tmp_path) -> None:
    _write_cache(tmp_path, "scene_a", layers=2, revision="other")
    dataset = LayeredSceneDataset({str(tmp_path): None}, split="train", model_revision="rev")
    with pytest.raises(ValueError, match=r"model\.revision"):
        dataset[0]


def test_collate_stacks_one_layer_count_and_rejects_mixed_budgets(tmp_path) -> None:
    _write_cache(tmp_path, "scene_a", layers=2)
    _write_cache(tmp_path, "scene_b", layers=2)
    _write_cache(tmp_path, "scene_c", layers=3)
    dataset = LayeredSceneDataset({str(tmp_path): None}, split="train")

    batch = collate_layered_scenes([dataset[0], dataset[1]])

    assert batch.latents.shape == (2, 3 * _T, 8)
    assert batch.conditioning["condition_latents"].shape == (2, _T, 8)
    assert batch.conditioning["layers"] == 2
    with pytest.raises(ValueError, match="one layer count"):
        collate_layered_scenes([dataset[0], dataset[2]])
