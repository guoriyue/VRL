"""The layered reward on hand-built layer stacks: grouping, completion, fidelity, row decoding."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.layered_grouping import (
    LayeredGroupingRewardModel,
    alpha_over,
    layers_from_media,
    visible_regions,
)

H = W = 32
GRAY, RED, BLUE = (0.5, 0.5, 0.5), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)


def _box(y0, y1, x0, x1):
    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _stack(masks, colors):
    """Opaque gray background plus one layer per mask, back to front, as [L, H, W, 4]."""

    layers = [np.concatenate([np.full((H, W, 3), GRAY), np.ones((H, W, 1))], -1)]
    for mask, color in zip(masks, colors, strict=True):
        layer = np.zeros((H, W, 4))
        layer[mask, :3] = color
        layer[mask, 3] = 1.0
        layers.append(layer)
    return np.stack(layers).astype(np.float32)


BALL = _box(4, 20, 4, 20)
CUBE = _box(12, 28, 12, 28)  # in front of the ball, hiding its lower right
EMPTY = np.zeros((H, W), dtype=bool)
OBJECTS = [BALL & ~CUBE, CUBE]  # what the source shows of each


def _grouping(layers):
    model = LayeredGroupingRewardModel({})
    return model.score_layers(layers, layers[0, ..., :3], OBJECTS)["grouping"]


def test_one_object_per_layer_scores_one_and_empty_layers_cost_nothing():
    assert _grouping(_stack([BALL, CUBE], [RED, BLUE])) == pytest.approx(1.0)
    assert _grouping(_stack([BALL, EMPTY, CUBE, EMPTY], [RED, RED, BLUE, BLUE])) == pytest.approx(
        1.0
    )


def test_splits_merges_and_objects_left_in_the_background_score_lower():
    left, right = _box(4, 20, 4, 12) & ~CUBE, _box(4, 20, 12, 20) & ~CUBE
    even_split = _grouping(_stack([left, right, CUBE], [RED, RED, BLUE]))
    sliver = _box(4, 20, 4, 6)
    uneven_split = _grouping(_stack([sliver, BALL & ~sliver, CUBE], [RED, RED, BLUE]))
    merged = _grouping(_stack([BALL | CUBE], [RED]))
    in_background = _grouping(_stack([CUBE], [BLUE]))

    assert max(even_split, merged, in_background) < 0.8
    # Continuous: a split that keeps most of the ball in one layer beats an even one.
    assert even_split < uneven_split < 1.0


def test_a_layer_of_unannotated_pixels_costs_nothing():
    plant = _box(24, 31, 1, 8)  # no mask names it
    with_plant = _grouping(_stack([BALL, CUBE, plant], [RED, BLUE, RED]))
    assert with_plant == pytest.approx(1.0)
    # The masks still count: half of the ball in that extra layer is a split.
    half = _box(4, 20, 4, 12) & ~CUBE
    assert _grouping(_stack([BALL & ~half, CUBE, half | plant], [RED, BLUE, RED])) < 0.9


def _truth_dir(tmp_path, layers, inferable):
    for index, layer in enumerate(layers):
        Image.fromarray((layer * 255).round().astype(np.uint8), "RGBA").save(
            tmp_path / f"layer_{index}.png"
        )
    Image.fromarray(inferable.astype(np.uint8) * 255).save(tmp_path / "inferable_mask.png")
    return tmp_path


def _artifact(tmp_path, layers, metadata):
    source = tmp_path / "source.png"
    Image.fromarray((alpha_over(layers) * 255).round().astype(np.uint8)).save(source)
    media = torch.from_numpy(layers).permute(0, 3, 1, 2).contiguous()
    return RewardInferenceArtifact(
        "a", "a", "", media=media, metadata={"reference_images": [str(source)], **metadata}
    )


def test_completion_grades_the_hidden_part_of_each_truth_object(tmp_path):
    truth = _stack([BALL, CUBE], [RED, BLUE])
    hidden = BALL & CUBE
    truth_dir = _truth_dir(tmp_path, truth, hidden)
    model = LayeredGroupingRewardModel({"completion_weight": 0.5})

    whole = model(_artifact(tmp_path, truth, {"truth_dir": str(truth_dir)}))
    cut = model(
        _artifact(
            tmp_path, _stack([BALL & ~CUBE, CUBE], [RED, BLUE]), {"truth_dir": str(truth_dir)}
        )
    )

    # The truth layers' visible regions are the objects, so both group perfectly;
    # only the layer that also draws the ball behind the cube completes it (the
    # cut one misses its alpha there and a third of its premultiplied color).
    assert whole["grouping"] == pytest.approx(1.0) and cut["grouping"] == pytest.approx(1.0)
    assert whole["completion"] == pytest.approx(1.0)
    assert cut["completion"] == pytest.approx(1.0 - 0.5 / 3 - 0.5)
    assert whole["layered_grouping"] == pytest.approx(1.0 + whole["fidelity"] + 0.5)
    assert whole["fidelity"] == pytest.approx(1.0, abs=0.005)  # 8-bit source


def test_a_row_scores_against_its_instance_masks(tmp_path):
    layers = _stack([BALL, CUBE], [RED, BLUE])
    np.savez(tmp_path / "instances.npz", masks=np.stack([*OBJECTS, EMPTY]))
    scores = LayeredGroupingRewardModel({})(
        _artifact(tmp_path, layers, {"instances": str(tmp_path / "instances.npz")})
    )
    assert scores["grouping"] == pytest.approx(1.0)
    assert (scores["objects"], scores["layers"], scores["completion"]) == (2.0, 2.0, 0.0)

    with pytest.raises(ValueError, match=r"instances or metadata\.truth_dir"):
        LayeredGroupingRewardModel({})(_artifact(tmp_path, layers, {}))


def test_the_reference_image_is_square_cropped_to_the_layer_size(tmp_path):
    wide = np.zeros((H, 2 * W, 3), dtype=np.uint8)
    wide[:, W // 2 : W // 2 + W] = 128  # the center square is the gray background
    path = tmp_path / "wide.png"
    Image.fromarray(wide).save(path)
    np.savez(tmp_path / "instances.npz", masks=np.stack(OBJECTS))
    layers = _stack([EMPTY], [RED])
    artifact = RewardInferenceArtifact(
        "a",
        "a",
        "",
        media=torch.from_numpy(layers).permute(0, 3, 1, 2).contiguous(),
        metadata={"reference_images": [str(path)], "instances": str(tmp_path / "instances.npz")},
    )
    # An uncropped stretch would bring the black margins in and cost fidelity.
    assert LayeredGroupingRewardModel({})(artifact)["fidelity"] == pytest.approx(1.0, abs=0.01)


def test_media_decodes_to_last_channel_layers():
    media = torch.zeros(3, 4, 8, 6, dtype=torch.uint8)
    media[1, 3, 2:5, 1:4] = 255
    media[1, 0, 2:5, 1:4] = 255
    layers = layers_from_media(media)
    assert layers.shape == (3, 8, 6, 4)
    assert layers[1, 3, 2].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert visible_regions(layers[..., 3])[1].sum() == 9
    with pytest.raises(ValueError, match="one sample"):
        layers_from_media(torch.zeros(2, 3, 4, 8, 6))
    with pytest.raises(ValueError, match=r"\[L, 4, H, W\]"):
        layers_from_media(torch.zeros(3, 3, 8, 6))


def test_dump_dir_writes_each_scored_sample(tmp_path):
    layers = _stack([BALL, CUBE], [RED, BLUE])
    np.savez(tmp_path / "instances.npz", masks=np.stack(OBJECTS))
    model = LayeredGroupingRewardModel({"dump_dir": str(tmp_path / "dumps")})
    scores = model(_artifact(tmp_path, layers, {"instances": str(tmp_path / "instances.npz")}))
    (dump,) = (tmp_path / "dumps").iterdir()
    assert sorted(p.name for p in dump.iterdir()) == [
        "layer_0.png",
        "layer_1.png",
        "layer_2.png",
        "scores.json",
        "source.png",
    ]
    assert json.loads((dump / "scores.json").read_text()) == pytest.approx(scores)
