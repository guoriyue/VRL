"""local_edit scoring with the two models replaced at their seams: DINOv2 patch tokens read off the
pixels of a textured scene, and the execution scorer returning a fixed value."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from vrl.rewards.models.local_edit import LocalEditRewardModel

W = H = 200
CELL = 20
BOX = [0.1, 0.4, 0.3, 0.6]  # x 20-60, y 80-120: the region the instruction may touch
SPEC = {"task": "attribute_modification", "box": BOX, "hint": False}


def _scene(patch: tuple[int, int, int, int] | None, shift: int = 0, seed: int = 0) -> Image.Image:
    noise = torch.Generator().manual_seed(seed)
    texture = torch.randint(40, 200, (H, W, 3), generator=noise).to(torch.uint8)
    image = Image.frombytes("RGB", (W, H), texture.numpy().tobytes())
    if patch:
        image.paste((200, 30, 30), patch)
    if shift:
        shifted = Image.new("RGB", (W, H), (0, 0, 0))
        shifted.paste(image, (shift, 0))
        return shifted
    return image


class _Stubbed(LocalEditRewardModel):
    def __init__(self) -> None:
        super().__init__({"device": "cpu"})

    def _patches(self, image):
        # Each cell's 4x4 sub-block colours stand in for its DINOv2 token (48-d, so two
        # unrelated textures almost never agree by chance): identical pixels match at 1.
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(H, W, 3)
        sub = CELL // 4
        blocks = (
            pixels.float().reshape(H // CELL, 4, sub, W // CELL, 4, sub, 3).mean((2, 5)) - 120.0
        )
        cells = blocks.permute(0, 2, 1, 3, 4).reshape(H // CELL, W // CELL, -1)
        return torch.nn.functional.normalize(cells, dim=-1)


def test_an_edit_inside_the_box_keeps_everything_and_scores_the_execution() -> None:
    out = _Stubbed().score(_scene(None), _scene((20, 80, 60, 120)), SPEC, execution=0.81)
    assert out["local_edit_keep"] == pytest.approx(1.0)
    assert out["local_edit"] == pytest.approx(0.9)
    assert out["local_edit_shaped"] == pytest.approx((0.9 + 0.2) / 1.2)


def test_a_change_outside_the_box_a_shifted_frame_and_a_redraw_lose_keep() -> None:
    model = _Stubbed()
    spill = model.score(_scene(None), _scene((120, 80, 160, 120)), SPEC, execution=1.0)
    assert spill["local_edit_kept_share"] < 1.0
    shifted = model.score(_scene(None), _scene((20, 80, 60, 120), shift=20), SPEC, execution=1.0)
    assert shifted["local_edit_keep"] == 0.0 and shifted["local_edit"] == 0.0
    redrawn = model.score(_scene(None), _scene(None, seed=1), SPEC, execution=1.0)
    assert redrawn["local_edit_keep"] < 0.05 and redrawn["local_edit_shaped"] < 0.05


def test_a_phase_is_one_execution_request_against_the_clean_source(tmp_path) -> None:
    from vrl.rewards.inference import RewardInferenceArtifact
    from vrl.scripts.data.local_edit import HINT_SUFFIX

    source = _scene(None)
    source.save(tmp_path / "src.jpg")
    calls: list[tuple[list[str], list[str]]] = []

    class _Batched(_Stubbed):
        def _executions(self, artifacts, instructions, source_paths):
            calls.append((list(instructions), list(source_paths)))
            return [0.81, 0.0]

    def artifact(index: int, image: Image.Image) -> RewardInferenceArtifact:
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        return RewardInferenceArtifact(
            artifact_id=f"a{index}",
            sample_id=f"s{index}",
            path="",
            prompt="paint the box red" + HINT_SUFFIX,
            metadata={
                "reference_images": [str(tmp_path / "hint.jpg")],
                "local_edit": {**SPEC, "source_image": str(tmp_path / "src.jpg")},
            },
            media=pixels.reshape(H, W, 3).permute(2, 0, 1).float() / 255.0,
        )

    out = _Batched().score_batch([artifact(0, _scene((20, 80, 60, 120))), artifact(1, source)])
    # One request for the phase, carrying the plain instruction and the clean source.
    assert calls == [(["paint the box red", "paint the box red"], [str(tmp_path / "src.jpg")] * 2)]
    assert out[0]["local_edit"] == pytest.approx(0.9)
    assert out[1]["local_edit"] == 0.0 and out[1]["local_edit_keep"] == pytest.approx(1.0)


def test_a_faithful_no_op_keeps_only_the_floor_and_the_spec_is_validated() -> None:
    model = _Stubbed()
    noop = model.score(_scene(None), _scene(None), SPEC, execution=0.0)
    assert noop["local_edit"] == 0.0 and noop["local_edit_shaped"] == pytest.approx(0.2 / 1.2)
    with pytest.raises(ValueError, match="box"):
        model.score(_scene(None), _scene(None), {"task": "swap"}, execution=0.5)
    with pytest.raises(ValueError, match="equal size"):
        model.score(_scene(None), Image.new("RGB", (10, 10)), SPEC, execution=0.5)
