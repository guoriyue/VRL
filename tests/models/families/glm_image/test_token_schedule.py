"""GLM-Image grid math + mrope position-schedule tests.

The schedule contract: our pure ``glm_image_decode_position_schedule`` /
``glm_image_prefill_position_ids`` must reproduce byte-for-byte what the
reference ``GlmImageModel.get_rope_index`` computes (prefill position ids and
the cached ``_cached_decode_position_ids`` decode schedule) — that parity is
what keeps rollout, replay, and the upstream checkpoint's training-time
positions identical. Verified here against the REAL transformers
implementation on a tiny config.
"""

from __future__ import annotations

import pytest
import torch

from tests.models.families.glm_image.fixtures import (
    TINY_IMAGE_START_ID,
    tiny_hf_glm_image_config,
)
from vrl.models.families.glm_image.model import (
    glm_image_decode_position_schedule,
    glm_image_grid_dims,
    glm_image_prefill_position_ids,
    glm_image_token_num,
)

# Small non-square target: 128x192 -> large grid 4x6, preview 2x3.
HEIGHT, WIDTH = 128, 192


def test_grid_dims_match_processor_formula() -> None:
    # Mirrors GlmImageProcessor._build_prompt_with_target_shape math:
    # 128x192 -> large 4x6; preview int(sqrt(4/6)*16)=13 x int(sqrt(6/4)*16)=19.
    assert glm_image_grid_dims(HEIGHT, WIDTH) == (4, 6, 13, 19)
    # The real 1024x1024 default: 32x32 large, 16x16 preview.
    assert glm_image_grid_dims(1024, 1024) == (32, 32, 16, 16)
    # The checkpoint's documented 1152x768 default: 36x24 large, 19x13 preview.
    assert glm_image_grid_dims(1152, 768) == (36, 24, 19, 13)


def test_grid_dims_reject_non_multiple_of_32() -> None:
    with pytest.raises(ValueError, match="multiples of 32"):
        glm_image_grid_dims(1000, 1024)
    with pytest.raises(ValueError, match="multiples of 32"):
        glm_image_grid_dims(0, 1024)


def test_token_num_is_preview_plus_large_without_eos() -> None:
    # 32x32 + 16x16 = 1280 (the reference generates 1281 = +1 trailing EOS,
    # which is never consumed by the decode segment and is not sampled here).
    assert glm_image_token_num(1024, 1024) == 32 * 32 + 16 * 16
    token_h, token_w, prev_h, prev_w = glm_image_grid_dims(HEIGHT, WIDTH)
    assert glm_image_token_num(HEIGHT, WIDTH) == token_h * token_w + prev_h * prev_w


def _reference_positions(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Run the REAL transformers get_rope_index on a tiny GlmImageModel."""
    from transformers.models.glm_image.modeling_glm_image import GlmImageModel

    model = GlmImageModel(tiny_hf_glm_image_config())
    token_h, token_w, prev_h, prev_w = glm_image_grid_dims(HEIGHT, WIDTH)
    grids = torch.tensor([[1, token_h, token_w], [1, prev_h, prev_w]])
    batch = input_ids.shape[0]
    all_grids = grids.repeat(batch, 1)
    positions, _deltas = model.get_rope_index(
        input_ids,
        image_grid_thw=all_grids,
        images_per_sample=torch.tensor([2] * batch),
        attention_mask=attention_mask,
    )
    return positions, model._cached_decode_position_ids


def test_prefill_and_decode_schedule_match_reference_get_rope_index() -> None:
    # t2i prompt: text ids ending with image_start; row 1 is left-padded.
    input_ids = torch.tensor(
        [
            [30, 31, 32, 33, 34, TINY_IMAGE_START_ID],
            [0, 0, 30, 31, 32, TINY_IMAGE_START_ID],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
        ]
    )
    ref_prefill, ref_decode = _reference_positions(input_ids, attention_mask)

    ours_prefill = glm_image_prefill_position_ids(attention_mask)
    # Valid positions must match exactly; padding slots are attention-masked
    # (the reference leaves them at the init value 1, and so do we).
    valid = attention_mask == 1
    for axis in range(3):
        assert torch.equal(ours_prefill[axis][valid], ref_prefill[axis][valid])
    assert torch.equal(ours_prefill, ref_prefill)

    token_h, token_w, prev_h, prev_w = glm_image_grid_dims(HEIGHT, WIDTH)
    grids = ((token_h, token_w), (prev_h, prev_w))
    for row, prompt_len in enumerate((6, 4)):
        ours = glm_image_decode_position_schedule(prompt_len, grids)
        # Reference schedule includes the trailing EOS end-marker entry.
        assert ours.shape == (3, prev_h * prev_w + token_h * token_w + 1)
        assert torch.equal(ours, ref_decode[row])


def test_decode_schedule_is_translation_equivariant_in_prompt_len() -> None:
    # The runner exploits schedule(n) == schedule(0) + n for per-row offsets.
    grids = ((4, 6), (2, 3))
    base = glm_image_decode_position_schedule(0, grids)
    shifted = glm_image_decode_position_schedule(7, grids)
    assert torch.equal(shifted, base + 7)


def test_decode_schedule_orders_preview_before_large() -> None:
    grids = ((4, 6), (2, 3))  # (large, preview) — processor grid order
    schedule = glm_image_decode_position_schedule(10, grids)
    # First 6 entries: preview raster at base 10 (temporal constant).
    assert (schedule[0, :6] == 10).all()
    assert schedule[1, :6].tolist() == [10, 10, 10, 11, 11, 11]
    assert schedule[2, :6].tolist() == [10, 11, 12, 10, 11, 12]
    # Next 24 entries: large raster at base 10 + max(2, 3) = 13.
    assert (schedule[0, 6:30] == 13).all()
    # End marker: base 13 + max(4, 6) = 19 on all axes.
    assert schedule[:, 30].tolist() == [19, 19, 19]


def test_decode_schedule_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="grids must be non-empty"):
        glm_image_decode_position_schedule(0, ())
    with pytest.raises(ValueError, match="grid dims"):
        glm_image_decode_position_schedule(0, ((0, 3),))
    with pytest.raises(ValueError, match="prompt_len"):
        glm_image_decode_position_schedule(-1, ((2, 3),))
