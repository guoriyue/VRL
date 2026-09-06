"""Kling VideoReward scoring math on a tiny real Qwen2-VL reward model.

``KlingQwen2VLRewardModel.forward`` selects reward-head logits per pooling
branch and ``KlingVideoRewardModel._reward`` z-scores the three axes; neither
had a test. Every expectation here is recomputed from the same model's
per-token head logits (``head_logits``), never a hard-coded float.
"""

from __future__ import annotations

import pytest
import torch

from tests.rewards.kling_video_reward.fixtures import (
    build_tiny_kling_reward_model,
    head_logits,
)

pytest.importorskip("transformers")

_PAD = 0


def _mask(input_ids: torch.Tensor) -> torch.Tensor:
    return (input_ids != _PAD).long()


def test_last_pooling_reads_the_final_non_pad_row() -> None:
    """``sequence_lengths = argmax(ids == pad) - 1`` lands on the last real token."""

    model = build_tiny_kling_reward_model()
    ids = torch.tensor([[3, 4, 5, _PAD, _PAD], [7, 8, 9, 10, _PAD]])

    pooled = model(input_ids=ids, attention_mask=_mask(ids))["logits"]

    reference = head_logits(model, ids, _mask(ids))
    assert pooled.shape == (2, 4)
    assert torch.equal(pooled[0], reference[0, 2])
    assert torch.equal(pooled[1], reference[1, 3])


def test_last_pooling_wraps_to_the_final_row_when_no_pad_is_present() -> None:
    """No pad -> argmax 0 -> -1 -> ``% length`` wraps to the last row.

    Naively "fixing" the off-by-one would break exactly this half of the contract.
    """

    model = build_tiny_kling_reward_model()
    ids = torch.tensor([[3, 4, 5, 6, 7]])

    pooled = model(input_ids=ids, attention_mask=_mask(ids))["logits"]

    assert torch.equal(pooled[0], head_logits(model, ids, _mask(ids))[0, 4])


def test_mean_pooling_drops_the_final_real_token_and_ignores_padding() -> None:
    """``mean`` averages ``logits[:sequence_length]`` -- the prefix BEFORE the
    last real token, not the whole valid span. The behaviour is pinned as-is so a
    reader cannot mistake it for a valid-prefix mean."""

    model = build_tiny_kling_reward_model(reward_token="mean")
    ids = torch.tensor([[3, 4, 5, _PAD, _PAD], [7, 8, 9, 10, _PAD]])

    pooled = model(input_ids=ids, attention_mask=_mask(ids))["logits"]

    reference = head_logits(model, ids, _mask(ids))
    assert torch.allclose(pooled[0], reference[0, :2].mean(dim=0))
    assert not torch.allclose(pooled[0], reference[0, :3].mean(dim=0))
    # Padding is invisible: the same rows without trailing pad give the same pooled value.
    unpadded = torch.tensor([[3, 4, 5]])
    assert torch.allclose(
        model(input_ids=unpadded, attention_mask=_mask(unpadded))["logits"][0],
        pooled[0],
        atol=1e-6,
    )


def test_special_pooling_takes_the_diagonal_in_sequence_order() -> None:
    """Boolean-mask indexing returns rows in SEQUENCE order, not ``special_token_ids``
    order; passing ``special_token_ids`` also overrides ``reward_token`` unconditionally."""

    model = build_tiny_kling_reward_model(output_dim=3, special_token_ids=[1, 2, 3])
    ids = torch.tensor([[1, 5, 2, 6, 3, 7, _PAD], [7, 1, 6, 2, 5, 3, _PAD]])

    pooled = model(input_ids=ids, attention_mask=_mask(ids))["logits"]

    reference = head_logits(model, ids, _mask(ids))
    assert model.reward_token == "special"
    assert pooled.shape == (2, 3)
    expected_row_1 = torch.stack([reference[1, 1], reference[1, 3], reference[1, 5]]).diagonal()
    assert torch.equal(pooled[1], expected_row_1)


def test_special_pooling_with_output_dim_one_yields_three_axes() -> None:
    """The shipped checkpoint is ``output_dim=1`` + special tokens: three special
    rows of one logit each become the (B, 3) VQ/MQ/TA vector ``_reward`` reads."""

    model = build_tiny_kling_reward_model(output_dim=1, special_token_ids=[1, 2, 3])
    ids = torch.tensor([[1, 5, 2, 6, 3, 7, _PAD]])

    pooled = model(input_ids=ids, attention_mask=_mask(ids))["logits"]

    reference = head_logits(model, ids, _mask(ids))
    assert pooled.shape == (1, 3)
    assert torch.equal(pooled[0], reference[0, [0, 2, 4], 0])


def test_pooling_rejects_unpadded_batches_and_unknown_reward_tokens() -> None:
    no_pad = build_tiny_kling_reward_model(pad_token_id=None)
    ids = torch.tensor([[3, 4, 5], [6, 7, 8]])
    with pytest.raises(ValueError, match="no padding token"):
        no_pad(input_ids=ids, attention_mask=torch.ones_like(ids))

    bogus = build_tiny_kling_reward_model(reward_token="bogus")
    with pytest.raises(ValueError, match="Invalid reward_token"):
        bogus(input_ids=ids, attention_mask=torch.ones_like(ids))


def _scoring_model(tiny: torch.nn.Module, inference_config: dict[str, float] | None):
    """A ``KlingVideoRewardModel`` around ``tiny`` with ``_prepare_batch`` stubbed to
    a fixed token batch, the same construction ``test_model_loading.py`` uses."""

    from vrl.rewards.models.kling_video_reward import KlingVideoRewardModel

    ids = torch.tensor([[1, 5, 2, 6, 3, 7, _PAD]])
    batch = {"input_ids": ids, "attention_mask": _mask(ids)}
    model = KlingVideoRewardModel.__new__(KlingVideoRewardModel)
    model.model = tiny
    model.device = "cpu"
    model.inference_config = inference_config
    model._prepare_batch = lambda video_paths, prompts, max_pixels=None, min_pixels=None: batch
    return model, batch


def test_reward_z_scores_each_axis_and_sums_overall() -> None:
    """Three DIFFERENT (mean, std) pairs: swapping axes or mean/std cannot pass."""

    tiny = build_tiny_kling_reward_model(output_dim=1, special_token_ids=[1, 2, 3])
    inference_config = {
        "VQ_mean": 1.0,
        "VQ_std": 2.0,
        "MQ_mean": -0.5,
        "MQ_std": 4.0,
        "TA_mean": 0.25,
        "TA_std": 0.5,
    }
    model, batch = _scoring_model(tiny, inference_config)

    (reward,) = model._reward(["clip.mp4"], ["prompt"], use_norm=True)

    with torch.no_grad():
        logits = tiny(return_dict=True, **batch)["logits"][0]
    expected_vq = (float(logits[0]) - 1.0) / 2.0
    expected_mq = (float(logits[1]) + 0.5) / 4.0
    expected_ta = (float(logits[2]) - 0.25) / 0.5
    assert reward["VQ"] == pytest.approx(expected_vq)
    assert reward["MQ"] == pytest.approx(expected_mq)
    assert reward["TA"] == pytest.approx(expected_ta)
    assert reward["Overall"] == pytest.approx(expected_vq + expected_mq + expected_ta)


def test_reward_without_norm_keeps_the_raw_head_logits() -> None:
    """``use_norm=False`` (or a checkpoint without inference_config) scores raw."""

    tiny = build_tiny_kling_reward_model(output_dim=1, special_token_ids=[1, 2, 3])
    model, batch = _scoring_model(tiny, {"VQ_mean": 9.0, "VQ_std": 9.0})

    (raw,) = model._reward(["clip.mp4"], ["prompt"], use_norm=False)
    model.inference_config = None
    (missing_config,) = model._reward(["clip.mp4"], ["prompt"], use_norm=True)

    with torch.no_grad():
        logits = tiny(return_dict=True, **batch)["logits"][0]
    assert raw["VQ"] == pytest.approx(float(logits[0]))
    assert raw["Overall"] == pytest.approx(float(logits.sum()))
    assert missing_config == raw
