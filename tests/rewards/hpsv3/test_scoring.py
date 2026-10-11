"""HPSv3 pure scoring logic — no 7B judge load.

Covers the three pieces where a silent mistake would ship a plausible but
wrong reward: the Flash-GRPO top-fraction frame aggregation, the pre-4.52
checkpoint key remap, and the byte-exact upstream prompt build.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from vrl.rewards.assets.hpsv3_prompts import (
    HPSV3_REWARD_SPECIAL_TOKEN,
    build_hpsv3_frame_prompt,
)
from vrl.rewards.models.hpsv3 import _aggregate_frame_scores
from vrl.rewards.models.hub import relocate_checkpoint_keys


class TestAggregateFrameScores:
    def test_top_fraction_means_best_frames(self) -> None:
        """10 frames at 0.3 -> mean of the 3 best, descending order."""
        scores = [float(v) for v in (1, 9, 2, 8, 3, 7, 4, 6, 5, 10)]
        out = _aggregate_frame_scores(scores, 0.3)
        assert out["top_frame_mean"] == pytest.approx((10 + 9 + 8) / 3)
        assert out["frame_mean"] == pytest.approx(5.5)
        assert out["frame_min"] == pytest.approx(1.0)

    def test_fraction_rounds_down_like_upstream(self) -> None:
        """int(l * fraction) truncation, matching flow_grpo's int(l*0.3)."""
        out = _aggregate_frame_scores([4.0, 2.0, 3.0, 1.0, 5.0], 0.3)
        # int(5 * 0.3) == 1 -> only the single best frame.
        assert out["top_frame_mean"] == pytest.approx(5.0)

    def test_short_videos_keep_at_least_one_frame(self) -> None:
        out = _aggregate_frame_scores([2.0], 0.3)
        assert out["top_frame_mean"] == pytest.approx(2.0)
        assert out["frame_min"] == pytest.approx(2.0)

    def test_empty_scores_raise(self) -> None:
        with pytest.raises(ValueError, match="no frames"):
            _aggregate_frame_scores([], 0.3)


class TestCheckpointKeyRelocation:
    """The HPSv3 checkpoint predates the Qwen2-VL nesting; the overlay must land strict."""

    @staticmethod
    def _tiny_qwen2vl() -> Any:
        from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration

        config = Qwen2VLConfig(
            text_config={
                "vocab_size": 64,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "rope_scaling": {"type": "mrope", "mrope_section": [2, 2, 4]},
                "bos_token_id": None,
                "eos_token_id": None,
            },
            vision_config={
                "depth": 1,
                "embed_dim": 32,
                "hidden_size": 32,
                "num_heads": 2,
                "in_channels": 3,
                "patch_size": 14,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
            bos_token_id=None,
            eos_token_id=None,
        )
        torch.manual_seed(0)
        return Qwen2VLForConditionalGeneration(config)

    def test_flat_checkpoint_strict_loads_after_relocation(self) -> None:
        source = self._tiny_qwen2vl()
        flat = {
            key.replace("model.language_model.", "model.", 1).replace(
                "model.visual.", "visual.", 1
            ): value
            for key, value in source.state_dict().items()
        }
        assert flat.keys() != source.state_dict().keys()

        target = self._tiny_qwen2vl()
        target.load_state_dict(relocate_checkpoint_keys(target, flat), strict=True)
        for key, value in source.state_dict().items():
            assert torch.equal(target.state_dict()[key], value), key

    def test_nested_checkpoint_is_returned_unchanged(self) -> None:
        model = self._tiny_qwen2vl()
        state = model.state_dict()
        assert relocate_checkpoint_keys(model, state) is state


class TestPromptBuild:
    def test_prompt_embeds_text_and_one_reward_token(self) -> None:
        text = build_hpsv3_frame_prompt("a red fox on mossy stones")
        assert "Textual prompt - a red fox on mossy stones" in text
        assert text.count(HPSV3_REWARD_SPECIAL_TOKEN) == 1

    def test_prompt_keeps_upstream_scaffolding(self) -> None:
        """The wording the checkpoint was trained against must survive edits."""
        text = build_hpsv3_frame_prompt("p")
        assert "**Visual Quality:**" in text
        assert "**Text Alignment:**" in text
        assert text.rstrip().endswith("END")
