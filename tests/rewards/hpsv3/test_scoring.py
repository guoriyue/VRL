"""HPSv3 pure scoring logic — no 7B judge load.

Covers the three pieces where a silent mistake would ship a plausible but
wrong reward: the Flash-GRPO top-fraction frame aggregation, the pre-4.52
checkpoint key remap, and the byte-exact upstream prompt build.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
import torch

from vrl.rewards.assets.hpsv3_prompts import (
    HPSV3_REWARD_SPECIAL_TOKEN,
    build_hpsv3_frame_prompt,
)
from vrl.rewards.models.hpsv3 import (
    _aggregate_frame_scores,
    _remap_qwen2vl_state_dict,
)


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


class TestStateDictRemap:
    _NESTED_MODEL_KEYS: ClassVar[dict[str, None]] = {
        "model.language_model.embed_tokens.weight": None,
        "model.visual.patch_embed.proj.weight": None,
        "lm_head.weight": None,
        "rm_head.0.weight": None,
    }

    def test_flat_checkpoint_remaps_to_nested_layout(self) -> None:
        tensor = torch.zeros(1)
        state = {
            "model.embed_tokens.weight": tensor,
            "model.layers.0.self_attn.q_proj.weight": tensor,
            "model.norm.weight": tensor,
            "visual.patch_embed.proj.weight": tensor,
            "lm_head.weight": tensor,
            "rm_head.0.weight": tensor,
        }
        remapped = _remap_qwen2vl_state_dict(state, self._NESTED_MODEL_KEYS)
        assert set(remapped) == {
            "model.language_model.embed_tokens.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.norm.weight",
            "model.visual.patch_embed.proj.weight",
            "lm_head.weight",
            "rm_head.0.weight",
        }

    def test_already_nested_checkpoint_is_untouched(self) -> None:
        state = {"model.language_model.embed_tokens.weight": torch.zeros(1)}
        assert _remap_qwen2vl_state_dict(state, self._NESTED_MODEL_KEYS) is state

    def test_flat_target_model_is_untouched(self) -> None:
        state = {"model.embed_tokens.weight": torch.zeros(1)}
        flat_model_keys = {"model.embed_tokens.weight": None}
        assert _remap_qwen2vl_state_dict(state, flat_model_keys) is state


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
