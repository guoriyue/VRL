"""Regression tests for the Kling VideoReward checkpoint prompt protocol."""

import pytest

from vrl.rewards.models.kling_prompt_templates import (
    build_kling_video_reward_prompt,
)


def test_none_template_preserves_the_generation_prompt() -> None:
    assert build_kling_video_reward_prompt("a red fox", "VQ", "none") == "a red fox"


def test_simple_template_describes_one_requested_dimension() -> None:
    result = build_kling_video_reward_prompt("a red fox", "VQ", "simple")

    assert "visual quality" in result
    assert "clearness, resolution, brightness, and color" in result
    assert '"a red fox"' in result


def test_video_score_template_combines_requested_dimensions() -> None:
    result = build_kling_video_reward_prompt(
        "a red fox",
        ["VQ", "MQ"],
        "video_score",
    )

    assert "overall performance(visual quality, motion quality)" in result
    assert "the overall performance of the video" in result
    assert '"a red fox"' in result


def test_detailed_special_template_includes_all_reward_tokens() -> None:
    result = build_kling_video_reward_prompt(
        "a red fox",
        ["VQ", "MQ", "TA"],
        "detailed_special",
    )

    assert "Textual prompt - a red fox" in result
    assert "<|VQ_reward|>" in result
    assert "<|MQ_reward|>" in result
    assert "<|TA_reward|>" in result


def test_detailed_template_includes_all_categories_without_reward_tokens() -> None:
    result = build_kling_video_reward_prompt(
        "a red fox",
        ["VQ", "MQ", "TA"],
        "detailed",
    )

    assert "**Visual Quality:**" in result
    assert "**Motion Quality:**" in result
    assert "**Text Alignment:**" in result
    assert "Textual prompt - a red fox" in result
    assert "<|VQ_reward|>" not in result


@pytest.mark.parametrize(
    ("dimension", "template_type", "message"),
    [
        ([], "detailed", "non-empty"),
        ("unknown", "none", "dimension"),
        ("VQ", "unknown", "prompt template"),
    ],
)
def test_invalid_checkpoint_prompt_protocol_is_rejected(
    dimension: str | list[str],
    template_type: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_kling_video_reward_prompt(
            "a red fox",
            dimension,
            template_type,
        )
