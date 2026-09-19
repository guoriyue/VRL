"""Model-trained prompt and parser grammars for video reward judges.

Each block pairs a judge's evaluation prompt with the regex/pattern that parses
its fixed output format; prompt and parser define one grammar and must change in
lockstep, which is why they live side by side here instead of inside the model
modules. Consumers: ``vrl.rewards.models.unified_reward_video`` / ``videocon_physics``. The
UnifiedReward and VideoCon templates follow the upstream inference scripts
those checkpoints ship with; treat every string as checkpoint-calibrated —
rewording a prompt silently shifts the score distribution.
"""

from __future__ import annotations

import re

UNIFIED_REWARD_VIDEO_PROBLEM_TEMPLATE = (
    "You are presented with a generated video and its associated text caption. "
    "Your task is to analyze the video across multiple dimensions in relation to "
    "the caption. Specifically:\n"
    "Provide overall assessments for the video along the following axes "
    "(each rated from 1 to 5):\n"
    "- Alignment Score: How well the video matches the caption in terms of content.\n"
    "- Physics Score: How well the gravity, movements, collisions, and interactions "
    "make physical sense.\n"
    "- Style Score: How visually appealing the video looks, regardless of caption "
    "accuracy.\n\n"
    "Output your evaluation using the format below:\n\n"
    "Alignment Score (1-5): X\n"
    "Physics Score (1-5): Y\n"
    "Style Score (1-5): Z\n\n"
    "Your task is provided as follows:\n"
    "Text Caption: [{prompt}]"
)
_NUMBER_PATTERN = r"([0-9]+(?:\.[0-9]+)?)"
UNIFIED_REWARD_VIDEO_AXIS_PATTERNS = {
    axis: re.compile(
        rf"{label} Score\s*\(\s*{_NUMBER_PATTERN}\s*-\s*{_NUMBER_PATTERN}\s*\)"
        rf"\s*:\s*{_NUMBER_PATTERN}",
        re.IGNORECASE,
    )
    for axis, label in (
        ("alignment", "Alignment"),
        ("physics", "Physics"),
        ("style", "Style"),
    )
}

VIDEOCON_PHYSICS_TEMPLATE = "\nHuman: {video}\nDoes the video follow physical commonsense?\nAI: "
VIDEOCON_SEMANTIC_TEMPLATE = (
    '\nHuman: {video}\nDoes the video entail the caption: "{caption}"?\nAI: '
)

__all__ = [
    "UNIFIED_REWARD_VIDEO_AXIS_PATTERNS",
    "UNIFIED_REWARD_VIDEO_PROBLEM_TEMPLATE",
    "VIDEOCON_PHYSICS_TEMPLATE",
    "VIDEOCON_SEMANTIC_TEMPLATE",
]
