"""Model-trained prompt and parser grammars for video reward judges."""

from __future__ import annotations

import re

VIDEOSCORE2_SYSTEM_PROMPT = (
    "You are an expert for evaluating AI-generated videos from three dimensions: "
    "(1) visual quality - clarity, smoothness, artifacts; "
    "(2) text-to-video alignment - fidelity to the prompt; "
    "(3) physical/common-sense consistency - naturalness and physics plausibility. "
    "Reason briefly, then output the final line exactly as: "
    "visual quality: <v_score>; text-to-video alignment: <t_score>, "
    "physical/common-sense consistency: <p_score>, "
    "where each score is an integer from 1 to 5."
)
VIDEOSCORE2_USER_TEMPLATE = (
    "Video prompt: {prompt}\n"
    "Please output in this format:\n"
    "visual quality: <v_score>; text-to-video alignment: <t_score>, "
    "physical/common-sense consistency: <p_score>"
)
VIDEOSCORE2_SCORE_REGEX = re.compile(
    r"visual quality:\s*(\d+).*?"
    r"text-to-video alignment:\s*(\d+).*?"
    r"physical/common-sense consistency:\s*(\d+)",
    re.IGNORECASE | re.DOTALL,
)

COSMOS3_SYSTEM_PROMPT = (
    "You are an expert evaluator of robot-manipulation and physical-interaction "
    "videos. Judge the video on four axes, each an integer from 1 to 5: "
    "(1) task success - does the action complete the instructed goal; "
    "(2) contact realism - grasps, contacts and forces are physically believable; "
    "(3) temporal consistency - object identity and motion are stable across frames; "
    "(4) physical plausibility - no clipping, teleporting, or physics violations. "
    "Reason briefly, then output the final line exactly as: "
    "task success: <s>; contact realism: <c>, temporal consistency: <t>, "
    "physical plausibility: <p>"
)
COSMOS3_USER_TEMPLATE = (
    "Task instruction: {prompt}\n"
    "Please output in this format:\n"
    "task success: <s>; contact realism: <c>, temporal consistency: <t>, "
    "physical plausibility: <p>"
)
COSMOS3_SCORE_REGEX = re.compile(
    r"task success:\s*(\d+).*?"
    r"contact realism:\s*(\d+).*?"
    r"temporal consistency:\s*(\d+).*?"
    r"physical plausibility:\s*(\d+)",
    re.IGNORECASE | re.DOTALL,
)

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
    "COSMOS3_SCORE_REGEX",
    "COSMOS3_SYSTEM_PROMPT",
    "COSMOS3_USER_TEMPLATE",
    "UNIFIED_REWARD_VIDEO_AXIS_PATTERNS",
    "UNIFIED_REWARD_VIDEO_PROBLEM_TEMPLATE",
    "VIDEOCON_PHYSICS_TEMPLATE",
    "VIDEOCON_SEMANTIC_TEMPLATE",
    "VIDEOSCORE2_SCORE_REGEX",
    "VIDEOSCORE2_SYSTEM_PROMPT",
    "VIDEOSCORE2_USER_TEMPLATE",
]
