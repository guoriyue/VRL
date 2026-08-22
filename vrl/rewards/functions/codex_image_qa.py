"""Codex-CLI image-QA reward (rubric-driven LLM judge, scored in-process).

Thin function-layer binding registered as ``codex_image_qa``. The judging logic
(render prompt/command, run ``codex exec`` on a temp PNG, parse the score) lives
in ``vrl.rewards.models.codex_image_qa``. The rubric in ``prompt_template`` is
the reward definition — point it at whatever axis you need (alignment, anatomy
plausibility, crisp cel-shading). Each score is a subprocess call, so it runs on
CPU; keep ``max_concurrency`` in line with your CLI rate limits.
"""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import InferenceRewardFunction
from vrl.rewards.models.codex_image_qa import (
    DEFAULT_PROMPT_TEMPLATE,
    CodexImageQARewardModel,
    _extract_score_from_text,
    _render_prompt_template,
)
from vrl.rewards.runtime import InProcessRewardScorer


class CodexImageQAReward(InferenceRewardFunction):
    """Score an image against a rubric by calling the Codex (or compatible) CLI."""

    def __init__(self, **kwargs: Any) -> None:
        # CLI subprocess judge; ``device`` is accepted for a uniform factory
        # signature but never used.
        kwargs.pop("device", None)
        model = CodexImageQARewardModel(kwargs)
        super().__init__(
            reward_name="codex_image_qa",
            score_key="codex_image_qa",
            scorer=InProcessRewardScorer(model=model),
        )


__all__ = [
    "DEFAULT_PROMPT_TEMPLATE",
    "CodexImageQAReward",
    "_extract_score_from_text",
    "_render_prompt_template",
]
