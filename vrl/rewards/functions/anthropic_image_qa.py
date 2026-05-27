"""Anthropic SDK-backed image QA reward (model-backed over the local transport)."""

from __future__ import annotations

from vrl.rewards.base import RewardFunction
from vrl.rewards.models.anthropic_image_qa import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT_TEMPLATE,
    AnthropicImageQARewardModel,
    _extract_score_from_text,
    _render_prompt_template,
)
from vrl.rewards.runtime import LocalRewardRuntime


class AnthropicImageQAReward(RewardFunction):
    """Score image-text alignment via the Anthropic Python SDK.

    Requires ``ANTHROPIC_API_KEY`` in the environment. Raises ``PermissionError``
    with clear instructions when the key is missing or invalid.
    """

    def __init__(
        self,
        device: str = "cuda",
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 256,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        max_concurrency: int = 1,
    ) -> None:
        del device
        reward_model = AnthropicImageQARewardModel(
            {
                "api_key": api_key,
                "model": model,
                "max_tokens": max_tokens,
                "prompt_template": prompt_template,
                "max_concurrency": max_concurrency,
            },
        )
        self._model = reward_model
        super().__init__(
            reward_name="anthropic_image_qa",
            score_key="anthropic_image_qa",
            runtime=LocalRewardRuntime(model=reward_model),
            artifact_builder=lambda rollouts: RewardFunction.build_inmemory_artifacts(
                rollouts, media_type="image",
            ),
        )


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_PROMPT_TEMPLATE",
    "AnthropicImageQAReward",
    "_extract_score_from_text",
    "_render_prompt_template",
]
