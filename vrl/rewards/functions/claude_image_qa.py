"""Claude CLI-backed image QA reward (model-backed over the local transport)."""

from __future__ import annotations

from vrl.rewards.base import RewardFunction
from vrl.rewards.models.claude_image_qa import (
    DEFAULT_COMMAND,
    DEFAULT_PROMPT_TEMPLATE,
    ClaudeImageQARewardModel,
    _extract_score_from_text,
    _render_prompt_template,
)
from vrl.rewards.runtime import LocalRewardRuntime


class ClaudeImageQAReward(RewardFunction):
    """Score image-text alignment by calling the Claude CLI as a judge.

    Sends image + prompt as a stream-json message to
    ``claude -p --input-format stream-json``. The command may contain
    ``{output_path}`` and ``{prompt}`` placeholders.
    """

    def __init__(
        self,
        device: str = "cuda",
        command: str | list[str] | tuple[str, ...] | None = None,
        timeout_s: float = 300.0,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        max_concurrency: int = 1,
    ) -> None:
        del device
        model = ClaudeImageQARewardModel(
            {
                "command": command if command is not None else DEFAULT_COMMAND,
                "timeout_s": timeout_s,
                "prompt_template": prompt_template,
                "max_concurrency": max_concurrency,
            },
        )
        self._model = model
        super().__init__(
            reward_name="claude_image_qa",
            score_key="claude_image_qa",
            runtime=LocalRewardRuntime(model=model),
            artifact_builder=lambda rollouts: RewardFunction.build_inmemory_artifacts(
                rollouts, media_type="image",
            ),
        )


__all__ = [
    "DEFAULT_COMMAND",
    "DEFAULT_PROMPT_TEMPLATE",
    "ClaudeImageQAReward",
    "_extract_score_from_text",
    "_render_prompt_template",
]
