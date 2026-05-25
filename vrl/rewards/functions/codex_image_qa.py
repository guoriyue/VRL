"""Codex-backed image QA reward (model-backed over the local transport)."""

from __future__ import annotations

from vrl.rewards.base import RewardFunction
from vrl.rewards.models.codex_image_qa_model import (
    DEFAULT_PROMPT_TEMPLATE,
    CodexImageQARewardModel,
    _extract_score_from_text,
    _render_prompt_template,
    _write_image_png,
)
from vrl.rewards.runtime import LocalRewardRuntime


class CodexImageQAReward(RewardFunction):
    """Score image-text alignment by calling Codex or a compatible CLI judge.

    The command may contain ``{image_path}``, ``{output_path}``, and ``{prompt}``
    placeholders. The rendered prompt is sent to stdin by default, which matches
    ``codex exec --image {image_path} --output-last-message {output_path} -``.
    """

    def __init__(
        self,
        device: str = "cuda",
        command: str | list[str] | tuple[str, ...] | None = None,
        timeout_s: float = 300.0,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        pass_prompt_stdin: bool = True,
        max_concurrency: int = 1,
    ) -> None:
        del device
        # Build eagerly so config validation (command required) fires now.
        model = CodexImageQARewardModel(
            {
                "command": command,
                "timeout_s": timeout_s,
                "prompt_template": prompt_template,
                "pass_prompt_stdin": pass_prompt_stdin,
                "max_concurrency": max_concurrency,
            },
        )
        self._model = model
        super().__init__(
            reward_name="codex_image_qa",
            score_key="codex_image_qa",
            runtime=LocalRewardRuntime(model=model),
            artifact_builder=lambda rollouts: RewardFunction.build_inmemory_artifacts(
                rollouts, media_type="image",
            ),
        )


__all__ = [
    "DEFAULT_PROMPT_TEMPLATE",
    "CodexImageQAReward",
    "_extract_score_from_text",
    "_render_prompt_template",
    "_write_image_png",
]
