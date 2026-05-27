"""Anthropic SDK-backed image QA reward model.

Uses the ``anthropic`` Python client directly (no subprocess). Auth errors
surface as ``PermissionError`` with a clear message pointing to
``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vrl.utils.media import write_png

DEFAULT_MODEL = "claude-opus-4-7"

DEFAULT_PROMPT_TEMPLATE = """You are a strict image-text alignment judge.
Evaluate whether the attached image matches the text prompt.
Return exactly one JSON object and no extra text:
{{"score": 0.37}}

Use a dense continuous score in [0, 1]. Do not collapse most samples to 0.
Reserve 0.0 only for blank, broken, or completely unrelated images.
Assign fine-grained decimals based on visible evidence; avoid repeated generic
scores such as 0.10, 0.12, or 0.50 when images differ.

Scoring rubric:
- 0.85-1.00: the image clearly matches the prompt.
- 0.60-0.84: the image mostly matches with minor missing details.
- 0.35-0.59: the image partially matches but misses important details.
- 0.10-0.34: the image is coherent but weakly related to the prompt.
- 0.00-0.09: the image is blank, broken, or completely unrelated.

Text prompt: {prompt}
"""


class AnthropicImageQARewardModel:
    """RewardModel returning ``{"anthropic_image_qa": score}`` per artifact.

    Requires ``ANTHROPIC_API_KEY`` to be set (or passed via ``api_key`` kwarg).
    Raises ``PermissionError`` with actionable instructions on auth failure.
    """

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for AnthropicImageQAReward. "
                "Install it with: pip install anthropic",
            ) from exc

        cfg = dict(worker_config)
        api_key = cfg.get("api_key")  # None → reads ANTHROPIC_API_KEY from env
        self._anthropic = _anthropic
        self._client = _anthropic.Anthropic(api_key=api_key) if api_key else _anthropic.Anthropic()
        self.model = cfg.get("model", DEFAULT_MODEL)
        self.max_tokens = int(cfg.get("max_tokens", 256))
        self.prompt_template = cfg.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
        self.max_concurrency = max(1, int(cfg.get("max_concurrency", 1)))

    def __call__(self, *, artifact: Any, request: Any) -> dict[str, float]:
        del request
        import tempfile

        prompt = artifact.prompt
        prompt_text = _render_prompt_template(self.prompt_template, prompt=prompt)
        with tempfile.TemporaryDirectory(prefix="vrl-anthropic-image-qa-") as tmp:
            image_path = Path(tmp) / "image.png"
            write_png(artifact.as_media(), image_path)
            image_b64 = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_b64,
                                },
                            },
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ],
            )
        except self._anthropic.AuthenticationError as exc:
            raise PermissionError(
                "Anthropic API authentication failed.\n"
                "  Set the ANTHROPIC_API_KEY environment variable:\n"
                "    export ANTHROPIC_API_KEY=sk-ant-...\n"
                f"Raw error: {exc}",
            ) from exc
        except self._anthropic.PermissionDeniedError as exc:
            raise PermissionError(
                f"Anthropic API permission denied (check your API key scopes): {exc}",
            ) from exc

        output_text = response.content[0].text if response.content else ""
        return {"anthropic_image_qa": _extract_score_from_text(output_text)}


def anthropic_image_qa_reward_model(worker_config: Mapping[str, Any]) -> AnthropicImageQARewardModel:
    return AnthropicImageQARewardModel(worker_config)


def _render_prompt_template(template: str, *, prompt: str) -> str:
    placeholder = "__VRL_ANTHROPIC_IMAGE_QA_PROMPT_PLACEHOLDER__"
    while placeholder in template:
        placeholder = f"_{placeholder}"
    rendered = template.replace("{prompt}", placeholder)
    rendered = rendered.replace("{{", "{").replace("}}", "}")
    return rendered.replace(placeholder, prompt)


def _extract_score_from_text(text: str) -> float:
    """Parse judge response into a clamped [0, 1] score."""
    import json as _json

    stripped = text.strip()
    if not stripped:
        raise ValueError("Anthropic image-QA returned empty output")

    json_value = _find_first_json_value(stripped)
    if json_value is not None:
        return _score_from_value(json_value)

    lowered = stripped.lower()
    if lowered.startswith("yes"):
        return 1.0
    if lowered.startswith("no"):
        return 0.0

    score_match = re.search(r'"?score"?\s*[:=]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))', stripped)
    if score_match:
        return _clamp_score(float(score_match.group(1)))

    try:
        return _clamp_score(float(stripped))
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse Anthropic image-QA score from output: {text!r}",
        ) from exc


def _find_first_json_value(text: str) -> Any | None:
    import json as _json

    decoder = _json.JSONDecoder()
    for idx, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[idx:])
        except _json.JSONDecodeError:
            continue
        return value
    return None


def _score_from_value(value: Any) -> float:
    if isinstance(value, dict):
        if "score" in value:
            return _clamp_score(float(value["score"]))
        if "answer" in value:
            return _score_answer(value["answer"])
        if "scores" in value:
            return _score_from_value(value["scores"])
        if "resultMap" in value:
            return _score_from_value(value["resultMap"])
    if isinstance(value, list):
        if not value:
            raise ValueError("Anthropic image-QA returned an empty score list")
        return _score_from_value(value[0])
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered.startswith(("yes", "no")):
            return _score_answer(value)
        return _clamp_score(float(value))
    return _clamp_score(float(value))


def _score_answer(answer: Any) -> float:
    text = str(answer).strip().lower()
    if text.startswith("yes"):
        return 1.0
    if text.startswith("no"):
        return 0.0
    raise ValueError(f"Cannot convert Anthropic image-QA answer to reward score: {answer!r}")


def _clamp_score(value: float) -> float:
    return min(max(value, 0.0), 1.0)


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_PROMPT_TEMPLATE",
    "AnthropicImageQARewardModel",
    "_extract_score_from_text",
    "_render_prompt_template",
    "anthropic_image_qa_reward_model",
]
