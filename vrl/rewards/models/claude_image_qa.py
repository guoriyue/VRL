"""Claude CLI-backed image QA reward model.

Mirrors CodexImageQARewardModel but sends image + prompt together as a
stream-json message via stdin, since ``claude`` has no ``--image`` flag.
The image is base64-encoded inline in the JSON payload.
"""

from __future__ import annotations

import base64
import json
import re
import shlex
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vrl.utils.media import write_png

DEFAULT_COMMAND = (
    "claude -p --input-format stream-json --output-format text"
    ' --tools "" --no-session-persistence'
)

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


class ClaudeImageQARewardModel:
    """RewardModel returning ``{"claude_image_qa": score}`` per artifact.

    Image and prompt are packed together as a stream-json user message and sent
    to ``claude -p --input-format stream-json`` via stdin. The command may
    contain ``{output_path}`` and ``{prompt}`` placeholders.
    """

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        cfg = dict(worker_config)
        command = cfg.get("command", DEFAULT_COMMAND)
        self.command = _normalize_command(command)
        self.timeout_s = float(cfg.get("timeout_s", 300.0))
        self.prompt_template = cfg.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
        self.max_concurrency = max(1, int(cfg.get("max_concurrency", 1)))

    def __call__(self, *, artifact: Any, request: Any) -> dict[str, float]:
        del request
        prompt = artifact.prompt
        prompt_text = _render_prompt_template(self.prompt_template, prompt=prompt)
        with tempfile.TemporaryDirectory(prefix="vrl-claude-image-qa-") as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "image.png"
            output_path = tmp_path / "judge_output.txt"
            write_png(artifact.as_media(), image_path)
            stdin_bytes = _build_stream_json_message(image_path, prompt_text)
            command = _render_command(self.command, output_path=output_path, prompt=prompt)
            output_text = self._run_command(
                command,
                stdin_bytes=stdin_bytes,
                output_path=output_path,
                workdir=tmp_path,
            )
        return {"claude_image_qa": _extract_score_from_text(output_text)}

    def _run_command(
        self,
        command: list[str],
        *,
        stdin_bytes: bytes,
        output_path: Path,
        workdir: Path,
    ) -> str:
        try:
            completed = subprocess.run(
                command,
                input=stdin_bytes,
                capture_output=True,
                cwd=str(workdir),
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Claude image-QA timed out after {self.timeout_s:.1f}s: {command!r}",
            ) from exc

        stdout_text = completed.stdout.decode("utf-8", errors="replace")
        stderr_text = completed.stderr.decode("utf-8", errors="replace")
        if completed.returncode != 0:
            if _is_auth_error(stderr_text, stdout_text):
                raise PermissionError(
                    "Claude CLI is not authenticated.\n"
                    "  • OAuth login : run `claude` and complete the login flow\n"
                    "  • API key auth: set the ANTHROPIC_API_KEY environment variable\n"
                    f"Raw error: {(stderr_text or stdout_text).strip()!r}",
                )
            raise RuntimeError(
                "Claude image-QA failed "
                f"(exit={completed.returncode}): {command!r}\n"
                f"STDERR:\n{stderr_text}\nSTDOUT:\n{stdout_text}",
            )

        if output_path.exists():
            file_text = output_path.read_text(encoding="utf-8", errors="replace").strip()
            if file_text:
                return file_text
        return stdout_text


def claude_image_qa_reward_model(worker_config: Mapping[str, Any]) -> ClaudeImageQARewardModel:
    return ClaudeImageQARewardModel(worker_config)


def _normalize_command(command: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(command, str):
        return shlex.split(command)
    return [str(part) for part in command]


def _render_command(command: list[str], *, output_path: Path, prompt: str) -> list[str]:
    values = {"output_path": str(output_path), "prompt": prompt}
    return [part.format(**values) for part in command]


def _build_stream_json_message(image_path: Path, prompt_text: str) -> bytes:
    """Pack base64-encoded image + prompt text into a stream-json user message."""
    image_data = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
    message = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data,
                    },
                },
                {"type": "text", "text": prompt_text},
            ],
        },
    }
    return (json.dumps(message) + "\n").encode("utf-8")


def _render_prompt_template(template: str, *, prompt: str) -> str:
    placeholder = "__VRL_CLAUDE_IMAGE_QA_PROMPT_PLACEHOLDER__"
    while placeholder in template:
        placeholder = f"_{placeholder}"
    rendered = template.replace("{prompt}", placeholder)
    rendered = rendered.replace("{{", "{").replace("}}", "}")
    return rendered.replace(placeholder, prompt)


def _extract_score_from_text(text: str) -> float:
    """Parse judge response into a clamped [0, 1] score."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("Claude image-QA returned empty output")

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
            f"Cannot parse Claude image-QA score from output: {text!r}",
        ) from exc


def _find_first_json_value(text: str) -> Any | None:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
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
            raise ValueError("Claude image-QA returned an empty score list")
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
    raise ValueError(f"Cannot convert Claude image-QA answer to reward score: {answer!r}")


def _clamp_score(value: float) -> float:
    return min(max(value, 0.0), 1.0)


# Substrings (lowercased) that indicate the CLI exited due to missing auth.
_AUTH_PATTERNS = (
    "invalid api key",
    "authentication",
    "not authenticated",
    "log in",
    "login",
    "unauthorized",
    "401",
)


def _is_auth_error(stderr: str, stdout: str) -> bool:
    combined = (stderr + stdout).lower()
    return any(pat in combined for pat in _AUTH_PATTERNS)


__all__ = [
    "DEFAULT_COMMAND",
    "DEFAULT_PROMPT_TEMPLATE",
    "ClaudeImageQARewardModel",
    "_extract_score_from_text",
    "_render_prompt_template",
    "claude_image_qa_reward_model",
]
