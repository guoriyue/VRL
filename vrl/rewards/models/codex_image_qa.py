"""Codex-backed image QA reward as a synchronous RewardModel.

Ports the exact CLI-judge logic from the in-process ``CodexImageQAReward``:
write the rollout image to a temp PNG, render the command/prompt placeholders,
run the subprocess (with timeout + error handling), then parse the judge output
into a clamped ``[0, 1]`` score. The CLI needs a real file path, so ``__call__``
materializes the in-memory image to a temporary file and cleans it up.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vrl.utils.media import write_png

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


class CodexImageQARewardModel:
    """RewardModel returning ``{"codex_image_qa": score}`` per artifact.

    The command may contain ``{image_path}``, ``{output_path}``, and ``{prompt}``
    placeholders. The rendered prompt is sent to stdin by default, which matches
    ``codex exec --image {image_path} --output-last-message {output_path} -``.
    """

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        cfg = dict(worker_config)
        command = cfg.get("command")
        if command is None:
            raise ValueError("CodexImageQAReward requires reward.kwargs.codex_image_qa.command")
        self.command = _normalize_command(command)
        self.timeout_s = float(cfg.get("timeout_s", 300.0))
        self.prompt_template = cfg.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
        self.pass_prompt_stdin = bool(cfg.get("pass_prompt_stdin", True))
        self.max_concurrency = max(1, int(cfg.get("max_concurrency", 1)))

    def __call__(self, *, artifact: Any, request: Any) -> dict[str, float]:
        del request
        prompt = artifact.prompt
        prompt_text = _render_prompt_template(self.prompt_template, prompt=prompt)
        with tempfile.TemporaryDirectory(prefix="vrl-codex-image-qa-") as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "image.png"
            output_path = tmp_path / "judge_output.txt"
            write_png(artifact.as_media(), image_path)
            command = _render_command(
                self.command,
                image_path=image_path,
                output_path=output_path,
                prompt=prompt,
            )
            output_text = self._run_command(
                command,
                stdin_text=prompt_text if self.pass_prompt_stdin else "",
                output_path=output_path,
                workdir=tmp_path,
            )
        return {"codex_image_qa": _extract_score_from_text(output_text)}

    def _run_command(
        self,
        command: list[str],
        *,
        stdin_text: str,
        output_path: Path,
        workdir: Path,
    ) -> str:
        try:
            completed = subprocess.run(
                command,
                input=stdin_text.encode("utf-8"),
                capture_output=True,
                cwd=str(workdir),
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Codex image-QA timed out after {self.timeout_s:.1f}s: {command!r}",
            ) from exc

        stdout_text = completed.stdout.decode("utf-8", errors="replace")
        stderr_text = completed.stderr.decode("utf-8", errors="replace")
        if completed.returncode != 0:
            raise RuntimeError(
                "Codex image-QA failed "
                f"(exit={completed.returncode}): {command!r}\nSTDERR:\n{stderr_text}\nSTDOUT:\n{stdout_text}",
            )

        if output_path.exists():
            file_text = output_path.read_text(encoding="utf-8", errors="replace").strip()
            if file_text:
                return file_text
        return stdout_text


def codex_image_qa_reward_model(worker_config: Mapping[str, Any]) -> CodexImageQARewardModel:
    return CodexImageQARewardModel(worker_config)


def _normalize_command(command: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(command, str):
        return shlex.split(command)
    return [str(part) for part in command]


def _render_command(
    command: list[str],
    *,
    image_path: Path,
    output_path: Path,
    prompt: str,
) -> list[str]:
    values = {
        "image_path": str(image_path),
        "output_path": str(output_path),
        "prompt": prompt,
    }
    return [part.format(**values) for part in command]


def _render_prompt_template(template: str, *, prompt: str) -> str:
    """Render only the prompt placeholder so JSON examples can use literal braces."""

    placeholder = "__VRL_IMAGE_QA_PROMPT_PLACEHOLDER__"
    while placeholder in template:
        placeholder = f"_{placeholder}"
    rendered = template.replace("{prompt}", placeholder)
    rendered = rendered.replace("{{", "{").replace("}}", "}")
    return rendered.replace(placeholder, prompt)


def _extract_score_from_text(text: str) -> float:
    """Parse a judge response into a clamped score in ``[0, 1]``."""

    stripped = text.strip()
    if not stripped:
        raise ValueError("Codex image-QA returned empty output")

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
        raise ValueError(f"Cannot parse Codex image-QA score from output: {text!r}") from exc


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
            raise ValueError("Codex image-QA returned an empty score list")
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
    raise ValueError(f"Cannot convert Codex image-QA answer to reward score: {answer!r}")


def _clamp_score(value: float) -> float:
    return min(max(value, 0.0), 1.0)


__all__ = [
    "DEFAULT_PROMPT_TEMPLATE",
    "CodexImageQARewardModel",
    "_extract_score_from_text",
    "_render_prompt_template",
    "codex_image_qa_reward_model",
]
