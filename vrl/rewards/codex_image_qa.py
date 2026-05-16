"""Codex-backed image QA reward for prompt-image alignment."""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from vrl.rewards.base import RewardFunction
from vrl.rewards.types import RewardRollout

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
        if command is None:
            raise ValueError("CodexImageQAReward requires reward.kwargs.codex_image_qa.command")
        self.command = _normalize_command(command)
        self.timeout_s = float(timeout_s)
        self.prompt_template = prompt_template
        self.pass_prompt_stdin = bool(pass_prompt_stdin)
        self.max_concurrency = max(1, int(max_concurrency))

    async def score(self, rollout: RewardRollout) -> float:
        scores = await self.score_batch([rollout])
        return float(scores[0])

    async def score_batch(self, rollouts: list[RewardRollout]) -> list[float]:
        if not rollouts:
            return []
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _score_one(rollout: RewardRollout) -> float:
            async with semaphore:
                return await self._score_one(rollout)

        return list(await asyncio.gather(*(_score_one(rollout) for rollout in rollouts)))

    async def _score_one(self, rollout: RewardRollout) -> float:
        prompt = rollout.trajectory.prompt
        prompt_text = _render_prompt_template(self.prompt_template, prompt=prompt)
        with tempfile.TemporaryDirectory(prefix="vrl-codex-image-qa-") as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "image.png"
            output_path = tmp_path / "judge_output.txt"
            _write_image_png(rollout.trajectory.output, image_path)
            command = _render_command(
                self.command,
                image_path=image_path,
                output_path=output_path,
                prompt=prompt,
            )
            output_text = await self._run_command(
                command,
                stdin_text=prompt_text if self.pass_prompt_stdin else "",
                output_path=output_path,
                workdir=tmp_path,
            )
        return _extract_score_from_text(output_text)

    async def _run_command(
        self,
        command: list[str],
        *,
        stdin_text: str,
        output_path: Path,
        workdir: Path,
    ) -> str:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workdir),
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(stdin_text.encode("utf-8")),
                timeout=self.timeout_s,
            )
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise TimeoutError(
                f"Codex image-QA timed out after {self.timeout_s:.1f}s: {command!r}",
            ) from exc

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        if process.returncode != 0:
            raise RuntimeError(
                "Codex image-QA failed "
                f"(exit={process.returncode}): {command!r}\nSTDERR:\n{stderr_text}\nSTDOUT:\n{stdout_text}",
            )

        if output_path.exists():
            file_text = output_path.read_text(encoding="utf-8", errors="replace").strip()
            if file_text:
                return file_text
        return stdout_text


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


def _write_image_png(image: Any, path: Path) -> None:
    from PIL import Image

    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        pil_image = Image.fromarray(_image_to_uint8_hwc(image), mode="RGB")
    pil_image.save(path, format="PNG")


def _image_to_uint8_hwc(image: Any) -> np.ndarray:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a project dependency.
        torch = None  # type: ignore[assignment]

    if torch is not None and isinstance(image, torch.Tensor):
        array = image.detach().float().cpu()
        if array.ndim == 4:
            if array.shape[0] != 1:
                raise ValueError(
                    "CodexImageQAReward expects a single image per rollout, got a 4D batch",
                )
            array = array[0]
        if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
            array = array[:3].permute(1, 2, 0)
        array_np = array.numpy()
    else:
        array_np = np.asarray(image)

    if array_np.ndim != 3:
        raise ValueError(f"CodexImageQAReward expected image with 3 dims, got shape {array_np.shape}")
    if array_np.shape[-1] == 1:
        array_np = np.repeat(array_np, 3, axis=-1)
    if array_np.shape[-1] == 4:
        array_np = array_np[..., :3]
    if array_np.shape[-1] != 3:
        raise ValueError(f"CodexImageQAReward expected RGB image, got shape {array_np.shape}")

    if np.issubdtype(array_np.dtype, np.floating):
        if array_np.min() < 0.0:
            array_np = (array_np + 1.0) * 0.5
        array_np = np.clip(array_np, 0.0, 1.0)
        return (array_np * 255).round().astype(np.uint8)
    return np.clip(array_np, 0, 255).astype(np.uint8)


__all__ = [
    "DEFAULT_PROMPT_TEMPLATE",
    "CodexImageQAReward",
    "_extract_score_from_text",
    "_render_prompt_template",
    "_write_image_png",
]
