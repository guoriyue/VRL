"""Codex-CLI image-QA reward as a batch-capable RewardModel.

A rubric-driven LLM judge: write each rollout image to a temp PNG, render the
command/prompt placeholders, run the ``codex exec`` subprocess (timeout + error
handling), then parse the judge output into a clamped ``[0, 1]`` score. Unlike
the fixed aesthetic scorers, the rubric in ``prompt_template`` decides exactly
what is judged — so it can target axes the learned models miss (e.g. crisp
cel-shading, clean line art), not just generic prettiness.

Restored 2026-08-22 (removed in 51c78968) and adapted to the current
score_batch/InProcessRewardScorer interface. Each score is an independent CLI
call, so ``score_batch`` fans them out over a thread pool bounded by
``max_concurrency``.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
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

DEFAULT_GRID_PROMPT_TEMPLATE = """You are a strict anime image judge.
The attached image is a montage of {count} separate generations arranged in a
grid, each cell labeled with a number (1..{count}) in its top-left corner,
ordered left-to-right, top-to-bottom. Every cell was generated from the SAME
text prompt below.

Score EACH cell independently in [0, 1]. Use dense continuous decimals based on
visible evidence; do not collapse cells to the same value when they differ.

Return exactly one JSON object and no extra text, with {count} scores in cell
order:
{{"scores": [0.37, 0.81, ...]}}

Scoring rubric:
- 0.85-1.00: clear, high-quality anime that matches the prompt.
- 0.60-0.84: mostly good with minor issues.
- 0.35-0.59: partial match or noticeable quality problems.
- 0.10-0.34: coherent but weak.
- 0.00-0.09: blank, broken, or unrelated.

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
        # Grid batching: pack up to N same-prompt images into one downscaled,
        # cell-numbered montage and score them in a SINGLE CLI call, cutting
        # calls and image tokens ~N x. 1 keeps the one-image-per-call behavior.
        self.images_per_call = max(1, int(cfg.get("images_per_call", 1)))
        self.tile_size = max(64, int(cfg.get("tile_size", 256)))
        self.grid_prompt_template = cfg.get("grid_prompt_template", DEFAULT_GRID_PROMPT_TEMPLATE)

    def score_batch(self, artifacts: Sequence[Any]) -> list[dict[str, float]]:
        artifacts = list(artifacts)
        if not artifacts:
            return []
        if self.images_per_call > 1:
            return self._score_batch_grid(artifacts)
        workers = min(self.max_concurrency, len(artifacts))
        if workers <= 1:
            return [self(artifact) for artifact in artifacts]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self.__call__, artifacts))

    def _score_batch_grid(self, artifacts: list[Any]) -> list[dict[str, float]]:
        """Group by prompt, tile each group into montages, one CLI call per tile.

        Grouping by identical prompt keeps every cell in a montage sharing one
        text prompt, so the rubric names the prompt once and asks for a score per
        numbered cell. Scores are mapped back to each artifact's original index.
        """

        groups: dict[str, list[int]] = {}
        for idx, artifact in enumerate(artifacts):
            groups.setdefault(getattr(artifact, "prompt", ""), []).append(idx)

        chunks: list[tuple[str, list[int]]] = []
        for prompt, indices in groups.items():
            for start in range(0, len(indices), self.images_per_call):
                chunks.append((prompt, indices[start : start + self.images_per_call]))

        scores: list[float | None] = [None] * len(artifacts)

        def run_chunk(chunk: tuple[str, list[int]]) -> None:
            prompt, indices = chunk
            cell_scores = self._score_grid([artifacts[i] for i in indices], prompt)
            for i, score in zip(indices, cell_scores, strict=True):
                scores[i] = score

        workers = min(self.max_concurrency, len(chunks))
        if workers <= 1:
            for chunk in chunks:
                run_chunk(chunk)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(run_chunk, chunks))

        return [{"codex_image_qa": (0.0 if s is None else s)} for s in scores]

    def _score_grid(self, group: list[Any], prompt: str) -> list[float]:
        """Compose one numbered montage of ``group`` and parse a score per cell."""

        n = len(group)
        if n == 1:
            return [self(group[0])["codex_image_qa"]]
        prompt_text = _render_grid_prompt(self.grid_prompt_template, prompt=prompt, count=n)
        with tempfile.TemporaryDirectory(prefix="vrl-codex-image-qa-grid-") as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "grid.png"
            output_path = tmp_path / "judge_output.txt"
            _compose_grid([a.as_media() for a in group], self.tile_size, image_path)
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
        return _extract_grid_scores(output_text, n)

    def __call__(self, artifact: Any) -> dict[str, float]:
        prompt = getattr(artifact, "prompt", "")
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
                f"(exit={completed.returncode}): {command!r}\n"
                f"STDERR:\n{stderr_text}\nSTDOUT:\n{stdout_text}",
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


def _render_grid_prompt(template: str, *, prompt: str, count: int) -> str:
    """Render the grid rubric: fill {count}, then the prompt (braces preserved)."""

    rendered = _render_prompt_template(template, prompt=prompt)
    return rendered.replace("{count}", str(count))


def _compose_grid(medias: list[Any], tile: int, out_path: Path) -> None:
    """Downscale each media to a ``tile`` square and tile into a numbered montage."""

    import math

    from PIL import Image, ImageDraw

    from vrl.utils.media import to_pil_image

    imgs = [to_pil_image(m).convert("RGB").resize((tile, tile), Image.LANCZOS) for m in medias]
    n = len(imgs)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    pad = 2
    canvas = Image.new(
        "RGB", (cols * tile + (cols + 1) * pad, rows * tile + (rows + 1) * pad), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for i, img in enumerate(imgs):
        r, c = divmod(i, cols)
        x = pad + c * (tile + pad)
        y = pad + r * (tile + pad)
        canvas.paste(img, (x, y))
        label = str(i + 1)
        # High-contrast cell number in the top-left corner.
        draw.rectangle([x, y, x + 9 * len(label) + 6, y + 18], fill="black")
        draw.text((x + 3, y + 3), label, fill="white")
    canvas.save(out_path, format="PNG")


def _extract_grid_scores(text: str, count: int) -> list[float]:
    """Parse ``count`` per-cell scores from a grid judge response."""

    value = _find_first_json_value(text.strip())
    scores: list[Any] | None = None
    if isinstance(value, dict):
        for key in ("scores", "cells", "results"):
            if isinstance(value.get(key), list):
                scores = value[key]
                break
    elif isinstance(value, list):
        scores = value
    if scores is None:
        raise ValueError(f"Cannot parse {count} grid scores from output: {text!r}")
    out = [_score_from_value(s) for s in scores[:count]]
    # Missing cells default to 0.0 so one short response never crashes an update.
    out.extend([0.0] * (count - len(out)))
    return out


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
]
