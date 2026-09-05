"""Codex-CLI image-QA reward as a batch-capable RewardModel.

A rubric-driven LLM judge: write rollout images to temporary PNGs, render the
command/prompt placeholders, run the ``codex exec`` subprocess (timeout + error
handling), then parse the judge output. Absolute mode returns clamped ``[0, 1]``
scores. Scored rollout images and their judge outputs can be retained for
visual audit across training restarts.

Restored 2026-08-22 (removed in 51c78968) and adapted to the current
score_batch/InProcessRewardScorer interface. Judge calls fan out over a thread
pool bounded by ``max_concurrency``.
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
    """RewardModel returning one ``codex_image_qa`` score per rollout artifact.

    The command may contain ``{image_path}``, ``{output_path}``, ``{prompt}``,
    and ``{output_schema_path}`` placeholders. The rendered prompt is sent to
    stdin by default. Native ``codex exec`` commands automatically receive the
    generated ``--output-schema`` argument; compatible commands can opt in with
    the schema-path placeholder. Optional metadata targets and scored-rollout
    persistence preserve the exact target, generated pixels, and judge scores.
    """

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        cfg = dict(worker_config)
        command = cfg.get("command")
        if command is None:
            raise ValueError("CodexImageQAReward requires reward.kwargs.codex_image_qa.command")
        self.command = _normalize_command(command)
        self.timeout_s = float(cfg.get("timeout_s", 300.0))
        self.prompt_template = cfg.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
        raw_prompt_metadata_key = cfg.get("prompt_metadata_key", "")
        if not isinstance(raw_prompt_metadata_key, str):
            raise TypeError("Codex image-QA prompt_metadata_key must be a string")
        self.prompt_metadata_key = raw_prompt_metadata_key.strip()
        self.pass_prompt_stdin = bool(cfg.get("pass_prompt_stdin", True))
        self.max_concurrency = max(1, int(cfg.get("max_concurrency", 1)))
        # Grid batching: pack up to N same-prompt images into one downscaled,
        # cell-numbered montage and score them in a SINGLE CLI call, cutting
        # calls and image tokens ~N x. 1 keeps the one-image-per-call behavior.
        self.images_per_call = max(1, int(cfg.get("images_per_call", 1)))
        self.tile_size = max(64, int(cfg.get("tile_size", 256)))
        self.grid_prompt_template = cfg.get("grid_prompt_template", DEFAULT_GRID_PROMPT_TEMPLATE)
        scored_rollout_dir = str(cfg.get("scored_rollout_dir", "")).strip()
        self.scored_rollout_dir = Path(scored_rollout_dir) if scored_rollout_dir else None
        self._saved_batch_index = _next_saved_batch_index(self.scored_rollout_dir)

    def score_batch(self, artifacts: Sequence[Any]) -> list[dict[str, float]]:
        artifacts = list(artifacts)
        if not artifacts:
            return []
        if self.images_per_call > 1:
            scores = self._score_batch_grid(artifacts)
        else:
            workers = min(self.max_concurrency, len(artifacts))
            if workers <= 1:
                scores = [self(artifact) for artifact in artifacts]
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    scores = list(pool.map(self.__call__, artifacts))
        self._save_scored_rollouts(artifacts, scores)
        return scores

    def _save_scored_rollouts(
        self,
        artifacts: list[Any],
        scores: list[dict[str, float]],
    ) -> None:
        """Persist underlying rollout pixels and scores for visual audit."""

        root = self.scored_rollout_dir
        if root is None:
            return
        if len(artifacts) != len(scores):
            raise ValueError(
                "cannot save scored rollouts with mismatched artifacts and scores: "
                f"{len(artifacts)} != {len(scores)}",
            )

        root.mkdir(parents=True, exist_ok=True)
        batch_index = self._saved_batch_index
        final_dir = root / f"batch-{batch_index:06d}"
        if final_dir.exists():
            raise FileExistsError(f"scored rollout batch already exists: {final_dir}")

        with tempfile.TemporaryDirectory(prefix=".batch-", dir=root) as tmp:
            staging_dir = Path(tmp)
            medias = [artifact.as_media() for artifact in artifacts]
            policy_versions = {
                int(artifact.metadata["rollout_policy_version"])
                for artifact in artifacts
                if "rollout_policy_version" in artifact.metadata
            }
            if len(policy_versions) > 1:
                raise ValueError(
                    "one scored rollout batch cannot mix policy versions: "
                    f"{sorted(policy_versions)}",
                )
            items: list[dict[str, Any]] = []
            for sample_index, (artifact, media, score_map) in enumerate(
                zip(artifacts, medias, scores, strict=True),
            ):
                image_name = f"sample-{sample_index:02d}.png"
                write_png(media, staging_dir / image_name)
                items.append(
                    {
                        "artifact_id": str(artifact.artifact_id),
                        "image": image_name,
                        "prompt": str(getattr(artifact, "prompt", "")),
                        "judge_prompt": self._prompt_for_artifact(artifact),
                        "scores": {str(name): float(value) for name, value in score_map.items()},
                    },
                )

            montage_name = "montage.png"
            _compose_grid(medias, self.tile_size, staging_dir / montage_name)
            manifest = {
                "schema_version": 1,
                "batch_index": batch_index,
                "count": len(items),
                "montage": montage_name,
                "rollout_policy_version": next(iter(policy_versions), None),
                "items": items,
            }
            (staging_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            staging_dir.replace(final_dir)
        self._saved_batch_index += 1

    def _score_batch_grid(self, artifacts: list[Any]) -> list[dict[str, float]]:
        """Group by prompt, tile each group into montages, one CLI call per tile.

        Grouping by identical prompt keeps every cell in a montage sharing one
        text prompt, so the rubric names the prompt once and asks for a score per
        numbered cell. Scores are mapped back to each artifact's original index.
        """

        groups: dict[str, list[int]] = {}
        for idx, artifact in enumerate(artifacts):
            groups.setdefault(self._prompt_for_artifact(artifact), []).append(idx)

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
        prompt_text = _render_prompt_template(
            self.grid_prompt_template,
            prompt=prompt,
            count=n,
        )
        with tempfile.TemporaryDirectory(prefix="vrl-codex-image-qa-grid-") as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "grid.png"
            output_path = tmp_path / "judge_output.txt"
            output_schema_path = tmp_path / "output_schema.json"
            _compose_grid([a.as_media() for a in group], self.tile_size, image_path)
            _write_output_schema(output_schema_path, count=n)
            command = _render_command(
                self.command,
                image_path=image_path,
                output_path=output_path,
                output_schema_path=output_schema_path,
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
        prompt = self._prompt_for_artifact(artifact)
        prompt_text = _render_prompt_template(self.prompt_template, prompt=prompt)
        with tempfile.TemporaryDirectory(prefix="vrl-codex-image-qa-") as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "image.png"
            output_path = tmp_path / "judge_output.txt"
            output_schema_path = tmp_path / "output_schema.json"
            write_png(artifact.as_media(), image_path)
            _write_output_schema(output_schema_path)
            command = _render_command(
                self.command,
                image_path=image_path,
                output_path=output_path,
                output_schema_path=output_schema_path,
                prompt=prompt,
            )
            output_text = self._run_command(
                command,
                stdin_text=prompt_text if self.pass_prompt_stdin else "",
                output_path=output_path,
                workdir=tmp_path,
            )
        return {"codex_image_qa": _extract_score_from_text(output_text)}

    def _prompt_for_artifact(self, artifact: Any) -> str:
        """Resolve the rubric target without reparsing generation prose."""

        if not self.prompt_metadata_key:
            return str(getattr(artifact, "prompt", ""))
        metadata = getattr(artifact, "metadata", None)
        if not isinstance(metadata, Mapping):
            raise TypeError(
                f"Codex image-QA artifact {artifact.artifact_id!r} metadata must be a mapping",
            )
        if self.prompt_metadata_key not in metadata:
            raise ValueError(
                f"Codex image-QA artifact {artifact.artifact_id!r} is missing "
                f"metadata[{self.prompt_metadata_key!r}]",
            )
        value = metadata[self.prompt_metadata_key]
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise TypeError(
                f"Codex image-QA artifact {artifact.artifact_id!r} "
                f"metadata[{self.prompt_metadata_key!r}] must be a string or number",
            )
        prompt = str(value).strip()
        if not prompt:
            raise ValueError(
                f"Codex image-QA artifact {artifact.artifact_id!r} "
                f"metadata[{self.prompt_metadata_key!r}] must be non-empty",
            )
        return prompt

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
    output_schema_path: Path,
    prompt: str,
) -> list[str]:
    values = {
        "image_path": str(image_path),
        "output_path": str(output_path),
        "output_schema_path": str(output_schema_path),
        "prompt": prompt,
    }
    rendered = [part.format(**values) for part in command]
    is_codex_exec = (
        len(rendered) >= 2 and Path(rendered[0]).stem == "codex" and rendered[1] in {"exec", "e"}
    )
    has_output_schema = any(
        part == "--output-schema" or part.startswith("--output-schema=") for part in rendered
    )
    if is_codex_exec and not has_output_schema:
        # Insert next to the subcommand instead of appending after the positional
        # prompt, whose placement is intentionally owned by the configured command.
        rendered[2:2] = ["--output-schema", str(output_schema_path)]
    return rendered


def _write_output_schema(path: Path, *, count: int | None = None) -> None:
    """Write the per-call response contract beside the disposable image.

    Montage chunks can have different sizes (especially the final chunk), so
    their exact cardinality belongs to the invocation rather than static config.
    """

    score = {"type": "number", "minimum": 0.0, "maximum": 1.0}
    if count is None:
        properties: dict[str, Any] = {"score": score}
        required = ["score"]
    else:
        if count < 1:
            raise ValueError(f"Codex image-QA output schema count must be positive, got {count}")
        properties = {
            "scores": {
                "type": "array",
                "items": score,
                "minItems": count,
                "maxItems": count,
            },
        }
        required = ["scores"]
    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    path.write_text(json.dumps(schema, separators=(",", ":")), encoding="utf-8")


def _next_saved_batch_index(root: Path | None) -> int:
    """Continue after complete batches when a supervised run resumes."""

    if root is None or not root.exists():
        return 0
    indices = []
    for candidate in root.glob("batch-*"):
        if not (candidate / "manifest.json").is_file():
            continue
        try:
            indices.append(int(candidate.name.removeprefix("batch-")))
        except ValueError:
            continue
    return max(indices, default=-1) + 1


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
    if len(scores) != count:
        raise ValueError(
            f"Expected {count} grid scores, got {len(scores)} in output: {text!r}",
        )
    return [_score_from_value(score) for score in scores]


def _render_prompt_template(
    template: str,
    *,
    prompt: str,
    count: int | None = None,
    response_contract: str | None = None,
) -> str:
    """Render invocation-owned fields while preserving literal JSON braces."""

    if response_contract is None:
        response_contract = '{"score": 0.37}' if count is None else '{"scores": [0.37, 0.81, ...]}'
    values = {
        "count": str(1 if count is None else count),
        "prompt": prompt,
        "response_contract": response_contract,
    }

    rendered = template
    placeholders: dict[str, str] = {}
    for name, value in values.items():
        placeholder = f"__VRL_IMAGE_QA_{name.upper()}_PLACEHOLDER__"
        while placeholder in template:
            placeholder = f"_{placeholder}"
        rendered = rendered.replace(f"{{{name}}}", placeholder)
        placeholders[placeholder] = value
    rendered = rendered.replace("{{", "{").replace("}}", "}")
    for placeholder, value in placeholders.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


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
