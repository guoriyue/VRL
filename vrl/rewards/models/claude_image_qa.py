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
    "claude -p --input-format stream-json --output-format stream-json --verbose"
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
        # When the judge refuses or returns unparseable output, return this
        # score and log the case to refusal_log_dir for review. Default 0.0
        # gives GRPO a strong negative advantage on refused images so the
        # policy learns to avoid them. Setting this near the mean (e.g. 0.5)
        # is a deliberate choice to silence the signal — only set it that
        # way if you've manually audited the refusal log and confirmed the
        # refusals are false positives.
        self.refusal_score = float(cfg.get("refusal_score", 0.0))
        log_dir = cfg.get("refusal_log_dir")
        self.refusal_log_dir = Path(log_dir).expanduser() if log_dir else None
        # Prompt template precedence: explicit string > file path > default.
        # The file form lets a markdown rubric live separately on disk.
        prompt_template = cfg.get("prompt_template")
        prompt_template_file = cfg.get("prompt_template_file")
        if prompt_template_file:
            path = Path(str(prompt_template_file)).expanduser()
            if not path.is_file():
                raise FileNotFoundError(
                    f"claude_image_qa: prompt_template_file not found: {path!s}",
                )
            file_text = path.read_text(encoding="utf-8")
            if not file_text.strip():
                raise ValueError(
                    f"claude_image_qa: prompt_template_file is empty: {path!s}",
                )
            self.prompt_template = file_text
        elif prompt_template is not None:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = DEFAULT_PROMPT_TEMPLATE
        self.max_concurrency = max(1, int(cfg.get("max_concurrency", 1)))

    def score_request(self, request: Any) -> list[dict[str, float]]:
        """Batched scoring with bounded concurrency.

        Runs up to ``max_concurrency`` Claude CLI subprocesses in parallel
        via a thread pool. Each subprocess blocks on an external IO wait
        (subprocess.run releases the GIL), so threading yields real wall-
        clock parallelism. The runtime auto-detects this method and uses
        it in place of the per-artifact ``__call__`` loop.

        Per-artifact failures (judge refusal, parse error, timeout) are
        caught: the artifact is logged to ``refusal_log_dir`` and the
        score becomes ``refusal_score`` so the training step continues
        instead of crashing on one bad sample.
        """

        from concurrent.futures import ThreadPoolExecutor

        artifacts = list(request.artifacts)
        if not artifacts:
            return []
        workers = max(1, min(self.max_concurrency, len(artifacts)))

        def safe_call(artifact: Any) -> dict[str, float]:
            try:
                return self.__call__(artifact=artifact, request=request)
            except Exception as exc:
                self._log_refusal(artifact, exc)
                return {"claude_image_qa": float(self.refusal_score)}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(safe_call, artifacts))

    def _log_refusal(self, artifact: Any, exc: BaseException) -> None:
        """Persist a refused/failed scoring case so it can be reviewed.

        Writes a JSONL line with artifact id + prompt + raw exception,
        plus saves the artifact's image alongside for visual audit. No-op
        when ``refusal_log_dir`` is not configured.
        """

        if self.refusal_log_dir is None:
            return
        import json
        import time
        import traceback
        try:
            self.refusal_log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        artifact_id = str(getattr(artifact, "artifact_id", "unknown"))
        ts = time.strftime("%Y%m%dT%H%M%S")
        stem = f"{ts}_{artifact_id}"
        image_relpath: str | None = None
        try:
            media = artifact.as_media()
            if media is not None:
                image_path = self.refusal_log_dir / f"{stem}.png"
                write_png(media, image_path)
                image_relpath = str(image_path.name)
        except Exception:
            image_relpath = None
        entry = {
            "ts": ts,
            "artifact_id": artifact_id,
            "prompt": str(getattr(artifact, "prompt", "") or ""),
            "image_file": image_relpath,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:2000],
            "traceback_tail": "".join(traceback.format_exception_only(type(exc), exc))[
                :2000
            ],
        }
        path = self.refusal_log_dir / "refusals.jsonl"
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

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
    """Parse judge response into a clamped [0, 1] score.

    Accepts three output shapes:
      * Plain text: the assistant's response IS the JSON object.
      * stream-json JSONL (claude CLI 2.1+ with ``--output-format
        stream-json --verbose``): we pull the ``result`` field from the
        success ``type:"result"`` event, then parse it as JSON.
      * Free-form: regex fallback for ``"score": <num>``.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("Claude image-QA returned empty output")

    assistant_text = _extract_assistant_text(stripped)
    if assistant_text is not None and assistant_text != stripped:
        stripped = assistant_text.strip()

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


def _extract_assistant_text(stream_output: str) -> str | None:
    """Pull the assistant's final text from a stream-json JSONL transcript.

    Looks for the success ``type:"result"`` event and returns its
    ``result`` field. Falls back to concatenating ``type:"assistant"``
    text blocks when no result event is present. Returns ``None`` when
    the input does not look like stream-json at all (no parseable
    event objects), so the caller can keep treating it as plain text.
    """

    final_result: str | None = None
    assistant_chunks: list[str] = []
    parsed_any = False
    for line in stream_output.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        parsed_any = True
        etype = event.get("type")
        if etype == "result" and event.get("subtype") == "success":
            result_text = event.get("result")
            if isinstance(result_text, str) and result_text.strip():
                final_result = result_text
        elif etype == "assistant":
            content = event.get("message", {}).get("content", [])
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    txt = block.get("text", "")
                    if txt:
                        assistant_chunks.append(txt)
    if not parsed_any:
        return None
    if final_result is not None:
        return final_result
    if assistant_chunks:
        return "".join(assistant_chunks)
    return None


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
        if "axes" in value and isinstance(value["axes"], dict):
            # Multi-axis rubric output (e.g. anatomy hands+body). Compute the
            # aggregate ourselves so the judge does not need to emit `score`.
            return _aggregate_axes_score(value["axes"])
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


def _aggregate_axes_score(axes: Mapping[str, Any]) -> float:
    """Aggregate per-axis integer scores into ``[0, 1]``.

    Mirrors the formula documented in the anatomy rubric:
        score = ((geom_mean(visible) + min(visible)) / 2) / 10
    Null axes are skipped; any axis at 0 collapses the score to 0.
    """

    import math

    visible: list[float] = []
    for value in axes.values():
        if value is None:
            continue
        try:
            visible.append(float(value))
        except (TypeError, ValueError):
            continue
    if not visible:
        return 0.0
    if any(v <= 0.0 for v in visible):
        return 0.0
    geom = math.exp(sum(math.log(v) for v in visible) / len(visible))
    worst = min(visible)
    return _clamp_score(((geom + worst) / 2.0) / 10.0)


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
