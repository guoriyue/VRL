"""Tests for Codex image-QA reward helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vrl.algorithms import Rollout, Trajectory
from vrl.rewards.codex_image_qa import (
    CodexImageQAReward,
    _extract_score_from_text,
    _render_prompt_template,
    _write_image_png,
)


class _FakeCodexImageQAReward(CodexImageQAReward):
    def __init__(self, response: str):
        super().__init__(
            device="cpu",
            command=["judge", "--image", "{image_path}", "--out", "{output_path}", "-"],
        )
        self.response = response
        self.last_command: list[str] | None = None
        self.last_stdin = ""

    async def _run_command(
        self,
        command: list[str],
        *,
        stdin_text: str,
        output_path: Path,
        workdir: Path,
    ) -> str:
        del output_path, workdir
        self.last_command = command
        self.last_stdin = stdin_text
        return self.response


def _rollout(prompt: str) -> Rollout:
    image = torch.zeros(3, 8, 8)
    return Rollout(
        request=None,
        trajectory=Trajectory(prompt=prompt, seed=0, steps=[], output=image),
        metadata={"source": "unit"},
    )


def test_extract_score_accepts_json_score() -> None:
    assert _extract_score_from_text('{"score": 0.25, "answer": "yes"}') == pytest.approx(0.25)


def test_extract_score_accepts_yes_no_answers() -> None:
    assert _extract_score_from_text("yes, it matches") == 1.0
    assert _extract_score_from_text("No") == 0.0


def test_extract_score_accepts_janus_result_map_shape() -> None:
    text = '{"success": true, "resultMap": {"scores": [0.7]}}'
    assert _extract_score_from_text(text) == pytest.approx(0.7)


def test_tensor_image_writes_png(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    _write_image_png(torch.ones(3, 4, 4), path)
    assert path.read_bytes().startswith(b"\x89PNG")


def test_prompt_template_allows_literal_json_braces() -> None:
    template = 'Return JSON only: {"score": 0.0}\nPrompt: {prompt}'

    rendered = _render_prompt_template(template, prompt="a {red} cube")

    assert rendered == 'Return JSON only: {"score": 0.0}\nPrompt: a {red} cube'


def test_prompt_template_keeps_format_style_escaped_json_supported() -> None:
    template = 'Return JSON only: {{"score": 0.0}}\nPrompt: {prompt}'

    rendered = _render_prompt_template(template, prompt="a red cube")

    assert rendered == 'Return JSON only: {"score": 0.0}\nPrompt: a red cube'


@pytest.mark.asyncio
async def test_score_batch_sends_prompt_to_cli_stdin_and_image_path() -> None:
    reward = _FakeCodexImageQAReward('{"score": 0.8}')

    scores = await reward.score_batch([_rollout("a red cube")])

    assert scores == pytest.approx([0.8])
    assert reward.last_command is not None
    assert reward.last_command[0:2] == ["judge", "--image"]
    assert reward.last_command[2].endswith("image.png")
    assert reward.last_command[3] == "--out"
    assert reward.last_command[4].endswith("judge_output.txt")
    assert reward.last_command[-1] == "-"
    assert "a red cube" in reward.last_stdin
