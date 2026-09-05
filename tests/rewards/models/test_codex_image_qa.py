"""Codex image-QA structured-output contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.codex_image_qa import (
    CodexImageQARewardModel,
    _extract_grid_scores,
    _render_command,
    _render_prompt_template,
    _write_output_schema,
)


def _render(command: list[str], tmp_path: Path) -> list[str]:
    return _render_command(
        command,
        image_path=tmp_path / "image.png",
        output_path=tmp_path / "output.json",
        output_schema_path=tmp_path / "schema.json",
        prompt="1girl",
    )


def test_codex_exec_automatically_receives_portable_output_schema(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    codex_path = tmp_path / "bin" / "codex"
    _write_output_schema(schema_path)

    command = _render([str(codex_path), "exec", "-"], tmp_path)

    assert command == [
        str(codex_path),
        "exec",
        "--output-schema",
        str(schema_path),
        "-",
    ]
    assert json.loads(schema_path.read_text(encoding="utf-8")) == {
        "type": "object",
        "properties": {
            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["score"],
        "additionalProperties": False,
    }


@pytest.mark.parametrize("count", [2, 4])
def test_grid_schema_requires_the_current_montage_count(tmp_path: Path, count: int) -> None:
    schema_path = tmp_path / f"grid-{count}.json"

    _write_output_schema(schema_path, count=count)

    scores = json.loads(schema_path.read_text(encoding="utf-8"))["properties"]["scores"]
    assert scores["minItems"] == count
    assert scores["maxItems"] == count


def test_compatible_command_can_place_the_schema_with_a_placeholder(tmp_path: Path) -> None:
    command = _render(
        ["image-judge", "--schema", "{output_schema_path}", "{image_path}"],
        tmp_path,
    )

    assert command == [
        "image-judge",
        "--schema",
        str(tmp_path / "schema.json"),
        str(tmp_path / "image.png"),
    ]


def test_grid_parser_rejects_a_partial_score_list() -> None:
    with pytest.raises(ValueError, match="Expected 4 grid scores, got 3"):
        _extract_grid_scores('{"scores":[0.91,0.9,0.88]}', count=4)


def test_shared_rubric_renders_the_schema_matching_response_contract() -> None:
    template = "Judge {count} image(s). Return {response_contract}. Prompt: {prompt}"

    assert _render_prompt_template(template, prompt="blue {glass}") == (
        'Judge 1 image(s). Return {"score": 0.37}. Prompt: blue {glass}'
    )
    assert _render_prompt_template(template, prompt="blue {glass}", count=4) == (
        'Judge 4 image(s). Return {"scores": [0.37, 0.81, ...]}. Prompt: blue {glass}'
    )


def test_scored_rollouts_keep_exact_images_prompt_and_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    output_dir = tmp_path / "scored_rollouts"
    model = CodexImageQARewardModel(
        {
            "command": ["unused-judge"],
            "images_per_call": 2,
            "scored_rollout_dir": str(output_dir),
        },
    )
    artifacts = [
        RewardInferenceArtifact(
            artifact_id=f"request:prompt:0:sample:{index}:in-memory",
            sample_id=f"request:prompt:0:sample:{index}",
            prompt="single anime girl, full body",
            path="",
            media=torch.full((3, 8, 8), float(index)),
            metadata={"rollout_policy_version": 7},
        )
        for index in range(2)
    ]
    monkeypatch.setattr(
        model,
        "_score_batch_grid",
        lambda _artifacts: [
            {"codex_image_qa": 0.25},
            {"codex_image_qa": 0.75},
        ],
    )

    assert model.score_batch(artifacts) == [
        {"codex_image_qa": 0.25},
        {"codex_image_qa": 0.75},
    ]

    batch_dir = output_dir / "batch-000000"
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["count"] == 2
    assert manifest["rollout_policy_version"] == 7
    assert [item["prompt"] for item in manifest["items"]] == [
        "single anime girl, full body",
        "single anime girl, full body",
    ]
    assert [item["judge_prompt"] for item in manifest["items"]] == [
        "single anime girl, full body",
        "single anime girl, full body",
    ]
    assert [item["scores"]["codex_image_qa"] for item in manifest["items"]] == [0.25, 0.75]
    assert (batch_dir / "sample-00.png").is_file()
    assert (batch_dir / "sample-01.png").is_file()
    assert (batch_dir / "montage.png").is_file()
