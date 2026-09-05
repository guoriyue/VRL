"""Codex image-QA structured-output contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.codex_image_qa import (
    CodexImageQARewardModel,
    _extract_grid_scores,
    _extract_reference_verdicts,
    _render_command,
    _render_prompt_template,
    _write_output_schema,
    _write_reference_output_schema,
)


def _render(command: list[str], tmp_path: Path) -> list[str]:
    return _render_command(
        command,
        image_path=tmp_path / "image.png",
        output_path=tmp_path / "output.json",
        output_schema_path=tmp_path / "schema.json",
        prompt="1girl",
    )


def _reference_model(tmp_path: Path) -> CodexImageQARewardModel:
    return CodexImageQARewardModel(
        {
            "command": ["unused-judge"],
            "comparison_mode": "reference_listwise",
            "images_per_call": 8,
            "expected_group_size": 8,
            "reference_data_root": str(tmp_path),
            "reference_prompt_template": (
                "Judge {count} candidates. Return {response_contract}. Prompt: {prompt}"
            ),
            "max_concurrency": 1,
            "tile_size": 64,
        },
    )


def _reference_artifacts(
    tmp_path: Path,
    *,
    count: int = 8,
    group_id: str = "request-0:prompt:0",
) -> list[RewardInferenceArtifact]:
    import torch
    from PIL import Image

    Image.new("RGB", (8, 8), "navy").save(tmp_path / "reference.png")
    return [
        RewardInferenceArtifact(
            artifact_id=f"{group_id}:sample:{index}",
            sample_id=f"{group_id}:sample:{index}",
            prompt="an anime room with coherent amber window light",
            path="",
            media=torch.full((3, 8, 8), index / 8),
            metadata={
                "reward_group_id": group_id,
                "target_image": "reference.png",
            },
        )
        for index in range(count)
    ]


def _reference_response(
    preferences: list[str],
    *,
    regress: set[str] | None = None,
) -> str:
    regress = regress or set()
    return json.dumps(
        {
            "candidates": [
                {
                    "id": f"C{index + 1}",
                    "integrity": "regress" if f"C{index + 1}" in regress else "pass",
                    "preference": preference,
                }
                for index, preference in enumerate(preferences)
            ],
        },
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


def test_reference_listwise_scores_one_complete_group_with_mirrored_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vrl.rewards.models.codex_image_qa as codex_image_qa

    model = _reference_model(tmp_path)
    artifacts = _reference_artifacts(tmp_path)
    response = _reference_response(
        ["strong_win", "win", "tie", "loss", "strong_loss", "strong_win", "tie", "loss"],
        regress={"C6"},
    )
    responses = iter([response, response])
    observed_labels: list[tuple[str, ...]] = []
    observed_prompts: list[str] = []

    def compose_grid(_medias, _tile, out_path, *, labels=None):
        observed_labels.append(tuple(labels or ()))
        out_path.write_bytes(b"grid")

    def run_command(_command, *, stdin_text, output_path, workdir):
        del output_path, workdir
        observed_prompts.append(stdin_text)
        return next(responses)

    monkeypatch.setattr(codex_image_qa, "_compose_grid", compose_grid)
    monkeypatch.setattr(model, "_run_command", run_command)

    scores = model.score_batch(artifacts)

    assert [score["codex_image_qa"] for score in scores] == [
        2.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert all(score["codex_image_qa_group_active"] == 1.0 for score in scores)
    assert scores[5]["codex_image_qa_integrity_pass"] == 0.0
    assert scores[5]["codex_image_qa_relative"] == -2.0
    assert all(score["codex_image_qa_mirror_agreement"] == 1.0 for score in scores)
    assert observed_labels == [
        ("R", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"),
        ("C8", "C7", "C6", "C5", "C4", "C3", "C2", "C1", "R"),
    ]
    assert all('"id":"C8"' in prompt for prompt in observed_prompts)


def test_reference_listwise_neutralizes_an_order_sensitive_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _reference_model(tmp_path)
    artifacts = _reference_artifacts(tmp_path)
    responses = iter(
        [
            _reference_response(["win", *("tie" for _ in range(7))]),
            _reference_response(["tie"] * 8),
        ],
    )
    monkeypatch.setattr(
        model,
        "_run_command",
        lambda _command, *, stdin_text, output_path, workdir: next(responses),
    )

    scores = model.score_batch(artifacts)

    assert [score["codex_image_qa"] for score in scores] == [0.0] * 8
    assert all(score["codex_image_qa_group_active"] == 0.0 for score in scores)
    assert scores[0]["codex_image_qa_mirror_agreement"] == 0.0
    assert scores[0]["codex_image_qa_relative"] == 0.0


def test_reference_schema_and_parser_require_exact_candidate_identity(tmp_path: Path) -> None:
    schema_path = tmp_path / "reference-schema.json"
    _write_reference_output_schema(schema_path, ("C1", "C2"))

    candidate_schema = json.loads(schema_path.read_text(encoding="utf-8"))["properties"][
        "candidates"
    ]
    assert candidate_schema["minItems"] == candidate_schema["maxItems"] == 2
    assert candidate_schema["items"]["properties"]["id"]["enum"] == ["C1", "C2"]
    assert set(candidate_schema["items"]["properties"]["preference"]["enum"]) == {
        "strong_loss",
        "loss",
        "tie",
        "win",
        "strong_win",
    }

    parsed = _extract_reference_verdicts(
        '{"candidates":['
        '{"id":"C2","integrity":"pass","preference":"win"},'
        '{"id":"C1","integrity":"regress","preference":"loss"}]}',
        ("C1", "C2"),
    )
    assert list(parsed) == ["C1", "C2"]
    with pytest.raises(ValueError, match="duplicate reference-listwise candidate id"):
        _extract_reference_verdicts(
            '{"candidates":['
            '{"id":"C1","integrity":"pass","preference":"tie"},'
            '{"id":"C1","integrity":"pass","preference":"tie"}]}',
            ("C1", "C2"),
        )


def test_reference_listwise_rejects_incomplete_group_metadata(tmp_path: Path) -> None:
    model = _reference_model(tmp_path)

    with pytest.raises(ValueError, match="has 7 candidates; expected 8"):
        model.score_batch(_reference_artifacts(tmp_path, count=7))

    missing_target = _reference_artifacts(tmp_path)
    missing_target[0].metadata.pop("target_image")
    with pytest.raises(ValueError, match="has no target_image"):
        model.score_batch(missing_target)

    missing_group_id = _reference_artifacts(tmp_path)
    missing_group_id[0].metadata.pop("reward_group_id")
    with pytest.raises(ValueError, match="missing non-empty metadata"):
        model.score_batch(missing_group_id)


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
