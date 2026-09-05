"""Codex image-QA structured-output contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.codex_image_qa import (
    CodexImageQARewardModel,
    _extract_binary_guard_verdicts,
    _extract_grid_scores,
    _extract_reference_verdicts,
    _render_command,
    _render_prompt_template,
    _write_binary_guard_output_schema,
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


def _binary_guard_model() -> CodexImageQARewardModel:
    return CodexImageQARewardModel(
        {
            "command": ["unused-judge"],
            "comparison_mode": "binary_guard",
            "images_per_call": 8,
            "expected_group_size": 8,
            "binary_guard_prompt_template": (
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


def _binary_guard_response(
    *,
    grounding_failures: set[str] | None = None,
    integrity_failures: set[str] | None = None,
    reverse_rows: bool = False,
) -> str:
    grounding_failures = grounding_failures or set()
    integrity_failures = integrity_failures or set()
    candidate_ids = [f"C{index + 1}" for index in range(8)]
    if reverse_rows:
        candidate_ids.reverse()
    return json.dumps(
        {
            "candidates": [
                {
                    "id": candidate_id,
                    "carrier_grounding": (
                        "fail" if candidate_id in grounding_failures else "pass"
                    ),
                    "image_integrity": ("fail" if candidate_id in integrity_failures else "pass"),
                }
                for candidate_id in candidate_ids
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


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({}, "expected_group_size >= 2"),
        (
            {"expected_group_size": 8, "images_per_call": 4},
            "images_per_call to equal expected_group_size",
        ),
        (
            {"expected_group_size": 8, "images_per_call": 8},
            "requires binary_guard_prompt_template",
        ),
    ],
)
def test_binary_guard_requires_complete_group_configuration(
    overrides: dict[str, object],
    message: str,
) -> None:
    config = {
        "command": ["unused-judge"],
        "comparison_mode": "binary_guard",
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        CodexImageQARewardModel(config)


def test_binary_guard_scores_two_true_order_reversals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vrl.rewards.models.codex_image_qa as codex_image_qa

    model = _binary_guard_model()
    artifacts = _reference_artifacts(tmp_path)
    responses = iter(
        [
            _binary_guard_response(
                grounding_failures={"C2", "C4"},
                integrity_failures={"C3", "C4"},
            ),
            _binary_guard_response(
                grounding_failures={"C3", "C4"},
                integrity_failures={"C4"},
                reverse_rows=True,
            ),
        ],
    )
    observed_labels: list[tuple[str, ...]] = []
    observed_media_means: list[tuple[float, ...]] = []
    observed_prompts: list[str] = []

    def compose_grid(medias, _tile, out_path, *, labels=None):
        observed_labels.append(tuple(labels or ()))
        observed_media_means.append(tuple(float(media.mean()) for media in medias))
        out_path.write_bytes(b"grid")

    def run_command(_command, *, stdin_text, output_path, workdir):
        del output_path, workdir
        observed_prompts.append(stdin_text)
        return next(responses)

    monkeypatch.setattr(codex_image_qa, "_compose_grid", compose_grid)
    monkeypatch.setattr(model, "_run_command", run_command)

    scores = model.score_batch(artifacts)

    assert [score["codex_image_qa"] for score in scores] == [
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert scores[1] == {
        "codex_image_qa": 0.0,
        "codex_image_qa_forward": 0.0,
        "codex_image_qa_reverse": 1.0,
        "codex_image_qa_consensus": 0.0,
        "codex_image_qa_mirror_agreement": 0.0,
    }
    assert scores[2]["codex_image_qa_forward"] == 0.0
    assert scores[2]["codex_image_qa_reverse"] == 0.0
    assert scores[2]["codex_image_qa_mirror_agreement"] == 0.0
    assert scores[3]["codex_image_qa_mirror_agreement"] == 1.0
    assert observed_labels == [
        ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"),
        ("C8", "C7", "C6", "C5", "C4", "C3", "C2", "C1"),
    ]
    assert observed_media_means[1] == tuple(reversed(observed_media_means[0]))
    assert all('"carrier_grounding":"pass"' in prompt for prompt in observed_prompts)
    assert all('"image_integrity":"pass"' in prompt for prompt in observed_prompts)


def test_binary_guard_schema_and_parser_require_exact_candidate_identity(tmp_path: Path) -> None:
    schema_path = tmp_path / "binary-guard-schema.json"
    _write_binary_guard_output_schema(schema_path, ("C1", "C2"))

    candidate_schema = json.loads(schema_path.read_text(encoding="utf-8"))["properties"][
        "candidates"
    ]
    assert candidate_schema["minItems"] == candidate_schema["maxItems"] == 2
    properties = candidate_schema["items"]["properties"]
    assert properties["id"]["enum"] == ["C1", "C2"]
    assert set(properties["carrier_grounding"]["enum"]) == {"pass", "fail"}
    assert set(properties["image_integrity"]["enum"]) == {"pass", "fail"}

    parsed = _extract_binary_guard_verdicts(
        '{"candidates":['
        '{"id":"C2","carrier_grounding":"fail","image_integrity":"pass"},'
        '{"id":"C1","carrier_grounding":"pass","image_integrity":"pass"}]}',
        ("C1", "C2"),
    )
    assert list(parsed) == ["C1", "C2"]
    assert parsed["C1"].passes
    assert not parsed["C2"].passes


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ('{"candidates":[],"extra":true}', "must contain only 'candidates'"),
        ('{"candidates":{}}', "candidate count mismatch"),
        (
            '{"candidates":[{"id":"C1","carrier_grounding":"pass","image_integrity":"pass"}]}',
            "candidate count mismatch",
        ),
        (
            '{"candidates":['
            '{"id":"C1","carrier_grounding":"pass","image_integrity":"pass","x":0},'
            '{"id":"C2","carrier_grounding":"pass","image_integrity":"pass"}]}',
            "must contain exactly",
        ),
        (
            '{"candidates":['
            '{"id":"C1","carrier_grounding":"pass"},'
            '{"id":"C2","carrier_grounding":"pass","image_integrity":"pass"}]}',
            "must contain exactly",
        ),
        (
            '{"candidates":['
            '{"id":"C3","carrier_grounding":"pass","image_integrity":"pass"},'
            '{"id":"C2","carrier_grounding":"pass","image_integrity":"pass"}]}',
            "unexpected binary-guard candidate id",
        ),
        (
            '{"candidates":['
            '{"id":"C1","carrier_grounding":"pass","image_integrity":"pass"},'
            '{"id":"C1","carrier_grounding":"pass","image_integrity":"pass"}]}',
            "duplicate binary-guard candidate id",
        ),
        (
            '{"candidates":['
            '{"id":"C1","carrier_grounding":"maybe","image_integrity":"pass"},'
            '{"id":"C2","carrier_grounding":"pass","image_integrity":"pass"}]}',
            "invalid binary-guard verdict",
        ),
        (
            '{"candidates":['
            '{"id":"C1","carrier_grounding":"pass","image_integrity":"pass"},'
            '{"id":"C2","carrier_grounding":"pass","image_integrity":"pass"}]} trailing',
            "Cannot parse binary-guard",
        ),
    ],
)
def test_binary_guard_parser_rejects_incomplete_or_malformed_output(
    response: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _extract_binary_guard_verdicts(response, ("C1", "C2"))


def test_binary_guard_rejects_incomplete_groups_and_mixed_prompts(tmp_path: Path) -> None:
    model = _binary_guard_model()

    with pytest.raises(ValueError, match="has 7 candidates; expected 8"):
        model.score_batch(_reference_artifacts(tmp_path, count=7))

    missing_group_id = _reference_artifacts(tmp_path)
    missing_group_id[0].metadata.pop("reward_group_id")
    with pytest.raises(ValueError, match="missing non-empty metadata"):
        model.score_batch(missing_group_id)

    mixed_prompts = _reference_artifacts(tmp_path)
    mixed_prompts[0] = replace(mixed_prompts[0], prompt="a different prompt")
    with pytest.raises(ValueError, match="mixes generation prompts"):
        model.score_batch(mixed_prompts)


def test_exact_count_compares_typed_target_with_two_observed_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vrl.rewards.models.codex_image_qa as codex_image_qa

    model = CodexImageQARewardModel(
        {
            "command": ["unused-judge", "{output_schema_path}"],
            "comparison_mode": "exact_count",
            "images_per_call": 8,
            "expected_group_size": 8,
            "exact_count_prompt_template": (
                "Count {count}. Target: {prompt}. Return {response_contract}"
            ),
            "prompt_metadata_key": "expected_people",
            "max_concurrency": 1,
            "tile_size": 64,
        },
    )
    artifacts = _reference_artifacts(tmp_path)
    for artifact in artifacts:
        artifact.metadata["expected_people"] = 4

    def response(
        counts: dict[str, int],
        *,
        ambiguous: set[str] | None = None,
        reverse_rows: bool = False,
    ) -> str:
        ambiguous = ambiguous or set()
        candidate_ids = [f"C{index + 1}" for index in range(8)]
        if reverse_rows:
            candidate_ids.reverse()
        return json.dumps(
            {
                "candidates": [
                    {
                        "id": candidate_id,
                        "observed_count": counts.get(candidate_id, 4),
                        "unambiguous": candidate_id not in ambiguous,
                    }
                    for candidate_id in candidate_ids
                ],
            },
        )

    responses = iter(
        [
            response({"C3": 5}, ambiguous={"C4"}),
            response({"C2": 3}, reverse_rows=True),
        ],
    )
    observed_labels: list[tuple[str, ...]] = []
    observed_media_means: list[tuple[float, ...]] = []
    observed_prompts: list[str] = []

    def compose_grid(medias, _tile, out_path, *, labels=None):
        observed_labels.append(tuple(labels or ()))
        observed_media_means.append(tuple(float(media.mean()) for media in medias))
        out_path.write_bytes(b"grid")

    def run_command(command, *, stdin_text, output_path, workdir):
        del output_path, workdir
        schema = json.loads(Path(command[1]).read_text(encoding="utf-8"))
        properties = schema["properties"]["candidates"]["items"]["properties"]
        assert properties["observed_count"] == {
            "type": "integer",
            "minimum": 0,
        }
        assert properties["unambiguous"] == {"type": "boolean"}
        observed_prompts.append(stdin_text)
        return next(responses)

    monkeypatch.setattr(codex_image_qa, "_compose_grid", compose_grid)
    monkeypatch.setattr(model, "_run_command", run_command)

    scores = model.score_batch(artifacts)

    assert [score["codex_image_qa"] for score in scores] == [
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert scores[1]["codex_image_qa_observed_forward"] == 4.0
    assert scores[1]["codex_image_qa_observed_reverse"] == 3.0
    assert scores[2]["codex_image_qa_mirror_agreement"] == 0.0
    assert scores[3]["codex_image_qa_unambiguous_forward"] == 0.0
    assert all(score["codex_image_qa_target"] == 4.0 for score in scores)
    assert observed_labels == [
        ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"),
        ("C8", "C7", "C6", "C5", "C4", "C3", "C2", "C1"),
    ]
    assert observed_media_means[1] == tuple(reversed(observed_media_means[0]))
    assert all("Target: 4" in prompt for prompt in observed_prompts)
    assert all('"observed_count":0' in prompt for prompt in observed_prompts)

    original = artifacts[0]
    artifacts[0] = replace(original, prompt="a different generation prompt")
    with pytest.raises(ValueError, match="mixes generation prompts"):
        model._candidate_groups(artifacts)
    artifacts[0] = original
    artifacts[0].metadata.pop("expected_people")
    with pytest.raises(ValueError, match=r"missing metadata\['expected_people'\]"):
        model._candidate_groups(artifacts)


def test_binary_guard_propagates_judge_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _binary_guard_model()
    artifacts = _reference_artifacts(tmp_path)

    def fail_command(_command, *, stdin_text, output_path, workdir):
        del stdin_text, output_path, workdir
        raise RuntimeError("judge failed")

    monkeypatch.setattr(model, "_run_command", fail_command)

    with pytest.raises(RuntimeError, match="judge failed"):
        model.score_batch(artifacts)


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
