from __future__ import annotations

import json

import pytest

from vrl.trainers.data.prompts import (
    JsonlPromptDataset,
    PromptExample,
    load_prompt_examples_from_jsonl_bytes,
)


def test_jsonl_bytes_loader_preserves_prompt_manifest_field_behavior(tmp_path) -> None:
    rows = [
        {
            "prompt": "six dancers",
            "target_text": "six",
            "metadata": {"split": "train"},
            "expected_people": 6,
        },
        {"prompt": "eight dancers", "task_type": "text_to_image"},
    ]
    payload = ("\n" + "\n".join(json.dumps(row) for row in rows) + "\n").encode()

    loaded = load_prompt_examples_from_jsonl_bytes(payload, source="frozen-prompts.jsonl")
    manifest = tmp_path / "prompts.jsonl"
    manifest.write_bytes(payload)

    assert loaded == JsonlPromptDataset(manifest).examples
    assert loaded == [
        PromptExample(
            prompt="six dancers",
            target_text="six",
            metadata={"split": "train", "expected_people": 6},
        ),
        PromptExample(prompt="eight dancers", task_type="text_to_image"),
    ]
    assert loaded[0].generation_input().task_type is None


def test_jsonl_bytes_loader_rejects_non_utf8_payload() -> None:
    with pytest.raises(ValueError, match=r"snapshot\.jsonl: prompt manifest must be valid UTF-8"):
        load_prompt_examples_from_jsonl_bytes(b'\xff{"prompt":"x"}\n', source="snapshot.jsonl")


def test_jsonl_bytes_loader_rejects_non_object_row_with_line_number() -> None:
    payload = b'\n{"prompt":"valid"}\n["not", "an", "object"]\n'

    with pytest.raises(ValueError, match=r"snapshot.jsonl:3: JSONL rows must be objects"):
        load_prompt_examples_from_jsonl_bytes(payload, source="snapshot.jsonl")


def test_jsonl_bytes_loader_requires_immutable_bytes() -> None:
    with pytest.raises(TypeError, match="must be immutable bytes"):
        load_prompt_examples_from_jsonl_bytes(bytearray(b'{"prompt":"x"}\n'))  # type: ignore[arg-type]
