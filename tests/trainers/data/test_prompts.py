from __future__ import annotations

import json

import pytest

from vrl.trainers.data.prompts import (
    ImageCaptionPromptDataset,
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


@pytest.mark.parametrize("field", ["metadata", "request_overrides"])
def test_jsonl_loader_rejects_nonobject_fields_at_source_line(field):
    payload = ("\n" + json.dumps({"prompt": "p", field: [["key", "value"]]})).encode()
    with pytest.raises(ValueError, match=f"snapshot.jsonl:2: {field} must be an object"):
        load_prompt_examples_from_jsonl_bytes(payload, source="snapshot.jsonl")


@pytest.mark.parametrize("row", [{}, {"prompt": 123}])
def test_jsonl_loader_requires_string_prompt_at_source_line(row):
    with pytest.raises(ValueError, match=r"snapshot.jsonl:1: prompt must be a string"):
        load_prompt_examples_from_jsonl_bytes(json.dumps(row).encode(), source="snapshot.jsonl")


def test_jsonl_loader_preserves_empty_prompt_and_optional_null_mappings():
    examples = load_prompt_examples_from_jsonl_bytes(
        b'{"prompt":"", "metadata":null, "request_overrides":null, "extra":3}'
    )
    assert examples == [PromptExample(prompt="", metadata={"extra": 3}, request_overrides={})]


@pytest.mark.parametrize("field", ["image", "caption"])
def test_image_caption_manifest_rejects_nonstring_fields(tmp_path, field):
    path = tmp_path / "images.jsonl"
    path.write_text(json.dumps({"image": "image.png", "caption": "p", field: 123}))
    with pytest.raises(ValueError, match=f"row 0 {field!r} must be a string"):
        ImageCaptionPromptDataset(path)


@pytest.mark.parametrize("field", ["metadata", "request_overrides"])
def test_image_caption_manifest_rejects_nonobject_fields(tmp_path, field):
    path = tmp_path / "images.jsonl"
    path.write_text(json.dumps({"image": "image.png", "caption": "p", field: []}))
    with pytest.raises(ValueError, match=f"row 0 {field!r} must be an object"):
        ImageCaptionPromptDataset(path)


def test_image_caption_json_error_identifies_manifest_and_physical_row(tmp_path):
    path = tmp_path / "captions.jsonl"
    path.write_text('{"image":"a.png","caption":"first"}\n\n{broken\n')
    with pytest.raises(ValueError, match="row 2: invalid JSON") as caught:
        ImageCaptionPromptDataset(path)
    assert str(path) in str(caught.value)
    assert isinstance(caught.value.__cause__, json.JSONDecodeError)


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_jsonl_preserves_unicode_separators_in_prompt(tmp_path, separator, newline):
    prompt = f"first{separator}second"
    payload = (json.dumps({"prompt": prompt}, ensure_ascii=False) + newline).encode("utf-8")
    manifest = tmp_path / "prompts.jsonl"
    manifest.write_bytes(payload)
    assert load_prompt_examples_from_jsonl_bytes(payload)[0].prompt == prompt
    assert JsonlPromptDataset(manifest).examples[0].prompt == prompt
