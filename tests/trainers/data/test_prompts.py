from __future__ import annotations

import json

import pytest

from vrl.trainers.data.prompts import (
    ImageCaptionPromptDataset,
    PromptExample,
    load_prompt_dataset_index,
    prompt_example_from_row,
)
from vrl.utils.paths import RootedPaths


def _load(tmp_path, payload: bytes) -> list[PromptExample]:
    manifest = tmp_path / "prompts.jsonl"
    manifest.write_bytes(payload)
    return load_prompt_dataset_index(manifest)


def test_jsonl_rows_map_known_fields_and_fold_unknown_keys_into_metadata(tmp_path) -> None:
    rows = [
        {
            "prompt": "six dancers",
            "target_text": "six",
            "metadata": {"split": "train"},
            "expected_people": 6,
        },
        {"prompt": "eight dancers", "task_type": "text_to_image"},
    ]
    loaded = _load(tmp_path, ("\n" + "\n".join(json.dumps(row) for row in rows) + "\n").encode())
    assert loaded == [
        PromptExample(
            prompt="six dancers",
            target_text="six",
            metadata={"split": "train", "expected_people": 6},
        ),
        PromptExample(prompt="eight dancers", task_type="text_to_image"),
    ]
    assert loaded[0].generation_input().task_type is None


def test_jsonl_loader_rejects_non_utf8_payload(tmp_path) -> None:
    with pytest.raises(ValueError, match=r"prompts\.jsonl: prompt manifest must be valid UTF-8"):
        _load(tmp_path, b'\xff{"prompt":"x"}\n')


def test_jsonl_loader_rejects_non_object_row_with_line_number(tmp_path) -> None:
    with pytest.raises(ValueError, match=r"prompts\.jsonl:3: JSONL rows must be objects"):
        _load(tmp_path, b'\n{"prompt":"valid"}\n["not", "an", "object"]\n')


@pytest.mark.parametrize("field", ["metadata", "request_overrides"])
def test_jsonl_loader_rejects_nonobject_fields_at_source_line(tmp_path, field):
    payload = ("\n" + json.dumps({"prompt": "p", field: [["key", "value"]]})).encode()
    with pytest.raises(ValueError, match=f"prompts.jsonl:2: {field} must be an object"):
        _load(tmp_path, payload)


def test_jsonl_loader_requires_string_prompt_at_source_line(tmp_path):
    with pytest.raises(ValueError, match=r"prompts\.jsonl:1: prompt must be a string"):
        _load(tmp_path, b"{}")


def test_jsonl_loader_preserves_empty_prompt_and_optional_null_mappings(tmp_path):
    examples = _load(
        tmp_path, b'{"prompt":"", "metadata":null, "request_overrides":null, "extra":3}'
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


@pytest.mark.parametrize(("separator", "newline"), [("\u0085", "\r\n"), ("\u2028", "\n")])
def test_jsonl_preserves_unicode_separators_in_prompt(tmp_path, separator, newline):
    prompt = f"first{separator}second"
    payload = (json.dumps({"prompt": prompt}, ensure_ascii=False) + newline).encode("utf-8")
    assert _load(tmp_path, payload)[0].prompt == prompt


def test_reference_resolution_preserves_target_identity_fields(tmp_path) -> None:
    """Reference consumers get absolute paths without changing target lookup keys."""

    example = PromptExample(
        prompt="open the drawer",
        reference_images=["references/frame.ppm"],
        reference_video="references/context.mp4",
        target_image="targets/result.ppm",
        target_video="targets/result.mp4",
    )

    resolved = example.with_resolved_references(RootedPaths.data_root(tmp_path))

    assert resolved.reference_images == [str((tmp_path / "references/frame.ppm").resolve())]
    assert resolved.reference_video == str((tmp_path / "references/context.mp4").resolve())
    assert resolved.target_image == "targets/result.ppm"
    assert resolved.target_video == "targets/result.mp4"
    assert example.reference_images == ["references/frame.ppm"]


def test_single_reference_image_row_key_loads_as_a_one_element_list() -> None:
    """``reference_image: str`` is manifest spelling for a one-element ``reference_images``."""
    single = prompt_example_from_row({"prompt": "p", "reference_image": "ref.png"}, context="r")
    blank = prompt_example_from_row({"prompt": "q", "reference_image": "  "}, context="r")

    assert single.reference_images == ["ref.png"]
    assert "reference_image" not in single.metadata
    assert blank.reference_images == []
    with pytest.raises(ValueError, match="not both"):
        prompt_example_from_row(
            {"prompt": "p", "reference_image": "a.png", "reference_images": ["b.png"]},
            context="r",
        )
