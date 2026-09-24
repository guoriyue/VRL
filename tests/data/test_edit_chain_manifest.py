"""Edit-chain manifests declare a source and a step schedule; steps own no conditioning."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from vrl.trainers.data.edit_chains import EditChain, load_edit_chains
from vrl.trainers.data.prompts import PromptExample


def _manifest(tmp_path, rows):
    Image.new("RGB", (4, 4), "white").save(tmp_path / "page.png")
    path = tmp_path / "chains.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_rows_resolve_source_and_accept_string_or_object_steps(tmp_path) -> None:
    path = _manifest(
        tmp_path,
        [
            {
                "chain_id": "comic-1",
                "source": "page.png",
                "steps": [
                    "Recolor the coat blue.",
                    {"prompt": "Fix the second bubble.", "target_text": "HELLO", "panel": 2},
                ],
                "metadata": {"page": "p1"},
            },
            {"source": "./page.png", "steps": ["Sharpen."]},
        ],
    )
    chains = load_edit_chains(path)
    assert [chain.chain_id for chain in chains] == ["comic-1", "chains:2"]
    first = chains[0]
    assert first.source == str(tmp_path / "page.png")
    assert first.steps[0] == PromptExample(prompt="Recolor the coat blue.")
    # Unknown step keys become step metadata, as in prompt manifests.
    assert first.steps[1].target_text == "HELLO"
    assert first.steps[1].metadata == {"panel": 2}

    step = first.step_example(1, "/tmp/parent.png", parent_sample_id="req:0:1")
    assert step.reference_images == ["/tmp/parent.png"]
    assert step.reward_metadata() == {
        "page": "p1",
        "panel": 2,
        "prompt_id": "comic-1:1",
        "chain_id": "comic-1",
        "chain_step": 1,
        "chain_length": 2,
        "chain_source": first.source,
        "chain_parent_sample_id": "req:0:1",
        "target_text": "HELLO",
    }
    assert first.step_example(0, first.source, parent_sample_id=None).reference_images == [
        first.source
    ]


@pytest.mark.parametrize(
    ("row", "match"),
    [
        ({"source": "page.png", "steps": []}, "non-empty list"),
        ({"source": "missing.png", "steps": ["a"]}, "does not exist"),
        ({"source": "page.png", "steps": ["a"], "extra": 1}, "unknown edit chain fields"),
        (
            {"source": "page.png", "steps": [{"prompt": "a", "reference_image": "x.png"}]},
            "conditioning",
        ),
        ({"source": "page.png", "steps": [{"prompt": "a", "chain_step": 3}]}, "collector-owned"),
    ],
)
def test_invalid_rows_are_rejected(tmp_path, row, match) -> None:
    path = _manifest(tmp_path, [row])
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        load_edit_chains(path)


def test_duplicate_chain_ids_are_rejected(tmp_path) -> None:
    path = _manifest(
        tmp_path,
        [
            {"chain_id": "same", "source": "page.png", "steps": ["a"]},
            {"chain_id": "same", "source": "page.png", "steps": ["b"]},
        ],
    )
    with pytest.raises(ValueError, match="unique"):
        load_edit_chains(path)


def test_map_steps_keeps_chain_identity() -> None:
    chain = EditChain("c", "/src.png", [PromptExample(prompt="a", references=["r.png"])])
    mapped = chain.map_steps(lambda step: PromptExample(prompt=step.prompt.upper()))
    assert mapped.chain_id == "c" and mapped.source == "/src.png"
    assert [step.prompt for step in mapped.steps] == ["A"]
    assert chain.steps[0].prompt == "a"
