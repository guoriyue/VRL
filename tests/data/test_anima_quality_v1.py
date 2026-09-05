"""Contract for the small reviewed Anima quality canary."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _read(name: str) -> list[dict[str, object]]:
    path = Path("datasets/anima/quality_v1") / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_anima_canary_is_balanced_natural_language_and_disjoint() -> None:
    train = _read("train_prompts.jsonl")
    heldout = _read("eval_prompts.jsonl")

    assert len(train) == 64
    assert len(heldout) == 32
    assert set(Counter(row["metadata"]["bucket"] for row in train).values()) == {4}
    assert set(Counter(row["metadata"]["bucket"] for row in heldout).values()) == {2}
    assert all(
        len({row["metadata"]["bucket"] for row in train[start : start + 4]}) == 4
        for start in range(0, len(train), 4)
    )
    assert {row["prompt"] for row in train}.isdisjoint(row["prompt"] for row in heldout)
    assert all(str(row["prompt"]).count(".") >= 2 for row in [*train, *heldout])
    assert all(
        set(row["metadata"]) == {"bucket", "prompt_style", "source"} for row in [*train, *heldout]
    )
