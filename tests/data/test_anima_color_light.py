"""Contract for the formal Anima color-and-light prompt corpus."""

from __future__ import annotations

import itertools
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

DATASET_DIR = Path("datasets/anima/color_light")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _normalized(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", text)


def _shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
    tokens = _normalized(text).split()
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def test_anima_color_light_is_balanced_ordered_and_disjoint() -> None:
    spec = json.loads((DATASET_DIR / "dataset_spec.json").read_text(encoding="utf-8"))
    train = _read_jsonl(DATASET_DIR / "train_prompts.jsonl")
    heldout = _read_jsonl(DATASET_DIR / "eval_prompts.jsonl")
    development = _read_jsonl(DATASET_DIR / "development_prompts.jsonl")
    axes = {axis["name"]: tuple(axis["buckets"]) for axis in spec["axes"]}
    axis_by_bucket = {bucket: axis for axis, buckets in axes.items() for bucket in buckets}

    assert len(train) == spec["train_prompt_count"] == 256
    assert len(heldout) == spec["eval_prompt_count"] == 96
    assert len(development) == spec["development_prompt_count"] == 32
    assert len(axis_by_bucket) == 16
    assert Counter(row["metadata"]["bucket"] for row in train) == {
        bucket: spec["train_prompts_per_bucket"] for bucket in axis_by_bucket
    }
    assert Counter(row["metadata"]["bucket"] for row in heldout) == {
        bucket: spec["eval_prompts_per_bucket"] for bucket in axis_by_bucket
    }
    assert Counter(row["metadata"]["bucket"] for row in development) == {
        bucket: spec["development_prompts_per_bucket"] for bucket in axis_by_bucket
    }

    update_width = spec["prompts_per_update"]
    train_blocks = [
        train[start : start + update_width] for start in range(0, len(train), update_width)
    ]
    for block in train_blocks:
        assert {axis_by_bucket[row["metadata"]["bucket"]] for row in block} == set(axes)
    for start in range(0, len(train_blocks), 4):
        superblock = train_blocks[start : start + 4]
        for axis, buckets in axes.items():
            observed = {
                row["metadata"]["bucket"]
                for block in superblock
                for row in block
                if axis_by_bucket[row["metadata"]["bucket"]] == axis
            }
            assert observed == set(buckets)

    cooccurrences: dict[tuple[str, str], Counter[tuple[str, str]]] = defaultdict(Counter)
    for block in train_blocks:
        bucket_by_axis = {
            axis_by_bucket[row["metadata"]["bucket"]]: row["metadata"]["bucket"] for row in block
        }
        for left, right in itertools.combinations(axes, 2):
            cooccurrences[(left, right)][(bucket_by_axis[left], bucket_by_axis[right])] += 1
    assert all(set(counts.values()) == {4} for counts in cooccurrences.values())

    train_text = {_normalized(str(row["prompt"])) for row in train}
    heldout_text = {_normalized(str(row["prompt"])) for row in heldout}
    development_text = {_normalized(str(row["prompt"])) for row in development}
    assert len(train_text) == len(train)
    assert len(heldout_text) == len(heldout)
    assert len(development_text) == len(development)
    assert train_text.isdisjoint(heldout_text)
    assert train_text.isdisjoint(development_text)


def test_anima_color_light_prompts_follow_the_language_contract() -> None:
    spec = json.loads((DATASET_DIR / "dataset_spec.json").read_text(encoding="utf-8"))
    contract = spec["prompt_contract"]
    rows = [
        *_read_jsonl(DATASET_DIR / "train_prompts.jsonl"),
        *_read_jsonl(DATASET_DIR / "eval_prompts.jsonl"),
    ]

    for row in rows:
        prompt = str(row["prompt"])
        metadata = row["metadata"]
        words = _word_tokens(prompt)
        sentence_count = len(re.findall(r"[.!?](?:\s|$)", prompt))
        assert set(row) == {"prompt", "metadata"}
        assert set(metadata) == {"bucket", "prompt_style", "source"}
        assert metadata["prompt_style"] == contract["prompt_style"]
        assert metadata["source"] == contract["source"]
        assert contract["required_term"] in prompt.casefold()
        assert contract["min_words"] <= len(words) <= contract["max_words"]
        assert contract["min_sentences"] <= sentence_count <= contract["max_sentences"]
        assert not any(phrase in prompt.casefold() for phrase in contract["forbidden_phrases"])


def test_anima_color_light_has_no_development_or_regression_leakage() -> None:
    train = _read_jsonl(DATASET_DIR / "train_prompts.jsonl")
    heldout = _read_jsonl(DATASET_DIR / "eval_prompts.jsonl")
    development = _read_jsonl(DATASET_DIR / "development_prompts.jsonl")
    general_eval = _read_jsonl(Path("datasets/anima/quality_v1/eval_prompts.jsonl"))

    heldout_normalized = {_normalized(str(row["prompt"])) for row in heldout}
    prior_eval_normalized = {
        _normalized(str(row["prompt"])) for row in [*development, *general_eval]
    }
    assert heldout_normalized.isdisjoint(prior_eval_normalized)

    train_shingles = [
        (_normalized(str(row["prompt"])), _shingles(str(row["prompt"]))) for row in train
    ]
    for heldout_row in heldout:
        heldout_prompt = _normalized(str(heldout_row["prompt"]))
        heldout_shingles = _shingles(str(heldout_row["prompt"]))
        for train_prompt, shingles in train_shingles:
            union = shingles | heldout_shingles
            assert len(shingles & heldout_shingles) / len(union) < 0.50
            assert SequenceMatcher(None, train_prompt, heldout_prompt).ratio() < 0.82
