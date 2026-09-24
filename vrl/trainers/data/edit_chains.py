"""Edit chains: one source image and an ordered list of editing instructions.

A chain is the multi-step counterpart of a ``PromptExample``. Step ``k`` is an
ordinary editing prompt whose conditioning image is one output of step ``k-1``
(the chain's source for step 0). The chain never chooses actions or stops:
the schedule is declared in the manifest. Learned control lives outside the
framework, in ``agentic``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from vrl.trainers.data.prompts import PromptExample, prompt_example_from_row

# Reward metadata every step carries. Chain and step metadata may not declare
# them: the ledger and rewards must see the collector's lineage, not a copy.
CHAIN_METADATA_KEYS = frozenset(
    {
        "prompt_id",
        "chain_id",
        "chain_step",
        "chain_length",
        "chain_source",
        "chain_parent_sample_id",
    }
)

_ROW_FIELDS = frozenset({"chain_id", "source", "steps", "metadata"})


@dataclass
class EditChain:
    """A declared editing schedule over one source image."""

    chain_id: str
    # Resolved path of the image step 0 edits.
    source: str
    # Step prompts and their reward-only fields. Conditioning images are the
    # chain's business, so steps declare none.
    steps: list[PromptExample]
    # Reward metadata shared by every step; a step's own metadata wins.
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chain_id or not self.source:
            raise ValueError("edit chain needs a chain_id and a source image")
        if not self.steps:
            raise ValueError(f"edit chain {self.chain_id!r} declares no steps")
        for index, step in enumerate(self.steps):
            if step.reference_images or step.reference_video:
                raise ValueError(
                    f"edit chain {self.chain_id!r} step {index} declares conditioning media; "
                    "the chain supplies each step's reference image"
                )
        for owner, metadata in (
            ("chain", self.metadata),
            *((f"step {index}", step.metadata) for index, step in enumerate(self.steps)),
        ):
            reserved = CHAIN_METADATA_KEYS & set(metadata)
            if reserved:
                raise ValueError(
                    f"edit chain {self.chain_id!r} {owner} metadata sets collector-owned keys "
                    f"{sorted(reserved)}"
                )

    def step_example(
        self, index: int, parent_image: str, *, parent_sample_id: str | None
    ) -> PromptExample:
        """The one-shot prompt for step ``index`` conditioned on ``parent_image``."""

        step = self.steps[index]
        metadata = {
            **self.metadata,
            **step.metadata,
            "prompt_id": f"{self.chain_id}:{index}",
            "chain_id": self.chain_id,
            "chain_step": index,
            "chain_length": len(self.steps),
            "chain_source": self.source,
            "chain_parent_sample_id": parent_sample_id,
        }
        return replace(step, reference_images=[parent_image], metadata=metadata)

    def map_steps(self, transform: Callable[[PromptExample], PromptExample]) -> EditChain:
        """A copy whose steps went through ``transform`` (for example path resolution)."""

        return replace(self, steps=[transform(step) for step in self.steps])


def load_edit_chains(path: str | Path) -> list[EditChain]:
    """Parse an edit-chain JSONL manifest.

    Each row is ``{"source": image, "steps": [...], "chain_id"?: str, "metadata"?: {}}``.
    A step is an instruction string or a ``PromptExample`` row without conditioning
    media. ``source`` resolves relative to the manifest and must exist. Unknown row
    keys are rejected rather than merged: a chain has two metadata owners, so a
    silent merge could not say which one a stray key meant.
    """

    manifest = Path(path).expanduser().resolve()
    chains: list[EditChain] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        context = f"{manifest}:{line_number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{context}: invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{context}: edit chain rows must be objects")
        unknown = set(row) - _ROW_FIELDS
        if unknown:
            raise ValueError(f"{context}: unknown edit chain fields {sorted(unknown)}")
        source_text = row.get("source")
        if not isinstance(source_text, str) or not source_text.strip():
            raise ValueError(f"{context}: source must be an image path")
        source = (manifest.parent / Path(source_text).expanduser()).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"{context}: source image does not exist: {source}")
        raw_steps = row.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError(f"{context}: steps must be a non-empty list")
        steps = []
        for index, raw in enumerate(raw_steps):
            if isinstance(raw, str):
                raw = {"prompt": raw}
            if not isinstance(raw, dict):
                raise ValueError(f"{context}: step {index} must be a string or an object")
            steps.append(prompt_example_from_row(raw, context=f"{context} step {index}"))
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"{context}: metadata must be an object")
        chain_id = row.get("chain_id", f"{manifest.stem}:{line_number}")
        if not isinstance(chain_id, str) or not chain_id:
            raise ValueError(f"{context}: chain_id must be a non-empty string")
        chains.append(EditChain(chain_id, str(source), steps, dict(metadata)))
    if len({chain.chain_id for chain in chains}) != len(chains):
        raise ValueError(f"{manifest}: chain_id values must be unique")
    return chains


__all__ = ["CHAIN_METADATA_KEYS", "EditChain", "load_edit_chains"]
