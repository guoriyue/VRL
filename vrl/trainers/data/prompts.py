"""Prompt data loading utilities for RL training."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


@dataclass
class PromptExample:
    """A single training example loaded from a JSONL prompt file."""

    prompt: str
    target_text: str = ""
    reference_image: str = ""
    reference_video: str = ""
    references: list[str] = field(default_factory=list)
    task_type: str = "text_to_video"
    request_overrides: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_prompt_manifest(path: str | Path) -> list[PromptExample]:
    """Load prompt examples from a manifest file. Supports two formats:

    * ``.jsonl``: one JSON per line with explicit fields — native
      :class:`PromptExample` manifest.
    * ``.txt``:   one prompt per line with target in double quotes,
      matching flow_grpo's ``dataset/ocr/train.txt`` convention. The
      target is extracted via ``prompt.split('"')[1]``.
    """
    p = Path(path)
    if p.suffix == ".jsonl":
        return list(JsonlPromptDataset(p).examples)
    if p.suffix == ".txt":
        examples: list[PromptExample] = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('"')
                target = parts[1] if len(parts) >= 3 else ""
                examples.append(PromptExample(prompt=line, target_text=target))
        return examples
    raise ValueError(f"Unsupported manifest suffix: {p.suffix}")


class JsonlPromptDataset(Dataset):
    """Dataset that loads :class:`PromptExample` objects from a JSONL file.

    Each line must be a JSON object whose keys match the
    :class:`PromptExample` fields.  Only ``prompt`` is required; all
    other fields fall back to their dataclass defaults when absent.
    """

    def __init__(self, path: str | Path) -> None:
        self.examples: list[PromptExample] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"{path}: JSONL rows must be objects")
                known_fields = set(PromptExample.__dataclass_fields__)
                extra_metadata = {
                    key: value for key, value in obj.items() if key not in known_fields
                }
                prompt_fields = {
                    key: value for key, value in obj.items() if key in known_fields
                }
                metadata = dict(prompt_fields.get("metadata") or {})
                metadata.update(extra_metadata)
                prompt_fields["metadata"] = metadata
                self.examples.append(PromptExample(**prompt_fields))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        return {"prompt": ex.prompt, "metadata": ex.metadata, "example": ex}

    @staticmethod
    def collate_fn(examples: list[dict[str, Any]]) -> tuple[list[str], list[dict]]:
        return (
            [e["prompt"] for e in examples],
            [e["metadata"] for e in examples],
        )


class TextPromptDataset(Dataset):
    """Simple dataset that loads prompts from a text file (one per line)."""

    def __init__(self, path: str) -> None:
        with open(path) as f:
            self.prompts = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples: list[dict[str, Any]]) -> tuple[list[str], list[dict]]:
        return (
            [e["prompt"] for e in examples],
            [e["metadata"] for e in examples],
        )


__all__ = [
    "JsonlPromptDataset",
    "PromptExample",
    "TextPromptDataset",
    "load_prompt_manifest",
]
