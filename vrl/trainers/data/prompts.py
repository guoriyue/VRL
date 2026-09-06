"""Prompt data loading utilities for RL training."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.config.schema import DataConfig

from torch.utils.data import Dataset

from vrl.config.data import manifest_sources
from vrl.generation import GenerationInput


@dataclass
class PromptExample:
    """A single training example loaded from a JSONL prompt file."""

    prompt: str
    target_text: str = ""
    reference_image: str | None = field(default=None, metadata={"artifact": True})
    reference_video: str | None = field(default=None, metadata={"artifact": True})
    # CONTRACT: clean targets stay manifest-relative for the whole run. Exactly
    # one target_image/target_video is the identity key into sft-latents shards;
    # target-similarity rewards resolve the same artifact per process. Load-time
    # reference resolution must never rewrite either target identity.
    target_image: str | None = field(default=None, metadata={"artifact": True})
    target_video: str | None = field(default=None, metadata={"artifact": True})
    references: list[str] = field(default_factory=list, metadata={"artifact": True})
    # Empty delegates the modality to the selected model-family registry entry.
    # A video default silently mislabeled image-family JSONL rows such as Anima.
    task_type: str = ""
    # Per-prompt sampling overrides. Keys are the selected family's sampling
    # section fields (plus the rollout denoise knobs); the collector rejects any
    # other key when it builds the generation request.
    request_overrides: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # These two methods are the ONLY place that maps PromptExample fields onto
    # the engine/reward split. Adding a field means extending exactly one of
    # them — nothing downstream picks fields out of an untyped kwargs dict.

    def generation_input(self) -> GenerationInput:
        """Engine-facing conditioning for this example.

        Reward-only targets travel separately through ``reward_metadata()``;
        generation receives only functional conditioning.
        """

        return GenerationInput(
            prompt=self.prompt,
            task_type=self.task_type or None,
            reference_image=self.reference_image or None,
            reference_video=self.reference_video or None,
        )

    def reward_metadata(self) -> dict[str, Any]:
        """Reward-scoring metadata attached to this example's rollout group."""

        metadata = dict(self.metadata)
        if self.target_text:
            metadata["target_text"] = self.target_text
        if self.references:
            metadata["references"] = list(self.references)
        if self.target_image is not None:
            metadata["target_image"] = self.target_image
        if self.target_video is not None:
            metadata["target_video"] = self.target_video
        return metadata


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
        return load_prompt_examples_from_jsonl_bytes(p.read_bytes(), source=p)
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


def load_prompt_examples_from_jsonl_bytes(
    payload: bytes,
    *,
    source: str | Path = "<prompt manifest>",
) -> list[PromptExample]:
    """Parse an immutable UTF-8 JSONL snapshot into prompt examples.

    Callers that already authenticated manifest bytes can pass that exact
    snapshot through training without reopening a mutable filesystem path.
    Unknown row fields retain the native manifest behavior: they are merged
    into ``PromptExample.metadata`` after any explicit metadata entries.
    """

    context = str(source)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{context}: prompt manifest must be valid UTF-8") from error

    examples: list[PromptExample] = []
    known_fields = set(PromptExample.__dataclass_fields__)
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{context}:{line_number}: invalid JSON") from error
        if not isinstance(obj, dict):
            raise ValueError(f"{context}:{line_number}: JSONL rows must be objects")
        extra_metadata = {key: value for key, value in obj.items() if key not in known_fields}
        prompt_fields = {key: value for key, value in obj.items() if key in known_fields}
        metadata = dict(prompt_fields.get("metadata") or {})
        metadata.update(extra_metadata)
        prompt_fields["metadata"] = metadata
        examples.append(PromptExample(**prompt_fields))
    return examples


def load_prompt_mixture(
    sources: Mapping[str, int | None],
    *,
    seed: int,
) -> list[PromptExample]:
    """Draw ``count`` prompts from each manifest and shuffle them into one set.

    The ratio between prompt families is an experiment knob — an anatomy
    majority that carries the quality gradient next to a safety-stress minority
    that exercises a guardrail, say — so it belongs in the recipe rather than in
    a mixed manifest built on the side. A source with no count contributes every
    row in file order.

    ``seed`` is required, and is a correctness requirement rather than a
    convenience: every rank builds this list independently while
    ``PromptBatchSampler`` hands out INDICES into it, so an unseeded draw would
    give rank 0's index 5 and rank 1's index 5 different prompts. The same
    applies across a resume, which restores the sampler generator and re-indexes
    into a freshly loaded list. There is deliberately no default — a silent one
    would hide which draw a run actually trained on.
    """

    rng = random.Random(seed)
    picked: list[PromptExample] = []
    for path, count in sources.items():
        available = load_prompt_manifest(path)
        if count is None:
            picked.extend(available)
            continue
        if count > len(available):
            raise ValueError(
                f"data.manifest[{path!r}] asks for {count} prompts, "
                f"only {len(available)} available",
            )
        picked.extend(rng.sample(available, count))
    rng.shuffle(picked)
    return picked


def load_prompt_image_manifest(
    path: str | Path,
    *,
    image_field: str = "image",
    caption_field: str = "caption",
    default_task_type: str = "image_to_video",
) -> list[PromptExample]:
    """Load image-caption JSONL rows as image-conditioned prompt examples."""

    return list(
        ImageCaptionPromptDataset(
            path,
            image_field=image_field,
            caption_field=caption_field,
            default_task_type=default_task_type,
        ).examples,
    )


def load_prompt_examples_from_config(data: DataConfig) -> list[PromptExample]:
    """Load examples the way the parsed ``data`` section declares."""

    manifest = data.manifest
    if not manifest:
        raise ValueError("config missing required field: data.manifest")
    preprocessing = data.preprocessing

    if data.loader == "prompt_manifest":
        sources = manifest_sources(manifest)
        if len(sources) == 1 and next(iter(sources.values())) is None:
            return load_prompt_manifest(next(iter(sources)))
        if data.mix_seed is None:
            raise ValueError("config missing required field: data.mix_seed")
        return load_prompt_mixture(sources, seed=int(data.mix_seed))

    if data.loader == "prompt_image_manifest":
        image_field = str((preprocessing.image_field if preprocessing else None) or "image")
        caption_field = str((preprocessing.caption_field if preprocessing else None) or "caption")
        return load_prompt_image_manifest(
            manifest,
            image_field=image_field,
            caption_field=caption_field,
            default_task_type=str(data.task_type or "image_to_video"),
        )

    raise ValueError(f"unknown data.loader={data.loader!r}")


class JsonlPromptDataset(Dataset):
    """Dataset that loads :class:`PromptExample` objects from a JSONL file.

    Each line must be a JSON object whose keys match the
    :class:`PromptExample` fields.  Only ``prompt`` is required; all
    other fields fall back to their dataclass defaults when absent.
    """

    def __init__(self, path: str | Path) -> None:
        manifest_path = Path(path)
        self.examples = load_prompt_examples_from_jsonl_bytes(
            manifest_path.read_bytes(),
            source=manifest_path,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        return {"prompt": ex.prompt, "metadata": ex.metadata, "example": ex}


class ImageCaptionPromptDataset(Dataset):
    """Dataset for JSONL rows containing image path + caption fields.

    The manifest schema intentionally stays small so I2V datasets can be
    generated by external image pipelines without knowing VRL's internal
    ``PromptExample`` field names.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        image_field: str = "image",
        caption_field: str = "caption",
        default_task_type: str = "image_to_video",
    ) -> None:
        self.examples: list[PromptExample] = []
        manifest_path = Path(path)
        with manifest_path.open(encoding="utf-8") as f:
            for row_index, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"{manifest_path}: row {row_index} must be an object")
                image = _required_string_field(
                    obj,
                    image_field,
                    manifest_path=manifest_path,
                    row_index=row_index,
                )
                caption = _required_string_field(
                    obj,
                    caption_field,
                    manifest_path=manifest_path,
                    row_index=row_index,
                )
                metadata = dict(obj.get("metadata") or {})
                metadata.update(
                    {
                        key: value
                        for key, value in obj.items()
                        if key
                        not in {
                            image_field,
                            caption_field,
                            "metadata",
                            "task_type",
                            "request_overrides",
                        }
                    },
                )
                task_type = str(obj.get("task_type") or default_task_type)
                request_overrides = dict(obj.get("request_overrides") or {})
                self.examples.append(
                    PromptExample(
                        prompt=caption,
                        reference_image=image,
                        task_type=task_type,
                        request_overrides=request_overrides,
                        metadata=metadata,
                    ),
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        return {"prompt": ex.prompt, "metadata": ex.metadata, "example": ex}


def _required_string_field(
    obj: dict[str, Any],
    field_name: str,
    *,
    manifest_path: Path,
    row_index: int,
) -> str:
    value = obj.get(field_name)
    if value is None or str(value).strip() == "":
        raise ValueError(
            f"{manifest_path}: row {row_index} missing required field {field_name!r}",
        )
    return str(value)


__all__ = [
    "ImageCaptionPromptDataset",
    "JsonlPromptDataset",
    "PromptExample",
    "load_prompt_examples_from_config",
    "load_prompt_examples_from_jsonl_bytes",
    "load_prompt_image_manifest",
    "load_prompt_manifest",
    "load_prompt_mixture",
]
