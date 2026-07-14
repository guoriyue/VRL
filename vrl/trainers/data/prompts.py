"""Prompt data loading utilities for RL training."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from vrl.generation import GenerationInput
from vrl.utils.config import cfg_get


@dataclass
class PromptExample:
    """A single training example loaded from a JSONL prompt file."""

    prompt: str
    target_text: str = ""
    reference_image: str | None = field(default=None, metadata={"artifact": True})
    reference_video: str | None = field(default=None, metadata={"artifact": True})
    target_image: str | None = field(default=None, metadata={"artifact": True})
    # CONTRACT: target_video stays manifest-relative for the whole run — it is
    # the identity key into sft-latents shards (save/load_sft_latents) and into
    # target-similarity rewards, which resolve it per-process themselves.
    # Load-time resolution (``_resolve_reference_artifacts``) covers REFERENCE
    # fields only and must never rewrite this one.
    target_video: str | None = field(default=None, metadata={"artifact": True})
    references: list[str] = field(default_factory=list, metadata={"artifact": True})
    task_type: str = "text_to_video"
    request_overrides: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # These two methods are the ONLY place that maps PromptExample fields onto
    # the engine/reward split. Adding a field means extending exactly one of
    # them — nothing downstream picks fields out of an untyped kwargs dict.

    def generation_input(self) -> GenerationInput:
        """Engine-facing conditioning for this example.

        Typed conditioning (task_type / reference media) rides the
        ``GenerationInput`` fields; everything else — including the reward-only
        targets — rides its metadata so engine sample rows can carry it.
        """

        return GenerationInput(
            prompt=self.prompt,
            task_type=self.task_type or None,
            reference_image=self.reference_image or None,
            reference_video=self.reference_video or None,
            metadata=self.reward_metadata(),
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


def load_prompt_examples_from_config(data_cfg: Any) -> list[PromptExample]:
    """Resolve ``data.loader`` and load examples from ``data.manifest``."""

    raw_loader = cfg_get(data_cfg, "loader", None)
    if raw_loader is None:
        # loader is optional for the prompt-* family: image-caption manifests are
        # the only ones whose preprocessing.format == "image_caption_jsonl", so the
        # plain prompt manifest is the default. pickapic_preference never reaches
        # this dispatch (it is loaded in scripts/data/bootstrap.py). Keep this rule
        # in sync with DataConfig._validate_data in vrl/config/schema.py.
        preprocessing = cfg_get(data_cfg, "preprocessing", {}) or {}
        fmt = str(cfg_get(preprocessing, "format", ""))
        loader = "prompt_image_manifest" if fmt == "image_caption_jsonl" else "prompt_manifest"
    else:
        loader = str(raw_loader)
    manifest = cfg_get(data_cfg, "manifest", None)
    if not manifest:
        raise ValueError("config missing required field: data.manifest")

    if loader == "prompt_manifest":
        return load_prompt_manifest(manifest)

    if loader == "prompt_image_manifest":
        preprocessing = cfg_get(data_cfg, "preprocessing", {}) or {}
        image_field = str(cfg_get(preprocessing, "image_field", "image"))
        caption_field = str(cfg_get(preprocessing, "caption_field", "caption"))
        task_type = str(cfg_get(data_cfg, "task_type", "image_to_video"))
        return load_prompt_image_manifest(
            manifest,
            image_field=image_field,
            caption_field=caption_field,
            default_task_type=task_type,
        )

    raise ValueError(f"unknown data.loader={loader!r}")


class JsonlPromptDataset(Dataset):
    """Dataset that loads :class:`PromptExample` objects from a JSONL file.

    Each line must be a JSON object whose keys match the
    :class:`PromptExample` fields.  Only ``prompt`` is required; all
    other fields fall back to their dataclass defaults when absent.
    """

    def __init__(self, path: str | Path) -> None:
        self.examples: list[PromptExample] = []
        known_fields = set(PromptExample.__dataclass_fields__)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"{path}: JSONL rows must be objects")
                extra_metadata = {
                    key: value for key, value in obj.items() if key not in known_fields
                }
                prompt_fields = {key: value for key, value in obj.items() if key in known_fields}
                metadata = dict(prompt_fields.get("metadata") or {})
                metadata.update(extra_metadata)
                prompt_fields["metadata"] = metadata
                self.examples.append(PromptExample(**prompt_fields))

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
    "load_prompt_image_manifest",
    "load_prompt_manifest",
]
