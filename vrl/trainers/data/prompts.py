"""Prompt data loading utilities for RL training."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from vrl.utils.config import cfg_get


@dataclass
class PromptExample:
    """A single training example loaded from a JSONL prompt file."""

    prompt: str
    target_text: str = ""
    reference_image: str = field(default="", metadata={"artifact": True})
    reference_video: str = field(default="", metadata={"artifact": True})
    target_image: str = field(default="", metadata={"artifact": True})
    target_video: str = field(default="", metadata={"artifact": True})
    references: list[str] = field(default_factory=list, metadata={"artifact": True})
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


def _load_prompt_examples_from_config(
    data_cfg: Any,
    *,
    manifest_key: str,
    missing_message: str,
) -> list[PromptExample]:
    """Resolve ``data.loader`` and load examples from ``data.<manifest_key>``.

    The train (``data.manifest``) and eval (``data.eval_manifest``) entry points
    differ only in which manifest key they read and the message they fail with,
    so the loader resolution, presence check, and dispatch live here once and the
    two paths cannot drift apart.
    """

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
    manifest = cfg_get(data_cfg, manifest_key, None)
    if not manifest:
        raise ValueError(missing_message)

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


def load_prompt_examples_from_config(data_cfg: Any) -> list[PromptExample]:
    """Dispatch prompt example loading from a resolved ``data`` config section."""

    return _load_prompt_examples_from_config(
        data_cfg,
        manifest_key="manifest",
        missing_message="config missing required field: data.manifest",
    )


def load_eval_prompt_examples_from_config(data_cfg: Any) -> list[PromptExample]:
    """Load the fixed eval prompt set from ``data.eval_manifest``.

    Held-out eval prompts come from ``data.eval_manifest`` (not ``data.manifest``)
    so train and eval never share a prompt source; fail fast if eval is requested
    without one. Same loader types as training.
    """

    return _load_prompt_examples_from_config(
        data_cfg,
        manifest_key="eval_manifest",
        missing_message=(
            "trainer.eval.enabled=true requires data.eval_manifest "
            "(the fixed eval prompt set); none is configured"
        ),
    )


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
    "load_eval_prompt_examples_from_config",
    "load_prompt_examples_from_config",
    "load_prompt_image_manifest",
    "load_prompt_manifest",
]
