"""Artifact dataset path resolution and manifest validation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from vrl.trainers.data.prompts import PromptExample, load_prompt_manifest
from vrl.utils.artifacts import (
    DATA_ROOT_ENV,
    ArtifactManifestError,
    default_data_root,
    repo_root,
    resolve_artifact_path,
)

# Derived from PromptExample fields tagged metadata={'artifact': True} — single source of truth.
DEFAULT_ARTIFACT_FIELDS = tuple(
    f.name for f in fields(PromptExample) if f.metadata.get("artifact")
)
IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".ppm", ".webp"}
# Shares the {source_repo, source_frame_index, decode_method, conditioning}
# provenance sub-vocabulary with the Image2Video manifest check in
# vrl/config/validation.py:_validate_image_to_video_manifest — keep in sync.
# Tuple order is load-bearing: required_metadata_fields is iterated in order and
# the first missing field is named in the error (tested in
# tests/data/test_video_world_manifests.py), so it is NOT auto-derived.
SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS = (
    "source",
    "source_repo",
    "source_split",
    "source_episode",
    "source_video",
    "source_frame_index",
    "decode_method",
    "conditioning",
)

@dataclass(frozen=True, slots=True)
class ResolvedArtifact:
    """One manifest artifact path resolved under the configured data root."""

    row_index: int
    field: str
    raw_path: str
    resolved_path: Path


@dataclass(frozen=True, slots=True)
class ArtifactManifestReport:
    """Validation report for one or two artifact manifests."""

    manifest_path: Path
    data_root: Path
    row_count: int
    artifact_count: int
    resolved_artifacts: tuple[ResolvedArtifact, ...] = ()
    warnings: tuple[str, ...] = ()
    source_episodes: tuple[str, ...] = ()
    eval_manifest_path: Path | None = None
    eval_source_episodes: tuple[str, ...] = ()
    source_episode_overlap: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report payload."""

        return {
            "manifest_path": self.manifest_path.as_posix(),
            "data_root": self.data_root.as_posix(),
            "row_count": self.row_count,
            "artifact_count": self.artifact_count,
            "resolved_artifacts": [
                {
                    "row_index": item.row_index,
                    "field": item.field,
                    "raw_path": item.raw_path,
                    "resolved_path": item.resolved_path.as_posix(),
                }
                for item in self.resolved_artifacts
            ],
            "warnings": list(self.warnings),
            "source_episodes": list(self.source_episodes),
            "eval_manifest_path": (
                None if self.eval_manifest_path is None else self.eval_manifest_path.as_posix()
            ),
            "eval_source_episodes": list(self.eval_source_episodes),
            "source_episode_overlap": list(self.source_episode_overlap),
        }

def resolve_prompt_example_artifacts(
    example: PromptExample,
    *,
    data_root: str | Path | None = None,
    allow_absolute: bool = False,
) -> PromptExample:
    """Return a copy of a prompt example with artifact paths resolved."""

    references = [
        str(resolve_artifact_path(item, data_root=data_root, allow_absolute=allow_absolute))
        for item in example.references
    ]
    reference_image = (
        str(
            resolve_artifact_path(
                example.reference_image,
                data_root=data_root,
                allow_absolute=allow_absolute,
            ),
        )
        if example.reference_image
        else ""
    )
    reference_video = (
        str(
            resolve_artifact_path(
                example.reference_video,
                data_root=data_root,
                allow_absolute=allow_absolute,
            ),
        )
        if example.reference_video
        else ""
    )
    target_image = (
        str(
            resolve_artifact_path(
                example.target_image,
                data_root=data_root,
                allow_absolute=allow_absolute,
            ),
        )
        if example.target_image
        else ""
    )
    target_video = (
        str(
            resolve_artifact_path(
                example.target_video,
                data_root=data_root,
                allow_absolute=allow_absolute,
            ),
        )
        if example.target_video
        else ""
    )
    metadata = dict(example.metadata)
    if reference_image:
        metadata["reference_image"] = reference_image
    if reference_video:
        metadata["reference_video"] = reference_video
    if target_image:
        metadata["target_image"] = target_image
    if target_video:
        metadata["target_video"] = target_video
    if references:
        metadata["references"] = references
    return PromptExample(
        prompt=example.prompt,
        target_text=example.target_text,
        reference_image=reference_image,
        reference_video=reference_video,
        target_image=target_image,
        target_video=target_video,
        references=references,
        task_type=example.task_type,
        request_overrides=dict(example.request_overrides),
        metadata=metadata,
    )


def validate_artifact_manifest(
    manifest_path: str | Path,
    *,
    data_root: str | Path | None = None,
    artifact_fields: Sequence[str] = DEFAULT_ARTIFACT_FIELDS,
    required_artifact_fields: Sequence[str] = (),
    required_metadata_fields: Sequence[str] = (),
    allow_absolute_paths: bool = False,
    require_readable: bool = True,
    reject_metadata_domain: bool = True,
) -> ArtifactManifestReport:
    """Validate one prompt manifest that references external binary artifacts."""

    path = Path(manifest_path)
    examples = load_prompt_manifest(path)
    root = _coerce_data_root(data_root)
    resolved: list[ResolvedArtifact] = []
    warnings: list[str] = []
    for row_index, example in enumerate(examples):
        metadata = dict(getattr(example, "metadata", None) or {})
        if reject_metadata_domain and "domain" in metadata:
            raise ArtifactManifestError(
                f"{path}: row {row_index} metadata.domain is reserved; "
                "use metadata.source or metadata.dataset instead",
            )
        for field_name in required_metadata_fields:
            value = metadata.get(field_name)
            if value is None or str(value).strip() == "":
                raise ArtifactManifestError(
                    f"{path}: row {row_index} metadata.{field_name} is required",
                )
        for field_name in required_artifact_fields:
            if not _artifact_values(example, field_name):
                raise ArtifactManifestError(
                    f"{path}: row {row_index} is missing required field {field_name}",
                )
        for field_name in artifact_fields:
            for raw_value in _artifact_values(example, field_name):
                resolved_path = resolve_artifact_path(
                    raw_value,
                    data_root=root,
                    allow_absolute=allow_absolute_paths,
                )
                if not resolved_path.exists():
                    raise ArtifactManifestError(
                        f"{path}: row {row_index} {field_name} does not exist: "
                        f"{resolved_path}",
                    )
                if require_readable:
                    _assert_readable(resolved_path, manifest_path=path, row_index=row_index)
                resolved.append(
                    ResolvedArtifact(
                        row_index=row_index,
                        field=field_name,
                        raw_path=str(raw_value),
                        resolved_path=resolved_path,
                    ),
                )
    source_episodes = tuple(sorted(_source_episodes(examples)))
    if not examples:
        warnings.append(f"{path}: manifest is empty")
    return ArtifactManifestReport(
        manifest_path=path,
        data_root=root,
        row_count=len(examples),
        artifact_count=len(resolved),
        resolved_artifacts=tuple(resolved),
        warnings=tuple(warnings),
        source_episodes=source_episodes,
    )


def validate_artifact_manifest_pair(
    train_manifest: str | Path,
    eval_manifest: str | Path,
    **kwargs: Any,
) -> ArtifactManifestReport:
    """Validate train/eval artifact manifests and report source-episode overlap."""

    train = validate_artifact_manifest(train_manifest, **kwargs)
    eval_report = validate_artifact_manifest(eval_manifest, **kwargs)
    overlap = tuple(
        sorted(set(train.source_episodes).intersection(eval_report.source_episodes)),
    )
    warnings = list(train.warnings)
    if overlap:
        warnings.append(
            "train/eval source_episode overlap: " + ", ".join(overlap),
        )
    return ArtifactManifestReport(
        manifest_path=train.manifest_path,
        data_root=train.data_root,
        row_count=train.row_count,
        artifact_count=train.artifact_count,
        resolved_artifacts=train.resolved_artifacts,
        warnings=tuple(warnings),
        source_episodes=train.source_episodes,
        eval_manifest_path=eval_report.manifest_path,
        eval_source_episodes=eval_report.source_episodes,
        source_episode_overlap=overlap,
    )


def validate_source_backed_video_world_manifest_pair(
    train_manifest: str | Path,
    eval_manifest: str | Path,
    *,
    data_root: str | Path | None = None,
    require_target_video: bool = False,
) -> ArtifactManifestReport:
    """Validate real Video2World manifests and first-frame reference provenance."""

    artifact_fields = ("reference_image", "target_video") if require_target_video else ("reference_image",)
    required_artifact_fields = artifact_fields
    return validate_artifact_manifest_pair(
        train_manifest,
        eval_manifest,
        data_root=data_root,
        artifact_fields=artifact_fields,
        required_artifact_fields=required_artifact_fields,
        required_metadata_fields=SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS,
    )


def _coerce_data_root(value: str | Path | None) -> Path:
    return Path(value).expanduser().resolve() if value is not None else default_data_root()


def _artifact_values(example: PromptExample, field_name: str) -> tuple[str, ...]:
    value = getattr(example, field_name, None)
    if value is None:
        value = example.metadata.get(field_name)
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value if str(item).strip())
    raise ArtifactManifestError(f"artifact field {field_name!r} must be a string or list")


def _source_episodes(examples: Sequence[PromptExample]) -> set[str]:
    out: set[str] = set()
    for example in examples:
        metadata = dict(getattr(example, "metadata", None) or {})
        value = metadata.get("source_episode")
        if value:
            out.add(str(value))
    return out


def _assert_readable(path: Path, *, manifest_path: Path, row_index: int) -> None:
    try:
        with path.open("rb") as handle:
            handle.read(1)
        if path.suffix.lower() in IMAGE_SUFFIXES:
            try:
                from PIL import Image
            except ImportError:
                return
            with Image.open(path) as image:
                image.verify()
    except Exception as exc:
        raise ArtifactManifestError(
            f"{manifest_path}: row {row_index} artifact is not readable: {path}",
        ) from exc


SFT_LATENTS_SCHEMA_VERSION = 1


def save_sft_latents(
    path: str | Path,
    *,
    family: str,
    model_path: str,
    latents_by_prompt: dict[str, Any],
) -> None:
    """Write the clean-latents shard the GRPO diffusion-loss regularizer reads.

    One file, one contract: ``{prompt -> [C, T, H, W] VAE latents}`` plus the
    provenance needed to reject a shard encoded with a different model. The
    producer is ``vrl/scripts/diffusion/encode_targets.py``.
    """

    import torch

    if not latents_by_prompt:
        raise ValueError("refusing to write an empty sft-latents shard")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": SFT_LATENTS_SCHEMA_VERSION,
            "family": str(family),
            "model_path": str(model_path),
            "latents": {
                str(prompt): value.detach().cpu()
                for prompt, value in latents_by_prompt.items()
            },
        },
        out,
    )


def load_sft_latents(path: str | Path, *, family: str | None = None) -> dict[str, Any]:
    """Load a clean-latents shard; optionally pin it to the training family."""

    import torch

    shard_path = Path(path).expanduser()
    if not shard_path.exists():
        raise FileNotFoundError(
            f"data.sft_latents shard not found: {shard_path} "
            "(produce it with vrl/scripts/diffusion/encode_targets.py)",
        )
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "latents" not in payload:
        raise ValueError(f"{shard_path} is not an sft-latents shard")
    version = int(payload.get("schema_version", 0))
    if version != SFT_LATENTS_SCHEMA_VERSION:
        raise ValueError(
            f"{shard_path}: unsupported sft-latents schema_version={version}; "
            f"expected {SFT_LATENTS_SCHEMA_VERSION}",
        )
    if family is not None and str(payload.get("family")) != str(family):
        raise ValueError(
            f"{shard_path} was encoded for family {payload.get('family')!r}, "
            f"but this run trains {family!r} — latent spaces are not "
            "interchangeable; re-encode with the training model",
        )
    latents = payload["latents"]
    if not isinstance(latents, dict) or not latents:
        raise ValueError(f"{shard_path}: empty sft-latents shard")
    return latents


__all__ = [
    "DATA_ROOT_ENV",
    "SFT_LATENTS_SCHEMA_VERSION",
    "SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS",
    "ArtifactManifestError",
    "ArtifactManifestReport",
    "ResolvedArtifact",
    "default_data_root",
    "load_sft_latents",
    "repo_root",
    "resolve_artifact_path",
    "resolve_prompt_example_artifacts",
    "save_sft_latents",
    "validate_artifact_manifest",
    "validate_artifact_manifest_pair",
    "validate_source_backed_video_world_manifest_pair",
]
