"""Lightweight public schema for the YAML ``model`` section.

The runtime registry refers to these classes by dotted path. Keeping them free
of family model imports lets config discovery validate family-owned keys
without importing torch, diffusers, or upstream model packages.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ConfigDict, Field, model_validator

from vrl.config.base import ConfigBase
from vrl.models.checkpoint_identity import checkpoint_identity_metadata


class _ClosedModelSection(ConfigBase):
    """Fail-closed base for the public ``model`` configuration subtree."""

    model_config = ConfigDict(extra="forbid")


class LoraSection(_ClosedModelSection):
    """Shared adapter inputs consumed by ``ModelBuild.lora``."""

    rank: int | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("value", required=True),
    )
    alpha: int | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("value", required=True),
    )
    path: str | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    target_modules: list[str] | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "value",
            required=True,
            canonicalize="sorted_unique",
        ),
    )
    init_lora_weights: str | bool | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    dropout: float | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("value", default=0.0),
    )
    init: str | bool | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )


class VaeDecodeMemorySection(_ClosedModelSection):
    """VAE decode memory switches consumed by the decode-memory policy."""

    tiling: bool | None = None
    slicing: bool | None = None


class ModelMemorySection(_ClosedModelSection):
    """Target-keyed generation memory configuration."""

    vae_decode: VaeDecodeMemorySection | None = None


# Runtime capabilities and generation-memory targets share this public section
# namespace. Derive it from the typed structure so adding a section cannot leave
# a stale hand-maintained allow-list behind.
MODEL_MEMORY_SECTIONS: tuple[str, ...] = tuple(ModelMemorySection.model_fields)


class TorchCompileSection(_ClosedModelSection):
    """Transformer compile inputs consumed by ``ModelBuild.torch_compile``."""

    enable: bool | None = None
    mode: str | None = None


class ModelExecutorSection(_ClosedModelSection):
    """Shared ``GenericDiffusionBatchExecutor`` constructor inputs."""

    @model_validator(mode="before")
    @classmethod
    def _reject_renamed(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and "chunk_passthrough_keys" in data:
            raise ValueError(
                "model.executor.chunk_passthrough_keys was renamed to "
                "model.executor.batch_passthrough_keys; update the config key",
            )
        return data

    num_frames: int | None = None
    max_sequence_length: int | None = None
    fps: int | None = None
    batch_passthrough_keys: list[str] | None = None


class ModelSection(_ClosedModelSection):
    """Keys shared by every registered model family."""

    family: str = Field(
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    # Readers: ModelBuild plus family runtime LoRA projections.
    lora: LoraSection | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "lora",
            enabled_by="use_lora",
        ),
    )
    # Global section shape; the selected family validates supported targets.
    memory: ModelMemorySection | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    path: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "source",
            source="main",
            revision_field="revision",
        ),
    )
    # Immutable Hub snapshot used by full-pipeline rollout and component replay.
    revision: str | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "source_revision",
            source="main",
        ),
    )
    # Reader: ModelBuild.pretrained_kwargs. Long-run configs can
    # fail closed on a missing cached artifact instead of consulting the Hub.
    # Load-source knob only (same weights either way), so identity-excluded.
    local_files_only: bool = Field(
        default=False,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    torch_compile: TorchCompileSection | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    use_lora: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("value", default=False),
    )
    # Shared GenericDiffusionBatchExecutor constructor values. The selected family
    # validates this block at typed parse and again at launch projection.
    executor: ModelExecutorSection | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )


__all__ = [
    "MODEL_MEMORY_SECTIONS",
    "LoraSection",
    "ModelExecutorSection",
    "ModelMemorySection",
    "ModelSection",
    "TorchCompileSection",
    "VaeDecodeMemorySection",
]
