"""Lightweight public schema for the YAML ``model`` section.

The runtime registry refers to these classes by dotted path. Keeping them free
of family model imports lets config discovery validate family-owned keys
without importing torch, diffusers, or upstream model packages.
"""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict

from vrl.config.base import ConfigBase


class _ClosedModelSection(ConfigBase):
    """Fail-closed base for the public ``model`` configuration subtree."""

    model_config = ConfigDict(extra="forbid")


class LoraSection(_ClosedModelSection):
    """Shared adapter inputs consumed by ``ModelBuild.lora``."""

    rank: int | None = None
    alpha: int | None = None
    path: str | None = None
    target_modules: list[str] | None = None
    init_lora_weights: str | bool | None = None
    dropout: float | None = None
    init: str | bool | None = None


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
    """Shared ``DiffusionChunkExecutor`` constructor inputs."""

    num_frames: int | None = None
    max_sequence_length: int | None = None
    fps: int | None = None
    chunk_passthrough_keys: list[str] | None = None


class ModelSection(_ClosedModelSection):
    """Keys shared by every registered model family."""

    family: str
    # Readers: ModelBuild plus family runtime LoRA projections.
    lora: LoraSection | None = None
    # Global section shape; the selected family validates supported targets.
    memory: ModelMemorySection | None = None
    path: Any = None
    # Immutable Hub snapshot used by full-pipeline rollout and component replay.
    revision: Any = None
    torch_compile: TorchCompileSection | None = None
    use_lora: Any = None
    # Shared DiffusionChunkExecutor constructor values. The selected family
    # validates this block at typed parse and again at launch projection.
    executor: ModelExecutorSection | None = None


__all__ = [
    "MODEL_MEMORY_SECTIONS",
    "LoraSection",
    "ModelExecutorSection",
    "ModelMemorySection",
    "ModelSection",
    "TorchCompileSection",
    "VaeDecodeMemorySection",
]
