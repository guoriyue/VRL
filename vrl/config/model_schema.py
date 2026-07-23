"""Lightweight public schema for the YAML ``model`` section.

The runtime registry refers to these classes by dotted path. Keeping them free
of family model imports lets config discovery validate family-owned keys
without importing torch, diffusers, or upstream model packages.
"""

from __future__ import annotations

from typing import Annotated, Any

from vrl.config.base import ConfigBase
from vrl.config.unknown_keys import ConfigBlock
from vrl.models.interfaces.runtime import MODEL_MEMORY_SECTIONS


class ModelSection(ConfigBase):
    """Keys shared by every registered model family."""

    family: str
    # Readers: ModelBuild plus family runtime LoRA projections.
    lora: Annotated[
        Any,
        ConfigBlock(
            (
                "rank",
                "alpha",
                "path",
                "target_modules",
                "init_lora_weights",
                "dropout",
                "init",
            )
        ),
    ] = None
    # Global section shape; the selected family validates supported targets.
    memory: Annotated[Any, ConfigBlock(MODEL_MEMORY_SECTIONS)] = None
    path: Any = None
    # Immutable Hub snapshot used by full-pipeline rollout and component replay.
    revision: Any = None
    torch_compile: Annotated[Any, ConfigBlock(("enable", "mode"))] = None
    use_lora: Any = None
    # Shared DiffusionChunkExecutor constructor values. The selected family
    # validates this block at typed parse and again at launch projection.
    executor: Annotated[
        Any,
        ConfigBlock(
            (
                "num_frames",
                "max_sequence_length",
                "fps",
                "chunk_passthrough_keys",
            )
        ),
    ] = None


__all__ = [
    "ModelSection",
]
