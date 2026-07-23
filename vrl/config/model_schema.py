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
    # The generation memory policy strictly validates each named subsection.
    memory: Annotated[Any, ConfigBlock(MODEL_MEMORY_SECTIONS)] = None
    path: Any = None
    # Immutable Hub snapshot used by full-pipeline rollout and component replay.
    revision: Any = None
    torch_compile: Annotated[Any, ConfigBlock(("enable", "mode"))] = None
    use_lora: Any = None
    # Shared DiffusionChunkExecutor constructor values. Family capability
    # validation remains at the runtime boundary.
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


class JanusProModelSection(ModelSection):
    """Janus-Pro optional model-wrapper keys."""

    trust_remote_code: Any = None
    vq_latent_channels: Any = None


class NextStep1ModelSection(ModelSection):
    """NextStep-1 tokenizer and frozen-module keys."""

    freeze_vae: Any = None
    vae_path: Any = None
    vae_revision: Any = None


class LlamaGenModelSection(ModelSection):
    """LlamaGen checkpoint-file and frozen-T5 keys."""

    gpt_ckpt: Any = None
    gpt_model: Any = None
    t5_path: Any = None
    t5_revision: Any = None
    vq_ckpt: Any = None


__all__ = [
    "JanusProModelSection",
    "LlamaGenModelSection",
    "ModelSection",
    "NextStep1ModelSection",
]
