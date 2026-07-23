"""Lightweight public schema for the YAML ``model`` section.

The runtime registry refers to these classes by dotted path. Keeping them free
of family model imports lets config discovery validate family-owned keys
without importing torch, diffusers, or upstream model packages.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

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


class WanModelSection(ModelSection):
    """Wan-specific public model keys."""

    boundary_ratio: Any = None
    expert_lifecycle_profiling: bool = False
    offload_mode: Literal["none", "model", "sequential"] = "none"
    trainable_transformers: Any = None


class CosmosPredict25ModelSection(ModelSection):
    """Cosmos Predict2.5 public model keys."""

    skip_text_encoder: Any = None


class CosmosAnimaModelSection(ModelSection):
    """Cosmos Anima single-file artifact paths and scheduler key."""

    qwen_tokenizer_path: Any = None
    qwen_tokenizer_revision: Any = None
    scheduler_shift: Any = None
    t5_tokenizer_path: Any = None
    t5_tokenizer_revision: Any = None
    text_encoder_file: Any = None
    text_encoder_path: Any = None
    transformer_file: Any = None
    transformer_path: Any = None
    vae_file: Any = None
    vae_path: Any = None


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


class EchoModelSection(ModelSection):
    """JoyAI-Echo checkpoint and Gemma text-encoder keys."""

    gemma_path: Any = None
    gemma_revision: Any = None


class FluxModelSection(ModelSection):
    """FLUX public model keys."""

    # DiffusionNFT's frozen ``previous`` adapter switch.
    nft_previous_adapter: Any = None


class CausVidModelSection(ModelSection):
    """CausVid pinned source, Wan base, and released checkpoint keys."""

    accept_noncommercial_license: bool = False
    base_model_path: Any = None
    base_model_revision: Any = None
    causvid_source_path: Any = None
    causvid_source_revision: Any = None
    checkpoint_file: Any = None
    checkpoint_sha256: Any = None


class Magi1ModelSection(ModelSection):
    """MAGI-1 isolated runtime and checkpoint component paths."""

    checkpoint_path: Any = None
    config_path: Any = None
    python_executable: Any = None
    source_path: Any = None
    source_revision: Any = None
    t5_pretrained_path: Any = None
    timeout_seconds: Any = None
    vae_pretrained_path: Any = None


__all__ = [
    "CausVidModelSection",
    "CosmosAnimaModelSection",
    "CosmosPredict25ModelSection",
    "EchoModelSection",
    "FluxModelSection",
    "JanusProModelSection",
    "LlamaGenModelSection",
    "Magi1ModelSection",
    "ModelSection",
    "NextStep1ModelSection",
    "WanModelSection",
]
