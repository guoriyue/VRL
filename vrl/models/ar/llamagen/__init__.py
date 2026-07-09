"""LlamaGen family — FoundationVision's llama-style autoregressive T2I model.

Currently supports the t2i_XL_stage1_256 checkpoint (775M GPT-XL + VQ-16
tokenizer, flan-t5-xl caption conditioning) under the visual-rl GRPO pipeline.

Upstream has no HuggingFace integration; the GPT and VQGAN definitions are
vendored under ``vendor/`` (MIT, FoundationVision) — see the vendor headers
for why the GPT cannot be mapped onto ``LlamaForCausalLM``.
"""

from __future__ import annotations

from vrl.models.ar.llamagen.model import (
    LLAMAGEN_CAPTION_TOKEN_NUM,
    LLAMAGEN_IMAGE_TOKEN_NUM,
    LLAMAGEN_IMAGE_VOCAB_SIZE,
    LlamaGenConfig,
    LlamaGenModel,
    LlamaGenReplayModel,
)
from vrl.models.ar.llamagen.runtime import (
    LLAMAGEN_FAMILY_CAPABILITY,
    LlamaGenChunkExecutor,
    build_llamagen_replay_runtime_bundle,
    build_llamagen_runtime_bundle,
    extract_llamagen_runtime_spec,
)

__all__ = [
    "LLAMAGEN_CAPTION_TOKEN_NUM",
    "LLAMAGEN_FAMILY_CAPABILITY",
    "LLAMAGEN_IMAGE_TOKEN_NUM",
    "LLAMAGEN_IMAGE_VOCAB_SIZE",
    "LlamaGenChunkExecutor",
    "LlamaGenConfig",
    "LlamaGenModel",
    "LlamaGenReplayModel",
    "build_llamagen_replay_runtime_bundle",
    "build_llamagen_runtime_bundle",
    "extract_llamagen_runtime_spec",
]
