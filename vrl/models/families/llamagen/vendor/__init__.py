"""Vendored FoundationVision/LlamaGen model definitions.

LlamaGen has no HuggingFace ``transformers`` integration and its GPT cannot be
mapped onto ``LlamaForCausalLM`` (2D grid rope with a zeroed caption-prefix
rope table, interleaved-pair rotary convention, fused ``wqkv``, static
buffer KV cache — see ``gpt.py`` header for the full evidence), so the two
upstream module files are vendored here verbatim apart from the documented
import fix. Upstream: https://github.com/FoundationVision/LlamaGen (MIT,
Copyright (c) 2024 FoundationVision).
"""

from vrl.models.families.llamagen.vendor.gpt import (
    GPT_models,
    ModelArgs,
    Transformer,
)
from vrl.models.families.llamagen.vendor.vq_model import (
    VQ_models,
    VQModel,
)

__all__ = [
    "GPT_models",
    "ModelArgs",
    "Transformer",
    "VQModel",
    "VQ_models",
]
