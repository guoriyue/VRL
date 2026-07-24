"""Vendored FoundationVision/LlamaGen model definitions.

LlamaGen has no HuggingFace ``transformers`` integration and its GPT cannot be
mapped onto ``LlamaForCausalLM`` (2D grid rope with a zeroed caption-prefix
rope table, interleaved-pair rotary convention, fused ``wqkv``, static
buffer KV cache — see ``gpt.py`` header for the full evidence), so the two
upstream module files are vendored here verbatim apart from the documented
import fix. Upstream: https://github.com/FoundationVision/LlamaGen (MIT,
Copyright (c) 2024 FoundationVision).
"""

from __future__ import annotations

# Deliberately exports nothing. The family registry dispatches by dotted
# submodule path (vrl/families/registry.py), so a package-root re-export is a
# second surface nothing imports; keeping this module empty is also what stops
# config discovery from pulling the torch-backed model runtime.
__all__: list[str] = []
