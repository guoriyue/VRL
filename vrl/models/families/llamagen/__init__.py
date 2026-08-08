"""LlamaGen family — FoundationVision's llama-style autoregressive T2I model.

Currently supports the t2i_XL_stage1_256 checkpoint (775M GPT-XL + VQ-16
tokenizer, flan-t5-xl caption conditioning) under the visual-rl GRPO pipeline.

Upstream has no HuggingFace integration; the GPT and VQGAN definitions are
vendored under ``vendor/`` (MIT, FoundationVision) — see the vendor headers
for why the GPT cannot be mapped onto ``LlamaForCausalLM``.
"""

from __future__ import annotations

# Deliberately exports nothing. The family registry dispatches by dotted
# submodule path (vrl/models/families/registry.py), so a package-root re-export is a
# second surface nothing imports; keeping this module empty is also what stops
# config discovery from pulling the torch-backed model runtime.
__all__: list[str] = []
