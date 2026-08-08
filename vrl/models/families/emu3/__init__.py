"""Emu3 family — BAAI's autoregressive multimodal model (HF-native).

Currently supports the HF-format ``BAAI/Emu3-Gen-hf`` checkpoint for
text-to-image generation under the visual-rl GRPO pipeline. Unlike janus_pro
(which needs the out-of-tree ``janus`` package), Emu3 ships inside
``transformers`` (>= 4.48), so no extra dependency is required.
"""

from __future__ import annotations

# Deliberately exports nothing. The family registry dispatches by dotted
# submodule path (vrl/models/families/registry.py), so a package-root re-export is a
# second surface nothing imports; keeping this module empty is also what stops
# config discovery from pulling the torch-backed model runtime.
__all__: list[str] = []
