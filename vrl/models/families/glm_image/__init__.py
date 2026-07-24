"""GLM-Image family — ZhipuAI's hybrid AR + diffusion t2i model.

Supports the ``zai-org/GLM-Image`` checkpoint for text-to-image generation
under the visual-rl GRPO pipeline. The 9B AR section
(``GlmImageForConditionalGeneration``, transformers >= 5.13) is the trainable
policy; the 7B DiT decoder (``GlmImagePipeline``, diffusers >= 0.37) is a
frozen postprocess.
"""

from __future__ import annotations

# Deliberately exports nothing. The family registry dispatches by dotted
# submodule path (vrl/families/registry.py), so a package-root re-export is a
# second surface nothing imports; keeping this module empty is also what stops
# config discovery from pulling the torch-backed model runtime.
__all__: list[str] = []
