"""Janus-Pro family — DeepSeek's autoregressive multimodal model.

Currently supports Janus-Pro-1B for text-to-image generation under
the visual-rl GRPO pipeline.

Requires the upstream package ``deepseek-ai/Janus`` (not on PyPI):
    git clone https://github.com/deepseek-ai/Janus
    cd Janus && pip install -e .
"""

from __future__ import annotations

JANUS_R1_SEGMENTS = ("initial_image", "selfcheck_text", "final_image")

from vrl.models.ar.janus_pro.model import (  # noqa: E402
    JanusProConfig,
    JanusProModel,
    image_token_logits_from_hidden,
)
from vrl.models.ar.janus_pro.runtime import (  # noqa: E402
    JanusProChunkExecutor,
    JanusProR1ChunkExecutor,
    JanusProR1ChunkGatherer,
    build_janus_pro_runtime_bundle,
    extract_janus_pro_runtime_spec,
)

__all__ = [
    "JANUS_R1_SEGMENTS",
    "JanusProChunkExecutor",
    "JanusProConfig",
    "JanusProModel",
    "JanusProR1ChunkExecutor",
    "JanusProR1ChunkGatherer",
    "build_janus_pro_runtime_bundle",
    "extract_janus_pro_runtime_spec",
    "image_token_logits_from_hidden",
]
