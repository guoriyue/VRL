"""Janus-Pro family — DeepSeek's autoregressive multimodal model.

Currently supports Janus-Pro-1B for text-to-image generation under
the visual-rl GRPO pipeline.

Requires the upstream package ``deepseek-ai/Janus`` (not on PyPI):
    git clone https://github.com/deepseek-ai/Janus
    cd Janus && pip install -e .
"""

from __future__ import annotations

JANUS_R1_SEGMENTS = ("initial_image", "selfcheck_text", "final_image")

from vrl.models.families.janus_pro.model import (  # noqa: E402
    JanusProConfig,
    JanusProModel,
    image_token_logits_from_hidden,
)
from vrl.models.families.janus_pro.runtime import (  # noqa: E402
    JanusProChunkExecutor,
    JanusProR1ChunkExecutor,
    JanusProR1ChunkGatherer,
)

__all__ = [
    "JANUS_R1_SEGMENTS",
    "JanusProChunkExecutor",
    "JanusProConfig",
    "JanusProModel",
    "JanusProR1ChunkExecutor",
    "JanusProR1ChunkGatherer",
    "image_token_logits_from_hidden",
]
