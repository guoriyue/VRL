"""Janus-Pro family — DeepSeek's autoregressive multimodal model.

Currently supports Janus-Pro-1B for text-to-image generation under
the visual-rl GRPO pipeline.

Requires the upstream package ``deepseek-ai/Janus`` (not on PyPI):
    git clone https://github.com/deepseek-ai/Janus
    cd Janus && pip install -e .
"""

from __future__ import annotations

from vrl.models.ar.janus_pro.model import (
    JanusProConfig,
    JanusProModel,
    image_token_logits_from_hidden,
)
from vrl.models.ar.janus_pro.r1_types import (
    JanusR1GenerationResult,
    JanusR1Segment,
)
from vrl.models.ar.janus_pro.runtime import (
    JanusProChunkGatherer,
    JanusProPipelineExecutor,
    JanusProR1ChunkGatherer,
    JanusProR1PipelineExecutor,
    build_janus_pro_runtime_bundle,
    extract_janus_pro_runtime_spec,
)

__all__ = [
    "JanusProChunkGatherer",
    "JanusProConfig",
    "JanusProModel",
    "JanusProPipelineExecutor",
    "JanusProR1ChunkGatherer",
    "JanusProR1PipelineExecutor",
    "JanusR1GenerationResult",
    "JanusR1Segment",
    "build_janus_pro_runtime_bundle",
    "extract_janus_pro_runtime_spec",
    "image_token_logits_from_hidden",
]
