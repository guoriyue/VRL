"""Janus-Pro family — DeepSeek's autoregressive multimodal model.

Currently supports Janus-Pro-1B for text-to-image generation under
the visual-rl GRPO pipeline.

Requires the upstream package ``deepseek-ai/Janus`` (not on PyPI):
    git clone https://github.com/deepseek-ai/Janus
    cd Janus && pip install -e .
"""

from __future__ import annotations

from vrl.models.families.janus_pro.policy import (
    JanusProConfig,
    JanusProPolicy,
    image_token_logits_from_hidden,
)
from vrl.models.families.janus_pro.r1_executor import JanusProR1PipelineExecutor
from vrl.models.families.janus_pro.r1_types import (
    JanusR1GenerationResult,
    JanusR1Segment,
)

__all__ = [
    "JanusProConfig",
    "JanusProPolicy",
    "JanusProR1PipelineExecutor",
    "JanusR1GenerationResult",
    "JanusR1Segment",
    "image_token_logits_from_hidden",
]
