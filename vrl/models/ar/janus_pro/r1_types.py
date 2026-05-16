"""Typed payloads for Janus-Pro-R1-style generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class JanusR1Segment:
    """One trainable R1 generation segment and its replay context."""

    name: str
    token_ids: torch.Tensor
    token_log_probs: torch.Tensor | None
    token_mask: torch.Tensor
    prompt_embeds: torch.Tensor
    attention_mask: torch.Tensor
    visual: bool
    cfg: bool


@dataclass(slots=True)
class JanusR1GenerationResult:
    """Three-stage Janus-Pro-R1 generation result.

    Segment names are fixed by convention:
      - initial_image
      - selfcheck_text
      - final_image
    """

    initial_image: torch.Tensor
    final_image: torch.Tensor
    selfcheck: torch.Tensor
    segments: dict[str, JanusR1Segment]
    context: dict[str, Any]


__all__ = [
    "JanusR1GenerationResult",
    "JanusR1Segment",
]
