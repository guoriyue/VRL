"""GRPO algorithm family.

The package-level ``GRPO`` / ``GRPOConfig`` names intentionally refer to the
continuous diffusion variant, so existing diffusion training imports keep the
same behavior after the package split.
"""

from vrl.algorithms.grpo.continuous import GRPO, GRPOConfig
from vrl.algorithms.grpo.multisegment import (
    MultiSegmentTokenGRPO,
    MultiSegmentTokenGRPOConfig,
)
from vrl.algorithms.grpo.token import TokenGRPO, TokenGRPOConfig

__all__ = [
    "GRPO",
    "GRPOConfig",
    "MultiSegmentTokenGRPO",
    "MultiSegmentTokenGRPOConfig",
    "TokenGRPO",
    "TokenGRPOConfig",
]
