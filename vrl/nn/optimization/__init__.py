"""Rollout optimization passes and the single point that applies them.

The public facade: consumers import from ``vrl.nn.optimization``, not from
``vrl.nn.optimization.passes``.
"""

from __future__ import annotations

from vrl.nn.optimization.passes import (
    ROLLOUT_PASSES,
    CompilePass,
    OptimizationPass,
    PassResult,
    QuantizationPass,
    apply_rollout_optimizations,
)

__all__ = [
    "ROLLOUT_PASSES",
    "CompilePass",
    "OptimizationPass",
    "PassResult",
    "QuantizationPass",
    "apply_rollout_optimizations",
]
