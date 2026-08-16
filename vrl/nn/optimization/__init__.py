"""Rollout optimization passes and the single point that applies them.

The public facade: consumers import from ``vrl.nn.optimization``, not from
``vrl.nn.optimization.passes``.
"""

from __future__ import annotations

from vrl.nn.optimization.passes import (
    REQUEST_SCOPED_DRIFT_SOURCES,
    ROLLOUT_PASSES,
    CompilePass,
    OptimizationPass,
    OptimizationReport,
    PassResult,
    QuantizationPass,
    apply_rollout_optimizations,
    unguarded_drift_sources,
)

__all__ = [
    "REQUEST_SCOPED_DRIFT_SOURCES",
    "ROLLOUT_PASSES",
    "CompilePass",
    "OptimizationPass",
    "OptimizationReport",
    "PassResult",
    "QuantizationPass",
    "apply_rollout_optimizations",
    "unguarded_drift_sources",
]
