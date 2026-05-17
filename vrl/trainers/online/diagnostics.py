"""Compatibility exports for trainer diagnostics."""

from vrl.utils.model_diagnostics import (
    parameter_state_summary,
    tensor_stats,
    trainable_state_digest,
    write_jsonl,
)

__all__ = [
    "parameter_state_summary",
    "tensor_stats",
    "trainable_state_digest",
    "write_jsonl",
]
