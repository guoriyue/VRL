"""Separable vocab-head contract for AR replay.

AR replay's memory hot spot is the final vocab projection: materializing
``[B, L, V]`` logits (and their gradient) when the evaluator only needs one
log-prob per position. ``vrl.nn.kernels.fused_linear_logprob`` removes that
tensor, but it needs the projection in algebraic form — a ``(weight, bias)``
GEMM it can chunk and recompute — which a black-box head module cannot
provide. This module is the contract by which a family DECLARES that split:

* :class:`VocabHeadSplit` — the family's structure knowledge, reduced to
  "everything before the final projection" plus the projection's tensors.
  ``ARModelBase.vocab_head_split`` returns it (or None), and
  ``ARModelBase._head_replay_values`` turns it into the replay payload that
  ``ReplaySegmentResult.logprobs`` consumes.

Families that cannot split return ``None`` from ``vocab_head_split`` and keep
shipping eager ``logits`` — known cases: emu3 applies a per-position
structural-token mask on the logits (the fused kernel has no mask input), and
llamagen's projection lives inside the vendored trunk forward.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch.nn as nn


@dataclass(frozen=True, slots=True)
class VocabHeadSplit:
    """A family's generation head, split at its final vocab projection.

    ``prefix`` maps trunk hidden states to the projection input (the
    projector+activation for janus-style MLP heads; ``None`` means the hidden
    states feed the projection directly); its output stays small.
    ``weight``/``bias`` are the final
    projection's tensors — pass slices when the sampleable vocab is a
    contiguous slice of a wider head (glm_image's codebook prefix).

    Callers must hand over PLAIN tensors whose values ARE the projection: a
    LoRA-wrapped final layer must not be split (``.weight`` would silently
    drop the adapter delta) — return ``None`` and fall back to eager logits.
    """

    weight: Any
    bias: Any | None = None
    prefix: Callable[[Any], Any] | None = None

    @classmethod
    def from_linear(
        cls,
        linear: Any,
        *,
        prefix: Callable[[Any], Any] | None = None,
        rows: int | None = None,
    ) -> VocabHeadSplit | None:
        """Build from a candidate final projection; None if it cannot be split.

        This owns the LoRA guard shared by every family probe: the check is
        EXACT type, not isinstance — PEFT's ``lora.Linear`` can subclass
        ``nn.Linear``, and reading ``.weight`` off a wrapped layer would
        silently drop the adapter delta, so anything but a plain Linear
        falls back to eager logits. ``rows`` slices the leading weight/bias
        rows (views, no copy) for heads whose sampleable vocab is a
        contiguous prefix of a wider projection (glm_image's codebook).
        """

        if type(linear) is not nn.Linear:
            return None
        weight = linear.weight if rows is None else linear.weight[:rows]
        bias = linear.bias
        if bias is not None and rows is not None:
            bias = bias[:rows]
        return cls(weight=weight, bias=bias, prefix=prefix)


__all__ = ["VocabHeadSplit"]
