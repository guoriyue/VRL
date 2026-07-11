"""Torch-native AR attention backend (``torch_native``).

The non-paged counterpart to the vLLM paged backend, named after SGLang's
``torch_native`` attention backend. It runs the family decoder with ordinary HF
KV caching (no vLLM block tables / FlashAttention), so it works on CPU and any
device the trunk supports, behind the same :class:`ARAttentionBackend` contract
as :class:`~vrl.nn.modules.ar_decoder.VllmDecoderPagedAttentionBackend`.

The per-row KV lives in the backend's opaque ``sequence_states`` (one cache per
sequence), so the runner select/scatters rows exactly as it does for the paged
backend. Families differ only in their forward call (Janus: a plain HF trunk
forward returning a ``DynamicCache``; NextStep-1: ``_init_kv``/``_step_llm`` over
a ``{past_key_values, last_hidden}`` handle), so each family supplies its forward
via the builder while sharing this skeleton.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from vrl.nn.layers.attention.cache_rows import ar_concat_rows, ar_split_rows
from vrl.nn.layers.attention.paged import (
    ARAttentionBackend,
    ARAttentionConfig,
    ARAttentionPrefillInput,
    ARAttentionPrefillOutput,
    ARAttentionStepInput,
    ARAttentionStepOutput,
)

# (inputs_embeds, attention_mask) -> (last_hidden [B, H], kv_for_batch)
PrefillFn = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, Any]]
# (inputs_embeds [B, 1, H], attention_mask, kv_for_batch) -> (last_hidden [B, H], kv_for_batch)
StepFn = Callable[[torch.Tensor, torch.Tensor, Any], tuple[torch.Tensor, Any]]


class TorchNativeDecoderAttentionBackend(ARAttentionBackend):
    """HF KV-cache decoder backend; family forward supplied via ``prefill_fn``/``step_fn``.

    ``prefill_fn``/``step_fn`` return the family's batched KV (a ``DynamicCache`` or
    a ``{past_key_values, last_hidden}`` dict); this skeleton splits it into one
    cache per row for ``sequence_states`` and re-concatenates the selected rows the
    runner hands back, so row select/scatter is identical to the paged backend.
    """

    def __init__(
        self,
        *,
        config: ARAttentionConfig,
        prefill_fn: PrefillFn,
        step_fn: StepFn,
    ) -> None:
        super().__init__(config)
        self._prefill_fn = prefill_fn
        self._step_fn = step_fn

    @torch.no_grad()
    def prefill(self, request: ARAttentionPrefillInput) -> ARAttentionPrefillOutput:
        last_hidden, kv = self._prefill_fn(request.inputs_embeds, request.attention_mask)
        states = ar_split_rows(kv, request.inputs_embeds.shape[0])
        return ARAttentionPrefillOutput(
            last_hidden=last_hidden,
            sequence_states=tuple(states),
        )

    @torch.no_grad()
    def step(self, request: ARAttentionStepInput) -> ARAttentionStepOutput:
        batch = request.input_embeds.shape[0]
        kv = ar_concat_rows(list(request.sequence_states))
        last_hidden, kv = self._step_fn(request.input_embeds, request.attention_mask, kv)
        states = ar_split_rows(kv, batch)
        return ARAttentionStepOutput(
            last_hidden=last_hidden,
            sequence_states=tuple(states),
        )


def require_past_key_values(outputs: Any, label: str) -> Any:
    """Return ``outputs.past_key_values``, or raise if the trunk dropped the cache."""

    past = getattr(outputs, "past_key_values", None)
    if past is None:
        raise RuntimeError(
            f"{label} trunk forward returned no past_key_values; use_cache must be enabled",
        )
    return past


__all__ = [
    "PrefillFn",
    "StepFn",
    "TorchNativeDecoderAttentionBackend",
    "require_past_key_values",
]
