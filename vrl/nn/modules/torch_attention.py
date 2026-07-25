"""Torch-native AR attention backend (``torch_native``).

The non-paged counterpart to the vLLM paged backend, named after SGLang's
``torch_native`` attention backend. It runs the family decoder trunk with
ordinary HF KV caching (no vLLM block tables / FlashAttention), so it works on
CPU and any device the trunk supports, behind the same :class:`ARAttentionBackend`
contract as :class:`~vrl.nn.modules.ar_decoder.VllmDecoderPagedAttentionBackend`.

The per-row KV lives in the backend's opaque ``sequence_states`` (one cache per
sequence), so the runner select/scatters rows exactly as it does for the paged
backend. Every family reaching this backend (emu3 / janus_pro / nextstep_1) is
driven through one plain HF trunk call — ``inputs_embeds`` / ``attention_mask`` /
``past_key_values`` with ``use_cache=True`` — so there is no per-family forward
to inject. Families that need extra trunk arguments (glm_image passes mrope
``position_ids``) drive their own cache in the family runner instead.
"""

from __future__ import annotations

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


class TorchNativeDecoderAttentionBackend(ARAttentionBackend):
    """Run a HF-style decoder trunk with its own KV cache, one cache per row.

    The trunk returns a batched KV (a ``DynamicCache`` or a plain tensor); this
    skeleton splits it into one cache per row for ``sequence_states`` and
    re-concatenates the selected rows the runner hands back, so row
    select/scatter is identical to the paged backend.
    """

    def __init__(self, *, trunk: Any, config: ARAttentionConfig) -> None:
        super().__init__(config)
        self.trunk = trunk
        self.backend_label = str(
            config.extra.get("backend_label", f"{config.family}_torch_native_attention")
        )

    @torch.no_grad()
    def prefill(self, request: ARAttentionPrefillInput) -> ARAttentionPrefillOutput:
        last_hidden, kv = self._forward(request.inputs_embeds, request.attention_mask)
        return ARAttentionPrefillOutput(
            last_hidden=last_hidden,
            sequence_states=tuple(ar_split_rows(kv, request.inputs_embeds.shape[0])),
        )

    @torch.no_grad()
    def step(self, request: ARAttentionStepInput) -> ARAttentionStepOutput:
        batch = request.input_embeds.shape[0]
        last_hidden, kv = self._forward(
            request.input_embeds,
            request.attention_mask,
            ar_concat_rows(list(request.sequence_states)),
        )
        return ARAttentionStepOutput(
            last_hidden=last_hidden,
            sequence_states=tuple(ar_split_rows(kv, batch)),
        )

    def _forward(
        self,
        embeds: torch.Tensor,
        mask: torch.Tensor,
        kv: Any = None,
    ) -> tuple[torch.Tensor, Any]:
        """One trunk forward -> (last-token hidden ``[B, H]``, batched KV).

        Prefill and step differ only in ``kv``; ``past_key_values=None`` is the
        HF default the prefill used to rely on implicitly.
        """

        outputs = self.trunk(
            inputs_embeds=embeds,
            attention_mask=mask,
            past_key_values=kv,
            use_cache=True,
            output_hidden_states=True,
        )
        past = getattr(outputs, "past_key_values", None)
        if past is None:
            raise RuntimeError(
                f"{self.backend_label} trunk forward returned no past_key_values; "
                "use_cache must be enabled",
            )
        return self._last_token_hidden(outputs), past

    @staticmethod
    def _last_token_hidden(outputs: Any) -> torch.Tensor:
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden_states = getattr(outputs, "hidden_states", None)
            if not hidden_states:
                raise RuntimeError(
                    "language model output must expose last_hidden_state or hidden_states",
                )
            hidden = hidden_states[-1]
        if hidden.ndim == 3:
            return hidden[:, -1, :]
        if hidden.ndim == 2:
            return hidden
        raise RuntimeError(f"language model hidden state must be rank 2 or 3, got {hidden.ndim}")


__all__ = ["TorchNativeDecoderAttentionBackend"]
