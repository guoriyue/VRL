"""Joint text/image attention processors backed by VRL kernels."""

from __future__ import annotations

from typing import Any

import torch

from vrl.nn.kernels.attention.sdpa import TorchSDPAAttentionKernel


class SD3JointAttentionProcessor:
    """SD3 joint text/image attention processor using the VRL SDPA kernel."""

    def __init__(self, kernel: TorchSDPAAttentionKernel | None = None) -> None:
        self.kernel = kernel or TorchSDPAAttentionKernel()

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del args, kwargs, attention_mask
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            enc_query = attn.add_q_proj(encoder_hidden_states)
            enc_key = attn.add_k_proj(encoder_hidden_states)
            enc_value = attn.add_v_proj(encoder_hidden_states)

            enc_query = enc_query.view(batch_size, -1, attn.heads, head_dim).transpose(
                1,
                2,
            )
            enc_key = enc_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_value = enc_value.view(batch_size, -1, attn.heads, head_dim).transpose(
                1,
                2,
            )

            if attn.norm_added_q is not None:
                enc_query = attn.norm_added_q(enc_query)
            if attn.norm_added_k is not None:
                enc_key = attn.norm_added_k(enc_key)

            query = torch.cat([query, enc_query], dim=2)
            key = torch.cat([key, enc_key], dim=2)
            value = torch.cat([value, enc_value], dim=2)

        hidden_states = self.kernel(
            query,
            key,
            value,
            dropout_p=0.0,
            causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size,
            -1,
            attn.heads * head_dim,
        )
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states


__all__ = ["SD3JointAttentionProcessor"]
