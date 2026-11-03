"""Cosmos Predict2 / Predict2.5 transformer blocks over the fused DiT elementwise kernels.

:func:`install_fused_blocks` swaps every ``CosmosTransformerBlock`` of a root
to :class:`FusedCosmosTransformerBlock` and its self-attention processor to
:class:`FusedCosmosAttnProcessor`. Modules, parameters and state-dict names
stay; only the elementwise chains between the projections change
(``vrl.nn.kernels.dit_elementwise``):

- each AdaLN-Zero LayerNorm and its modulation in one pass, the table read
  per sample or, with the per-frame timesteps video-to-world conditions on,
  per token;
- the gated residual adds;
- the self-attention's per-head q/k RMSNorm and rotate-half RoPE in one pass,
  where the stock path normalizes ``[B, H, S, D]`` transposes and rotates
  through fp32 copies.

The AdaLN projections, the cross-attention (text, and Predict2.5's image
context) and the feed-forward keep the stock modules. ``model.fused_kernels``
installs the blocks in the rollout and the replay alike, so both sides keep
running one computation. The swap is not bitwise to the stock block: it
rounds once where the stock chains round two or three times.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_cosmos import (
    CosmosAdaLayerNormZero,
    CosmosAttnProcessor2_0,
    CosmosTransformerBlock,
)

from vrl.nn.kernels.block_swap import swap_block_class, swap_processor
from vrl.nn.kernels.dit_elementwise import (
    gated_residual,
    layer_norm_modulate,
    rms_norm_rotate_half,
)


class FusedCosmosAttnProcessor(CosmosAttnProcessor2_0):
    """The stock self-attention with the q/k RMSNorm and RoPE fused; cross-attention as stock."""

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None or image_rotary_emb is None:
            return super().__call__(
                attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb
            )
        cos, sin = image_rotary_emb

        def heads(states: torch.Tensor) -> torch.Tensor:
            return states.unflatten(2, (attn.heads, -1))

        query = rms_norm_rotate_half(
            heads(attn.to_q(hidden_states)), attn.norm_q.weight, cos, sin, eps=attn.norm_q.eps
        )
        key = rms_norm_rotate_half(
            heads(attn.to_k(hidden_states)), attn.norm_k.weight, cos, sin, eps=attn.norm_k.eps
        )
        hidden_states = dispatch_attention_fn(
            query,
            key,
            heads(attn.to_v(hidden_states)),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


def _modulate(
    norm: CosmosAdaLayerNormZero,
    hidden_states: torch.Tensor,
    embedded_timestep: torch.Tensor,
    temb: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``CosmosAdaLayerNormZero``: its stock projection, then the fused norm and modulation."""

    modulation = norm.linear_2(norm.linear_1(norm.activation(embedded_timestep)))
    if temb is not None:
        modulation = modulation + temb
    shift, scale, gate = modulation.chunk(3, dim=-1)
    return layer_norm_modulate(hidden_states, scale, shift, eps=norm.norm.eps), gate


def _residual(hidden_states: torch.Tensor, gate: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``hidden_states + gate * y`` in the dtype the stock expression promotes to."""

    out_dtype = torch.promote_types(hidden_states.dtype, torch.promote_types(gate.dtype, y.dtype))
    return gated_residual(hidden_states, gate, y, out_dtype=out_dtype)


class FusedCosmosTransformerBlock(CosmosTransformerBlock):
    """``CosmosTransformerBlock`` whose norms, modulation, q/k norm, RoPE and residuals run fused."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor
        | None
        | tuple[torch.Tensor | None, torch.Tensor | None],
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        extra_pos_emb: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        controlnet_residual: torch.Tensor | None = None,
        latents: torch.Tensor | None = None,
        block_idx: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.before_proj is not None:
            hidden_states = self.before_proj(hidden_states) + latents
        if extra_pos_emb is not None:
            hidden_states = hidden_states + extra_pos_emb

        norm_hidden_states, gate = _modulate(self.norm1, hidden_states, embedded_timestep, temb)
        attn_output = self.attn1(norm_hidden_states, image_rotary_emb=image_rotary_emb)
        hidden_states = _residual(hidden_states, gate, attn_output)

        norm_hidden_states, gate = _modulate(self.norm2, hidden_states, embedded_timestep, temb)
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = _residual(hidden_states, gate, attn_output)

        norm_hidden_states, gate = _modulate(self.norm3, hidden_states, embedded_timestep, temb)
        hidden_states = _residual(hidden_states, gate, self.ff(norm_hidden_states))

        if controlnet_residual is not None:
            hidden_states += controlnet_residual
        if self.after_proj is not None:
            return hidden_states, self.after_proj(hidden_states)
        return hidden_states


def _check_fusable(block: CosmosTransformerBlock) -> None:
    """Refuse a block whose norms or self-attention the fused kernels do not compute."""

    for name in ("norm1", "norm2", "norm3"):
        norm = getattr(block, name)
        if type(norm) is not CosmosAdaLayerNormZero or norm.norm.weight is not None:
            raise ValueError(f"fused Cosmos blocks need an affine-free AdaLN-Zero at {name}")
    attn = block.attn1
    if not all(
        isinstance(norm, RMSNorm) and norm.weight is not None
        for norm in (attn.norm_q, attn.norm_k)
    ):
        raise ValueError("fused Cosmos blocks need weighted per-head q/k RMSNorms")
    if attn.to_k.out_features != attn.to_q.out_features:
        raise ValueError("fused Cosmos blocks need as many key heads as query heads")
    if type(attn.processor) is not CosmosAttnProcessor2_0:
        raise ValueError(
            f"fused Cosmos blocks replace CosmosAttnProcessor2_0, "
            f"found {type(attn.processor).__name__}"
        )


def _prepare(block: CosmosTransformerBlock) -> None:
    _check_fusable(block)
    swap_processor(block.attn1, FusedCosmosAttnProcessor)


def install_fused_blocks(root: nn.Module) -> int:
    """Swap every Cosmos block under ``root`` to the fused one; return how many are fused."""

    return swap_block_class(
        root, CosmosTransformerBlock, FusedCosmosTransformerBlock, prepare=_prepare
    )


__all__ = ["FusedCosmosAttnProcessor", "FusedCosmosTransformerBlock", "install_fused_blocks"]
