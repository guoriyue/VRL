"""Wan 2.1 / 2.2 transformer blocks over the fused DiT elementwise kernels.

:func:`install_fused_blocks` swaps every ``WanTransformerBlock`` of a root to
:class:`FusedWanTransformerBlock` and its self-attention processor to
:class:`FusedWanAttnProcessor`. Modules, parameters and state-dict names
stay; only the elementwise chains between the projections change
(``vrl.nn.kernels.dit_elementwise``):

- the fp32 LayerNorm and its modulation in one pass, per sample (Wan 2.1, the
  2.2 A14B experts) or per token (2.2 TI2V's per-token timesteps), where the
  stock block upcasts the sequence, normalizes, scales, shifts and casts back
  in four fp32 passes;
- the cross-attention's affine fp32 LayerNorm in one pass;
- the q/k RMSNorm over all heads and each head's RoPE in one pass, where the
  stock path normalizes in bf16 and rotates through strided fp32 copies;
- the gated residual adds.

The cross-attention (text, and the I2V image context) and the feed-forward
keep the stock modules. ``model.fused_kernels`` installs the blocks in the
rollout and the replay alike, so both sides keep running one computation.
The swap is not bitwise to the stock block: it rounds once where the stock
chains round two or three times.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.transformers.transformer_wan import (
    WanAttention,
    WanAttnProcessor,
    WanTransformerBlock,
)

from vrl.nn.kernels.block_swap import swap_block_class, swap_processor
from vrl.nn.kernels.dit_elementwise import (
    gated_residual,
    layer_norm,
    layer_norm_modulate,
    rms_norm_rope_across_heads,
)


class FusedWanAttnProcessor(WanAttnProcessor):
    """The stock self-attention with the q/k RMSNorm and RoPE fused; cross-attention as stock."""

    def __call__(
        self,
        attn: WanAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if (
            rotary_emb is None
            or encoder_hidden_states is not None
            or getattr(attn, "fused_projections", False)
        ):
            return super().__call__(
                attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb
            )
        # Wan's [1, S, 1, D] repeat-interleaved tables as [S, D/2] rotations: the
        # stock rotation reads pair j's cosine at channel 2j, its sine at 2j + 1.
        freqs_cos, freqs_sin = rotary_emb
        freqs = torch.complex(freqs_cos[0, :, 0, 0::2].float(), freqs_sin[0, :, 0, 1::2].float())

        def heads(states: torch.Tensor) -> torch.Tensor:
            return states.unflatten(2, (attn.heads, -1))

        query = rms_norm_rope_across_heads(
            heads(attn.to_q(hidden_states)), attn.norm_q.weight, freqs, eps=attn.norm_q.eps
        )
        key = rms_norm_rope_across_heads(
            heads(attn.to_k(hidden_states)), attn.norm_k.weight, freqs, eps=attn.norm_k.eps
        )
        hidden_states = dispatch_attention_fn(
            query,
            key,
            heads(attn.to_v(hidden_states)),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


class FusedWanTransformerBlock(WanTransformerBlock):
    """``WanTransformerBlock`` whose norms, modulation, q/k norm, RoPE and residuals run fused."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            # [B, S, 6, D] (2.2 TI2V): a modulation row per token.
            modulation = (self.scale_shift_table.unsqueeze(0) + temb.float()).unbind(2)
        else:
            # [B, 6, D] (2.1, 2.2 A14B): a modulation row per sample.
            modulation = (self.scale_shift_table + temb.float()).unbind(1)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = modulation

        attn_output = self.attn1(
            layer_norm_modulate(hidden_states, scale_msa, shift_msa, eps=self.norm1.eps),
            None,
            None,
            rotary_emb,
        )
        hidden_states = gated_residual(hidden_states, gate_msa, attn_output)

        norm_hidden_states = hidden_states
        if isinstance(self.norm2, FP32LayerNorm):
            norm_hidden_states = layer_norm(
                hidden_states, self.norm2.weight, self.norm2.bias, eps=self.norm2.eps
            )
        hidden_states = hidden_states + self.attn2(
            norm_hidden_states, encoder_hidden_states, None, None
        )

        ff_output = self.ffn(
            layer_norm_modulate(hidden_states, c_scale_msa, c_shift_msa, eps=self.norm3.eps)
        )
        return gated_residual(hidden_states, c_gate_msa, ff_output)


def _check_fusable(block: WanTransformerBlock) -> None:
    """Refuse a block whose norms or self-attention the fused kernels do not compute."""

    for name in ("norm1", "norm3"):
        norm = getattr(block, name)
        if type(norm) is not FP32LayerNorm or norm.weight is not None:
            raise ValueError(f"fused Wan blocks need an affine-free FP32LayerNorm at {name}")
    if not isinstance(block.norm2, (FP32LayerNorm, nn.Identity)):
        raise ValueError(
            "fused Wan blocks need an FP32LayerNorm or no norm before cross-attention"
        )
    attn = block.attn1
    for norm in (attn.norm_q, attn.norm_k):
        if (
            type(norm) is not nn.RMSNorm
            or norm.weight is None
            or norm.eps is None
            or norm.normalized_shape != (attn.inner_dim,)
        ):
            raise ValueError("fused Wan blocks need weighted q/k RMSNorms across all heads")
    if type(attn.processor) is not WanAttnProcessor:
        raise ValueError(
            f"fused Wan blocks replace WanAttnProcessor, found {type(attn.processor).__name__}"
        )


def _prepare(block: WanTransformerBlock) -> None:
    _check_fusable(block)
    swap_processor(block.attn1, FusedWanAttnProcessor)


def install_fused_blocks(root: nn.Module) -> int:
    """Swap every Wan block under ``root`` to the fused one; return how many are fused."""

    return swap_block_class(root, WanTransformerBlock, FusedWanTransformerBlock, prepare=_prepare)


__all__ = ["FusedWanAttnProcessor", "FusedWanTransformerBlock", "install_fused_blocks"]
