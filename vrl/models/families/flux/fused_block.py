"""FLUX.1 transformer blocks over the fused DiT elementwise kernels.

:func:`install_fused_blocks` swaps every ``FluxTransformerBlock`` (double
stream) and ``FluxSingleTransformerBlock`` of a root to its fused subclass
and their attention processors to :class:`FusedFluxAttnProcessor`. Modules,
parameters and state-dict names stay; only the elementwise chains between the
projections change (``vrl.nn.kernels.dit_elementwise``):

- every AdaLN-Zero LayerNorm and its modulation in one pass, where the stock
  block normalizes in fp32 and scales and shifts through fp32 temporaries;
- the gated residual adds;
- the q/k RMSNorm of both streams and the RoPE in one pass, written straight
  into the joint ``[text, image]`` sequence, where the stock path normalizes,
  concatenates and rotates through strided fp32 copies.

The AdaLN projections, the GELU MLPs and the output projections keep the
stock modules. ``model.fused_kernels`` installs the blocks in the rollout and
the replay alike, so both sides keep running one computation. The swap is
not bitwise to the stock block: it rounds once where the stock chains round
two or three times.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.normalization import (
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
    FP32LayerNorm,
)
from diffusers.models.transformers.transformer_flux import (
    FluxAttention,
    FluxAttnProcessor,
    FluxSingleTransformerBlock,
    FluxTransformerBlock,
)

from vrl.nn.kernels.batch_invariant import per_sample_attention
from vrl.nn.kernels.block_swap import swap_block_class, swap_processor
from vrl.nn.kernels.dit_elementwise import (
    gated_residual,
    joint_rms_norm_rope,
    layer_norm_modulate,
)


def _residual(hidden_states: torch.Tensor, gate: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``hidden_states + gate * y`` in the dtype the stock expression promotes to."""

    out_dtype = torch.promote_types(hidden_states.dtype, torch.promote_types(gate.dtype, y.dtype))
    return gated_residual(hidden_states, gate, y, out_dtype=out_dtype)


class FusedFluxAttnProcessor(FluxAttnProcessor):
    """The stock joint attention with the q/k RMSNorm and RoPE fused into the concatenation."""

    def __call__(
        self,
        attn: FluxAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if image_rotary_emb is None or attn.fused_projections:
            return super().__call__(
                attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb
            )

        def heads(projection: Any, states: torch.Tensor) -> torch.Tensor:
            return projection(states).unflatten(-1, (-1, attn.head_dim))

        # The repeat-interleaved [S, D] tables as [S, D/2] rotations.
        cos, sin = image_rotary_emb
        freqs = torch.complex(cos[:, 0::2].float(), sin[:, 0::2].float())
        joint = encoder_hidden_states is not None
        if joint:
            # Joint order is [text, image], as the stock processor concatenates.
            text = encoder_hidden_states.shape[1]
            query = joint_rms_norm_rope(
                [heads(attn.add_q_proj, encoder_hidden_states), heads(attn.to_q, hidden_states)],
                [attn.norm_added_q.weight, attn.norm_q.weight],
                [freqs[:text], freqs[text:]],
                eps=attn.norm_q.eps,
            )
            key = joint_rms_norm_rope(
                [heads(attn.add_k_proj, encoder_hidden_states), heads(attn.to_k, hidden_states)],
                [attn.norm_added_k.weight, attn.norm_k.weight],
                [freqs[:text], freqs[text:]],
                eps=attn.norm_k.eps,
            )
            value = torch.cat(
                [heads(attn.add_v_proj, encoder_hidden_states), heads(attn.to_v, hidden_states)],
                dim=1,
            )
        else:
            query = joint_rms_norm_rope(
                [heads(attn.to_q, hidden_states)],
                [attn.norm_q.weight],
                [freqs],
                eps=attn.norm_q.eps,
            )
            key = joint_rms_norm_rope(
                [heads(attn.to_k, hidden_states)],
                [attn.norm_k.weight],
                [freqs],
                eps=attn.norm_k.eps,
            )
            value = heads(attn.to_v, hidden_states)

        hidden_states = per_sample_attention(
            dispatch_attention_fn,
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        if not joint:
            return hidden_states
        encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
            [text, hidden_states.shape[1] - text], dim=1
        )
        hidden_states = attn.to_out[0](hidden_states.contiguous())
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states, attn.to_add_out(encoder_hidden_states.contiguous())


class FusedFluxTransformerBlock(FluxTransformerBlock):
    """``FluxTransformerBlock`` whose norms, modulation and residuals run fused."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.norm1.linear(self.norm1.silu(temb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        context_emb = self.norm1_context.linear(self.norm1_context.silu(temb))
        c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            context_emb.chunk(6, dim=1)
        )

        attention_outputs = self.attn(
            hidden_states=layer_norm_modulate(
                hidden_states, scale_msa, shift_msa, eps=self.norm1.norm.eps
            ),
            encoder_hidden_states=layer_norm_modulate(
                encoder_hidden_states, c_scale_msa, c_shift_msa, eps=self.norm1_context.norm.eps
            ),
            image_rotary_emb=image_rotary_emb,
            **(joint_attention_kwargs or {}),
        )
        attn_output, context_attn_output = attention_outputs[:2]

        hidden_states = _residual(hidden_states, gate_msa, attn_output)
        ff_output = self.ff(
            layer_norm_modulate(hidden_states, scale_mlp, shift_mlp, eps=self.norm2.eps)
        )
        hidden_states = _residual(hidden_states, gate_mlp, ff_output)
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + attention_outputs[2]

        encoder_hidden_states = _residual(encoder_hidden_states, c_gate_msa, context_attn_output)
        context_ff_output = self.ff_context(
            layer_norm_modulate(
                encoder_hidden_states, c_scale_mlp, c_shift_mlp, eps=self.norm2_context.eps
            )
        )
        encoder_hidden_states = _residual(encoder_hidden_states, c_gate_mlp, context_ff_output)
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states


class FusedFluxSingleTransformerBlock(FluxSingleTransformerBlock):
    """``FluxSingleTransformerBlock`` whose norm, modulation and residual run fused."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        shift, scale, gate = self.norm.linear(self.norm.silu(temb)).chunk(3, dim=1)
        norm_hidden_states = layer_norm_modulate(
            hidden_states, scale, shift, eps=self.norm.norm.eps
        )
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **(joint_attention_kwargs or {}),
        )
        projected = self.proj_out(torch.cat([attn_output, mlp_hidden_states], dim=2))
        hidden_states = _residual(hidden_states, gate, projected)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]


def _check_layer_norm(norm: Any, where: str) -> None:
    if type(norm) not in (nn.LayerNorm, FP32LayerNorm) or norm.weight is not None:
        raise ValueError(f"fused FLUX blocks need an affine-free LayerNorm at {where}")


def _check_attention(attn: FluxAttention, *, joint: bool) -> None:
    norms = [attn.norm_q, attn.norm_k]
    if joint:
        norms += [attn.norm_added_q, attn.norm_added_k]
        if attn.norm_q.eps != attn.norm_added_q.eps or attn.norm_k.eps != attn.norm_added_k.eps:
            raise ValueError("fused FLUX blocks need one q and one k RMSNorm eps per stream pair")
    if not all(
        type(norm) is nn.RMSNorm and norm.weight is not None and norm.eps is not None
        for norm in norms
    ):
        raise ValueError("fused FLUX blocks need weighted q/k RMSNorms")
    if type(attn.processor) is not FluxAttnProcessor:
        raise ValueError(
            f"fused FLUX blocks replace FluxAttnProcessor, found {type(attn.processor).__name__}"
        )


def _prepare_double(block: FluxTransformerBlock) -> None:
    for name in ("norm1", "norm1_context"):
        norm = getattr(block, name)
        if type(norm) is not AdaLayerNormZero or norm.emb is not None:
            raise ValueError(
                f"fused FLUX blocks need an embedding-free AdaLayerNormZero at {name}"
            )
        _check_layer_norm(norm.norm, f"{name}.norm")
    _check_layer_norm(block.norm2, "norm2")
    _check_layer_norm(block.norm2_context, "norm2_context")
    _check_attention(block.attn, joint=True)
    swap_processor(block.attn, FusedFluxAttnProcessor)


def _prepare_single(block: FluxSingleTransformerBlock) -> None:
    if type(block.norm) is not AdaLayerNormZeroSingle:
        raise ValueError("fused FLUX single blocks need an AdaLayerNormZeroSingle")
    _check_layer_norm(block.norm.norm, "norm.norm")
    _check_attention(block.attn, joint=False)
    swap_processor(block.attn, FusedFluxAttnProcessor)


def install_fused_blocks(root: nn.Module) -> int:
    """Swap every FLUX double- and single-stream block under ``root``; return how many are fused."""

    return swap_block_class(
        root, FluxTransformerBlock, FusedFluxTransformerBlock, prepare=_prepare_double
    ) + swap_block_class(
        root, FluxSingleTransformerBlock, FusedFluxSingleTransformerBlock, prepare=_prepare_single
    )


__all__ = [
    "FusedFluxAttnProcessor",
    "FusedFluxSingleTransformerBlock",
    "FusedFluxTransformerBlock",
    "install_fused_blocks",
]
