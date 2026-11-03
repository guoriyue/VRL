"""Cosmos Predict2 / Predict2.5 transformer blocks over the fused DiT elementwise kernels.

:func:`install_fused_blocks` swaps every ``CosmosTransformerBlock`` of a root
to :class:`FusedCosmosTransformerBlock` and its self-attention processor to
:class:`FusedCosmosAttnProcessor`. Modules, parameters and state-dict names
stay; only the elementwise chains between the projections change
(``vrl.nn.kernels.dit_elementwise``):

- each AdaLN-Zero LayerNorm and its modulation in one pass, the table read
  per sample or, with the per-frame timesteps video-to-world conditions on,
  per frame;
- the gated residual adds;
- the self-attention's per-head q/k RMSNorm and rotate-half RoPE in one pass,
  where the stock path normalizes ``[B, H, S, D]`` transposes and rotates
  through fp32 copies, and the text cross-attention's q/k RMSNorm.

With per-frame timesteps the stock transformer expands every frame's
timestep embedding to each of its tokens, so every block projects the same
row through its three AdaLN MLPs once per token (a 37k-token 480p clip of 24
frames: 6 ms of an 87 ms block). The model swaps to
:class:`FusedCosmosTransformer3DModel`, which hands the blocks one row per
frame instead; the fused kernels read that row for each of the frame's
tokens, and the output norm runs fused the same way.

The AdaLN projections, the image-context cross-attention of a Predict2.5
checkpoint that has one (the 2B has none) and the feed-forward keep the stock
modules. ``model.fused_kernels``
installs the blocks in the rollout and the replay alike, so both sides keep
running one computation. The swap is not bitwise to the stock block: it
rounds once where the stock chains round two or three times.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_cosmos import (
    CosmosAdaLayerNorm,
    CosmosAdaLayerNormZero,
    CosmosAttnProcessor2_0,
    CosmosTransformer3DModel,
    CosmosTransformerBlock,
    transforms,
)

from vrl.nn.kernels.batch_invariant import per_sample_attention
from vrl.nn.kernels.block_swap import swap_block_class, swap_processor
from vrl.nn.kernels.dit_elementwise import (
    gated_residual,
    joint_rms_norm_rope,
    layer_norm_modulate,
    rms_norm_rotate_half,
)


class FusedCosmosAttnProcessor(CosmosAttnProcessor2_0):
    """The stock attention with the q/k RMSNorm (and, in self-attention, the RoPE) fused."""

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states

        def normalized(projection: Any, states: torch.Tensor, norm: RMSNorm) -> torch.Tensor:
            projected = projection(states).unflatten(2, (attn.heads, -1))
            if image_rotary_emb is None:
                return joint_rms_norm_rope([projected], [norm.weight], [None], eps=norm.eps)
            cos, sin = image_rotary_emb
            return rms_norm_rotate_half(projected, norm.weight, cos, sin, eps=norm.eps)

        query = normalized(attn.to_q, hidden_states, attn.norm_q)
        hidden_states = per_sample_attention(
            dispatch_attention_fn,
            query,
            normalized(attn.to_k, context, attn.norm_k),
            attn.to_v(context).unflatten(2, (attn.heads, -1)),
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


class FusedCosmosTransformer3DModel(CosmosTransformer3DModel):
    """``CosmosTransformer3DModel`` that hands its fused blocks one timestep row per frame.

    The forward is the pinned diffusers' (0.41.0.dev0), but for step 4
    (per-frame timestep embeddings stay ``[B, T, C]`` instead of expanding to
    ``[B, THW, C]``; tokens are frame-major, so frame ``t`` is the ``t``-th
    run of ``HW`` consecutive tokens) and step 8 (the output norm reads them
    the same way).
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        block_controlnet_hidden_states: list[torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        fps: int | None = None,
        condition_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> tuple[torch.Tensor] | Transformer2DModelOutput:
        batch_size, _, num_frames, height, width = hidden_states.shape

        # 1. Concatenate padding mask if needed & prepare attention mask
        if condition_mask is not None:
            hidden_states = torch.cat([hidden_states, condition_mask], dim=1)

        if self.config.concat_padding_mask:
            padding_mask_resized = transforms.functional.resize(
                padding_mask,
                list(hidden_states.shape[-2:]),
                interpolation=transforms.InterpolationMode.NEAREST,
            )
            hidden_states = torch.cat(
                [
                    hidden_states,
                    padding_mask_resized.unsqueeze(2).repeat(batch_size, 1, num_frames, 1, 1),
                ],
                dim=1,
            )

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S]

        # 2. Generate positional embeddings
        image_rotary_emb = self.rope(hidden_states, fps=fps)
        extra_pos_emb = (
            self.learnable_pos_embed(hidden_states) if self.config.extra_pos_embed_type else None
        )

        # 3. Patchify input
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states.flatten(1, 3)  # [B, T, H, W, C] -> [B, THW, C]

        # 4. Timestep embeddings: per frame, [BT, C] -> [B, T, C] rows the blocks
        # read for each of a frame's HW tokens.
        if timestep.ndim == 1:
            temb, embedded_timestep = self.time_embed(hidden_states, timestep)
        elif timestep.ndim == 5:
            if timestep.shape != (batch_size, 1, num_frames, 1, 1):
                raise ValueError(
                    f"Expected timestep to have shape [B, 1, T, 1, 1], but got {timestep.shape}"
                )
            temb, embedded_timestep = self.time_embed(hidden_states, timestep.flatten())
            temb, embedded_timestep = (
                x.view(batch_size, post_patch_num_frames, -1) for x in (temb, embedded_timestep)
            )
        else:
            raise ValueError(
                f"Expected timestep to have shape [B, 1, T, 1, 1] or [T], but got {timestep.shape}"
            )

        # 5. Process encoder hidden states
        text_context, img_context = (
            encoder_hidden_states
            if isinstance(encoder_hidden_states, tuple)
            else (encoder_hidden_states, None)
        )
        if self.config.use_crossattn_projection:
            text_context = self.crossattn_proj(text_context)

        if img_context is not None and self.config.img_context_dim_in:
            img_context = self.img_context_proj(img_context)

        processed_encoder_hidden_states = (
            (text_context, img_context)
            if isinstance(encoder_hidden_states, tuple)
            else text_context
        )

        # 6. Build controlnet block index map
        controlnet_block_index_map = {}
        if block_controlnet_hidden_states is not None:
            n_blocks = len(self.transformer_blocks)
            controlnet_block_index_map = {
                block_idx: block_controlnet_hidden_states[idx]
                for idx, block_idx in list(
                    enumerate(range(0, n_blocks, self.config.controlnet_block_every_n))
                )
            }

        # 7. Transformer blocks
        for block_idx, block in enumerate(self.transformer_blocks):
            controlnet_residual = controlnet_block_index_map.get(block_idx)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    processed_encoder_hidden_states,
                    embedded_timestep,
                    temb,
                    image_rotary_emb,
                    extra_pos_emb,
                    attention_mask,
                    controlnet_residual,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    processed_encoder_hidden_states,
                    embedded_timestep,
                    temb,
                    image_rotary_emb,
                    extra_pos_emb,
                    attention_mask,
                    controlnet_residual,
                )

        # 8. Output norm (fused, per-frame rows as the blocks) & projection & unpatchify
        norm = self.norm_out
        modulation = norm.linear_2(norm.linear_1(norm.activation(embedded_timestep)))
        if temb is not None:
            modulation = modulation + temb[..., : 2 * norm.embedding_dim]
        shift, scale = modulation.chunk(2, dim=-1)
        hidden_states = layer_norm_modulate(hidden_states, scale, shift, eps=norm.norm.eps)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.unflatten(2, (p_h, p_w, p_t, -1))
        hidden_states = hidden_states.unflatten(
            1, (post_patch_num_frames, post_patch_height, post_patch_width)
        )
        # The stock unpatchify permutation, which is not the inverse of the patching.
        hidden_states = hidden_states.permute(0, 7, 1, 6, 2, 4, 3, 5)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)


def _fuses_attention(attn: Any) -> bool:
    """Whether ``attn`` is a stock attention with the q/k RMSNorms the fused processor fuses."""

    return (
        type(attn.processor) is CosmosAttnProcessor2_0
        and all(
            isinstance(norm, RMSNorm) and norm.weight is not None
            for norm in (attn.norm_q, attn.norm_k)
        )
        and attn.to_k.out_features == attn.to_q.out_features
    )


def _prepare(block: CosmosTransformerBlock) -> None:
    for name in ("norm1", "norm2", "norm3"):
        norm = getattr(block, name)
        if type(norm) is not CosmosAdaLayerNormZero or norm.norm.weight is not None:
            raise ValueError(f"fused Cosmos blocks need an affine-free AdaLN-Zero at {name}")
    if not _fuses_attention(block.attn1):
        raise ValueError(
            "fused Cosmos blocks need the stock self-attention with weighted per-head q/k "
            f"RMSNorms and as many key heads as query heads, found {block.attn1.processor}"
        )
    swap_processor(block.attn1, FusedCosmosAttnProcessor)
    # Predict2.5's image-context cross-attention (CosmosAttnProcessor2_5) stays stock.
    if _fuses_attention(block.attn2):
        swap_processor(block.attn2, FusedCosmosAttnProcessor)


def _check_output_norm(model: CosmosTransformer3DModel) -> None:
    """Refuse a model whose output norm is not the stock one the fused forward computes."""

    norm = model.norm_out
    if type(norm) is not CosmosAdaLayerNorm or norm.norm.weight is not None:
        raise ValueError("the fused Cosmos model needs the affine-free CosmosAdaLayerNorm output")


def install_fused_blocks(root: nn.Module) -> int:
    """Swap every Cosmos block and transformer under ``root`` to the fused ones.

    Returns how many blocks are fused. The transformers swap after their
    blocks, since only fused blocks read the per-frame rows they pass.
    """

    count = swap_block_class(
        root, CosmosTransformerBlock, FusedCosmosTransformerBlock, prepare=_prepare
    )
    swap_block_class(
        root, CosmosTransformer3DModel, FusedCosmosTransformer3DModel, prepare=_check_output_norm
    )
    return count


__all__ = [
    "FusedCosmosAttnProcessor",
    "FusedCosmosTransformer3DModel",
    "FusedCosmosTransformerBlock",
    "install_fused_blocks",
]
