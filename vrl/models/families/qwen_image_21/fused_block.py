"""Qwen-Image-2.1 transformer blocks over the fused DiT elementwise kernels.

:func:`install_fused_blocks` swaps every ``QwenImage21TransformerBlock`` of a
root to :class:`FusedQwenImage21TransformerBlock` and its attention processor
to :class:`FusedQwenImage21AttnProcessor`. Modules, parameters and state-dict
names stay; only the elementwise chains between the projections change
(``vrl.nn.kernels.dit_elementwise``):

- LayerNorm and the scale modulation in one pass, with the ``causal_condition``
  split (condition tokens modulate from the trailing ``t = 0`` row, target
  tokens from their own sample's) read as a per-token table row inside the
  kernel instead of a ``torch.where`` over the whole sequence;
- the ``tanh``-gated residual adds;
- the q/k RMSNorm and the complex RoPE in one pass;
- the SwiGLU product ``silu(gate) * up``.

``model.fused_kernels`` installs them in the rollout and the replay alike, so
both sides keep running one computation. The swap is not bitwise to the stock
block: it rounds once where the stock chains round two or three times.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21AttnProcessor,
    QwenImage21KVLayerCache,
    QwenImage21SwiGLUFeedForward,
    QwenImage21TransformerBlock,
)

from vrl.nn.kernels.batch_invariant import per_sample_attention
from vrl.nn.kernels.block_swap import swap_block_class, swap_processor
from vrl.nn.kernels.dit_elementwise import (
    gated_residual,
    joint_rms_norm_rope,
    layer_norm_modulate,
    silu_mul,
)


class FusedQwenImage21AttnProcessor(QwenImage21AttnProcessor):
    """The stock segmented attention with the q/k RMSNorm and RoPE fused.

    Everything after the q/k preparation (the KV cache bookkeeping, the
    per-segment block-causal prefill, the cached decode) is the stock
    processor's, copied from diffusers 0.40 because the stock ``__call__``
    reaches its preparation through a module-level function.
    """

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        attention_mask: Any | None = None,
        rotary_emb: torch.Tensor | None = None,
        layer_cache: QwenImage21KVLayerCache | None = None,
        kv_cache_mode: str | None = None,
        cache_write_slice: slice | None = None,
        segments: list[tuple[int, int, bool]] | None = None,
        key_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if rotary_emb is None:
            return super().__call__(
                attn,
                hidden_states,
                attention_mask,
                rotary_emb,
                layer_cache,
                kv_cache_mode,
                cache_write_slice,
                segments,
                key_valid,
            )
        query = joint_rms_norm_rope(
            [attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))],
            [attn.norm_q.weight],
            [rotary_emb],
            eps=attn.norm_q.eps,
        )
        key = joint_rms_norm_rope(
            [attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))],
            [attn.norm_k.weight],
            [rotary_emb],
            eps=attn.norm_k.eps,
        )
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))

        if layer_cache is not None:
            if kv_cache_mode == "extract" and cache_write_slice is not None:
                layer_cache.store(
                    key[:, cache_write_slice].clone(), value[:, cache_write_slice].clone()
                )
            elif kv_cache_mode == "cached":
                cached_k, cached_v = layer_cache.get()
                key = torch.cat([cached_k, key], dim=1)
                value = torch.cat([cached_v, value], dim=1)
        seq_len_q = query.shape[1]

        if segments is None:
            hidden_states = per_sample_attention(
                dispatch_attention_fn,
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        else:
            prefix_len = segments[-1][1] if segments else 0
            outputs = []
            for start, end, is_text in segments:
                seg_mask = None
                if is_text:
                    seg_len = end - start
                    seg_mask = torch.cat(
                        [
                            torch.ones(seg_len, start, dtype=torch.bool, device=query.device),
                            torch.tril(
                                torch.ones(seg_len, seg_len, dtype=torch.bool, device=query.device)
                            ),
                        ],
                        dim=1,
                    )[None, None]
                if key_valid is not None:
                    seg_key_valid = key_valid[:, None, None, :end]
                    seg_mask = seg_key_valid if seg_mask is None else (seg_mask & seg_key_valid)
                outputs.append(
                    per_sample_attention(
                        dispatch_attention_fn,
                        query[:, start:end],
                        key[:, :end],
                        value[:, :end],
                        attn_mask=seg_mask,
                        dropout_p=0.0,
                        backend=None,
                        parallel_config=self._parallel_config,
                    )
                )
            outputs.append(
                per_sample_attention(
                    dispatch_attention_fn,
                    query[:, prefix_len:],
                    key,
                    value,
                    attn_mask=None if key_valid is None else key_valid[:, None, None, :],
                    dropout_p=0.0,
                    backend=None,
                    parallel_config=self._parallel_config,
                )
            )
            hidden_states = torch.cat(outputs, dim=1)
        hidden_states = hidden_states[:, :seq_len_q]
        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


def _modulation_rows(target_token_mask: torch.Tensor | None, batch: int) -> torch.Tensor | None:
    """Table rows of ``_select_modulation_rows``: own sample on target tokens, ``t = 0`` else."""

    if target_token_mask is None:
        return None
    own = torch.arange(batch, device=target_token_mask.device, dtype=torch.int32)[:, None]
    return torch.where(target_token_mask.view(1, -1), own, batch)


class FusedQwenImage21TransformerBlock(QwenImage21TransformerBlock):
    """``QwenImage21TransformerBlock`` whose norms, modulation, residuals and SwiGLU run fused."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        modulation: torch.Tensor,
        rotary_emb: torch.Tensor | None = None,
        attention_mask: Any | None = None,
        target_token_mask: torch.Tensor | None = None,
        layer_cache: QwenImage21KVLayerCache | None = None,
        kv_cache_mode: str | None = None,
        cache_write_slice: slice | None = None,
        segments: list[tuple[int, int, bool]] | None = None,
        key_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mod1, mod2 = modulation.chunk(2, dim=-1)
        scale1, gate1 = mod1.chunk(2, dim=-1)
        scale2, gate2 = mod2.chunk(2, dim=-1)
        rows = _modulation_rows(target_token_mask, hidden_states.shape[0])

        attn_output = self.attn(
            hidden_states=layer_norm_modulate(
                hidden_states, scale1, None, eps=self.img_norm1.eps, rows=rows
            ),
            attention_mask=attention_mask,
            rotary_emb=rotary_emb,
            layer_cache=layer_cache,
            kv_cache_mode=kv_cache_mode,
            cache_write_slice=cache_write_slice,
            segments=segments,
            key_valid=key_valid,
        )
        hidden_states = gated_residual(hidden_states, gate1, attn_output, rows=rows, tanh=True)

        mlp = self.img_mlp
        modulated = layer_norm_modulate(
            hidden_states, scale2, None, eps=self.img_norm2.eps, rows=rows
        )
        mlp_output = mlp.out(silu_mul(mlp.gate_layer(modulated), mlp.proj(modulated)))
        hidden_states = gated_residual(hidden_states, gate2, mlp_output, rows=rows, tanh=True)

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states


def _check_fusable(block: QwenImage21TransformerBlock) -> None:
    """Refuse a block whose norms, attention or feed-forward the fused kernels do not compute."""

    for name in ("img_norm1", "img_norm2"):
        norm = getattr(block, name)
        if not isinstance(norm, nn.LayerNorm) or norm.weight is not None:
            raise ValueError(
                f"fused Qwen-Image-2.1 blocks need an affine-free LayerNorm at {name}"
            )
    attn = block.attn
    if not all(
        isinstance(norm, RMSNorm) and norm.weight is not None and norm.bias is None
        for norm in (attn.norm_q, attn.norm_k)
    ):
        raise ValueError("fused Qwen-Image-2.1 blocks need weighted, bias-free RMSNorm q/k norms")
    if type(attn.processor) is not QwenImage21AttnProcessor:
        raise ValueError(
            f"fused Qwen-Image-2.1 blocks replace QwenImage21AttnProcessor, "
            f"found {type(attn.processor).__name__}"
        )
    mlp = block.img_mlp
    if not isinstance(mlp, QwenImage21SwiGLUFeedForward) or not isinstance(
        mlp.activation_fn, nn.SiLU
    ):
        raise ValueError("fused Qwen-Image-2.1 blocks need the SiLU SwiGLU feed-forward")


def _prepare(block: QwenImage21TransformerBlock) -> None:
    _check_fusable(block)
    swap_processor(block.attn, FusedQwenImage21AttnProcessor)


def install_fused_blocks(root: nn.Module) -> int:
    """Swap every Qwen-Image-2.1 block under ``root`` to the fused one; return how many are fused."""

    return swap_block_class(
        root, QwenImage21TransformerBlock, FusedQwenImage21TransformerBlock, prepare=_prepare
    )


__all__ = [
    "FusedQwenImage21AttnProcessor",
    "FusedQwenImage21TransformerBlock",
    "install_fused_blocks",
]
