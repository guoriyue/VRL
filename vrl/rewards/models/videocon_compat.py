"""Narrow compatibility for VideoCon's pinned legacy Transformers imports."""

# Adapted from Hugging Face Transformers pytorch_utils.py (v4.57.6).
# Copyright 2020 The HuggingFace Inc. team. Licensed under Apache-2.0.

import torch


def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
    """Preserve the legacy Transformers head-mask broadcasting contract."""
    if head_mask is None:
        return [None] * num_hidden_layers
    if head_mask.dim() == 1:
        head_mask = head_mask[None, None, :, None, None].expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask[:, None, :, None, None]
    if head_mask.dim() != 5:
        raise ValueError(f"Expected a five-dimensional head mask, got {head_mask.dim()}")
    head_mask = head_mask.to(dtype=self.dtype)
    return head_mask.unsqueeze(-1) if is_attention_chunked else head_mask


def prepare_videocon_model_class(model_class) -> None:
    """Install missing legacy methods on the vendor class, never global HF bases."""
    if not hasattr(model_class, "get_head_mask"):
        model_class.get_head_mask = _get_head_mask


def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(h < head for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index


def prepare_videocon_imports() -> None:
    import transformers.pytorch_utils as utils

    # The vendor imports this symbol even when head pruning is not requested.
    if not hasattr(utils, "find_pruneable_heads_and_indices"):
        utils.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices
