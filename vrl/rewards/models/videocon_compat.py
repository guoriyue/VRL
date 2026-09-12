"""Narrow compatibility for VideoCon's pinned legacy Transformers imports."""

# Adapted from Hugging Face Transformers pytorch_utils.py (v4.57.6).
# Copyright 2020 The HuggingFace Inc. team. Licensed under Apache-2.0.

import torch


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
