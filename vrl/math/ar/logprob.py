"""Memory-bounded categorical log-prob helpers for AR replay."""

from __future__ import annotations

import torch


def gather_categorical_log_probs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Return log-probs for selected tokens without materializing full log-softmax.

    Janus replay logits can be ``[B, L, V]`` with a large image-token vocab.
    A full fp32 ``log_softmax`` creates another tensor of that same size and
    can OOM when trainer and rollout worker are colocated on one GPU. Chunking
    over flattened token positions keeps peak memory bounded while preserving
    fp32 normalization.
    """

    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if logits.ndim != token_ids.ndim + 1:
        raise ValueError(
            "logits rank must equal token_ids rank + 1; "
            f"got logits={tuple(logits.shape)} token_ids={tuple(token_ids.shape)}",
        )
    if tuple(logits.shape[:-1]) != tuple(token_ids.shape):
        raise ValueError(
            "logits leading shape must match token_ids shape; "
            f"got logits={tuple(logits.shape)} token_ids={tuple(token_ids.shape)}",
        )

    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)
    flat_ids = token_ids.to(device=logits.device, dtype=torch.long).reshape(-1)
    pieces: list[torch.Tensor] = []
    for start in range(0, flat_ids.numel(), chunk_size):
        end = min(start + chunk_size, flat_ids.numel())
        chunk = flat_logits[start:end].float()
        ids = flat_ids[start:end]
        selected = chunk.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        pieces.append(selected - torch.logsumexp(chunk, dim=-1))
    return torch.cat(pieces, dim=0).reshape(token_ids.shape)


__all__ = ["gather_categorical_log_probs"]
