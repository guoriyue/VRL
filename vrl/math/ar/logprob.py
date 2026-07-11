"""AR sampling and replay logits math shared by model-family runners."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Top-k / nucleus filtering, ported from upstream ``generate.py``.

    Kept numerically identical to FoundationVision/LlamaGen (MIT) so sampling
    with the upstream demo knobs (top_k=1000) reproduces upstream behavior.
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def gather_categorical_log_probs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    temperature: float = 1.0,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Return log-probs for selected tokens without materializing full log-softmax.

    Janus replay logits can be ``[B, L, V]`` with a large image-token vocab.
    A full fp32 ``log_softmax`` creates another tensor of that same size and
    can OOM when trainer and rollout worker are colocated on one GPU. Chunking
    over flattened token positions keeps peak memory bounded while preserving
    fp32 normalization.

    ``temperature`` scales logits before normalization. The AR policy contract
    scores the temperature-scaled conditional (rollout samplers divide by the
    sampling temperature before ``log_softmax``), so replay MUST apply the
    same recorded temperature or the GRPO old/new ratio is biased whenever
    temperature != 1. Greedy decoding is a separate policy mode, not a
    near-zero-temperature categorical distribution, so temperature must be
    finite and strictly positive.
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

    temp = require_positive_temperature(temperature)
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)
    flat_ids = token_ids.to(device=logits.device, dtype=torch.long).reshape(-1)
    pieces: list[torch.Tensor] = []
    for start in range(0, flat_ids.numel(), chunk_size):
        end = min(start + chunk_size, flat_ids.numel())
        chunk = flat_logits[start:end].float() / temp
        ids = flat_ids[start:end]
        selected = chunk.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        pieces.append(selected - torch.logsumexp(chunk, dim=-1))
    return torch.cat(pieces, dim=0).reshape(token_ids.shape)


def require_positive_temperature(value: float) -> float:
    """Return a finite positive categorical-policy temperature."""

    temperature = float(value)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            "categorical policy temperature must be finite and > 0; "
            "use an explicit greedy policy instead of temperature=0",
        )
    return temperature


__all__ = [
    "gather_categorical_log_probs",
    "require_positive_temperature",
    "top_k_top_p_filtering",
]
