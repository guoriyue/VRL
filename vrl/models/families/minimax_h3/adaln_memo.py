"""MiniMax-H3 AdaLN projections of a frozen base, answered from a per-timestep table.

Every block projects the shared timestep embedding to its six modulation
parameters through one ``Linear(time_embed_dim, 18 * hidden_size)``: 260M
weights a block, 13.0B of the checkpoint's 33.1B. Its input is one row per
distinct noise level of the forward, so over a run it only ever sees the rows
of the sampling schedule.

:func:`memoize_adaln_projections` replaces each block's ``adaln_proj`` with
:class:`MemoizedAdaLNModulation`. The projection weight stays on the host where
the loader put it; the first forward that brings a new timestep embedding
computes that row on the embedding's device and stores it, and every later
forward is an ``index_select``. A base quantized to NVFP4 then holds 20.1B
weights on the card instead of 33.1B.

The table is not bitwise to the stock module: each row is its own GEMM, so an
entry does not depend on which other timesteps first arrived with it, while the
stock module multiplies the rows of one forward together. The rollout and the
replay both read tables built this way.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3AdaLayerNormModulation,
    MiniMaxH3TransformerBlock,
)
from torch import nn


class _TimestepRows:
    """The distinct timestep embeddings one transformer has seen, in arrival order."""

    def __init__(self) -> None:
        self.keys: torch.Tensor | None = None
        self._last: tuple[torch.Tensor, torch.Tensor] | None = None

    @torch.no_grad()
    def rows(self, temb: torch.Tensor) -> torch.Tensor:
        """The key index of every row of ``temb``, adding the rows not seen before."""

        # Every block of a forward receives the same tensor, and a checkpointed
        # block is recomputed with the tensor it first ran on.
        if self._last is not None and self._last[0] is temb:
            return self._last[1]
        keys = temb.new_empty(0, temb.shape[-1]) if self.keys is None else self.keys
        match = (temb[:, None, :] == keys[None]).all(-1)
        known = match.any(-1)
        # The one device-to-host read of the memo, once per forward: whether
        # this forward brings a timestep the tables do not hold yet.
        if not bool(known.all()):
            keys = torch.cat([keys, temb[~known]])
            match = (temb[:, None, :] == keys[None]).all(-1)
        self.keys = keys
        rows = match.to(torch.uint8).argmax(-1)
        self._last = (temb, rows)
        return rows


class MemoizedAdaLNModulation(nn.Module):
    """``MiniMaxH3AdaLayerNormModulation`` of a frozen block, one stored row per timestep."""

    def __init__(self, source: MiniMaxH3AdaLayerNormModulation, timesteps: _TimestepRows) -> None:
        super().__init__()
        linear = source.linear
        if type(linear) is not nn.Linear or linear.weight.requires_grad:
            raise ValueError(
                "a memoized AdaLN projection needs the frozen plain Linear; "
                f"got {type(linear).__name__} (an adapter on adaln_proj cannot be memoized)"
            )
        self.hidden_size = source.hidden_size
        self._timesteps = timesteps
        # Held outside the module registry: a device move or a park of the
        # transformer leaves the projection on the host.
        self._host_linear = (linear.weight.detach(), linear.bias.detach())
        self.register_buffer(
            "table",
            torch.empty(0, linear.out_features, dtype=linear.weight.dtype),
            persistent=False,
        )

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        rows = self._timesteps.rows(temb)
        keys = self._timesteps.keys
        if self.table.shape[0] < keys.shape[0]:
            with torch.no_grad():
                weight, bias = (tensor.to(keys.device) for tensor in self._host_linear)
                new_rows = [
                    F.linear(F.silu(key).to(weight.dtype), weight, bias)
                    for key in keys[self.table.shape[0] :].split(1)
                ]
                self.table = torch.cat([self.table.to(keys.device), *new_rows])
        return self.table.index_select(0, rows).view(-1, 6 * self.hidden_size).chunk(6, dim=-1)


def memoize_adaln_projections(root: nn.Module) -> int:
    """Memoize the AdaLN projection of every MiniMax-H3 block under ``root``; return how many."""

    timesteps = _TimestepRows()
    blocks = [module for module in root.modules() if isinstance(module, MiniMaxH3TransformerBlock)]
    for block in blocks:
        block.adaln_proj = MemoizedAdaLNModulation(block.adaln_proj, timesteps)
    return len(blocks)


__all__ = ["MemoizedAdaLNModulation", "memoize_adaln_projections"]
