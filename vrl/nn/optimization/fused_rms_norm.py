"""One-launch RMSNorm for the diffusers ``RMSNorm`` modules of a policy core.

The diffusers ``RMSNorm.forward`` normalizes in fp32 by hand: upcast, square,
mean, rsqrt, multiply, downcast, multiply by the weight. On a ``[B, H, S, hd]``
q/k tensor that is seven launches and roughly fifteen full-tensor passes, most
of them over fp32 intermediates twice the size of the bf16 activation.
``torch.nn.functional.rms_norm`` runs the same math as one fused kernel with
fp32 accumulation and one bf16 write, so an eager DiT that Q/K-normalizes
every block (Cosmos, SD3.5, Qwen-Image) spends an order of magnitude less
memory traffic per norm. Families whose blocks already use ``torch.nn.RMSNorm``
(Wan, Flux) are on that kernel today and are left alone.

Numerics: the fused kernel rounds once (``x * rstd * w`` in fp32 -> bf16) where
the hand-written path rounds twice (``(x * rstd) -> bf16`` then ``* w``), so
half-precision outputs may differ by one ulp. Rollout and replay both apply the
swap when ``model.fused_rms_norm`` is set, so the two roles keep running one
kernel and the log-prob ratio sees no new drift. Mixed dtypes (an fp32
activation through a bf16 weight) keep the two-rounding path, because that is
the only way to reproduce diffusers' output dtype (the weight's).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FusedRMSNorm(nn.Module):
    """Drop-in for a diffusers ``RMSNorm``: same parameters, one kernel."""

    def __init__(
        self,
        dim: torch.Size,
        eps: float,
        weight: nn.Parameter | None,
        bias: nn.Parameter | None,
    ) -> None:
        super().__init__()
        self.dim = torch.Size(dim)
        self.eps = eps
        # The ORIGINAL parameter objects, so state_dict names, optimizer
        # membership and weight-sync ``load_state_dict`` targets are unchanged.
        self.weight = weight
        self.bias = bias

    @classmethod
    def from_module(cls, module: nn.Module) -> FusedRMSNorm:
        return cls(module.dim, module.eps, module.weight, module.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if weight is None or weight.dtype == hidden_states.dtype:
            out = F.rms_norm(hidden_states, self.dim, weight, self.eps)
        else:
            variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
            out = hidden_states * torch.rsqrt(variance + self.eps)
            out = out.to(weight.dtype) * weight
        if self.bias is not None:
            out = out + self.bias
        return out


def _diffusers_rms_norm_cls() -> type[nn.Module] | None:
    try:
        from diffusers.models.normalization import RMSNorm
    except ImportError:  # pragma: no cover - diffusers is a hard dependency of every DiT family
        return None
    return RMSNorm


def fuse_rms_norms(root: nn.Module) -> int:
    """Replace every diffusers ``RMSNorm`` under ``root`` in place; return the count.

    Zero is a valid result: a policy whose norms are already ``torch.nn.RMSNorm``
    has nothing to swap and already runs the fused kernel.
    """

    target = _diffusers_rms_norm_cls()
    if target is None:
        return 0
    # Materialize first: replacing while iterating ``named_modules`` would
    # mutate the tree under the walk.
    matches = [(name, module) for name, module in root.named_modules() if type(module) is target]
    for name, module in matches:
        parent_name, _, attr = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        setattr(parent, attr, FusedRMSNorm.from_module(module))
    return len(matches)
