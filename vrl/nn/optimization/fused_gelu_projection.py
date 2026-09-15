"""Feed-forward up-projection and its tanh-GELU as one GEMM for a policy core.

Every DiT feed-forward on the workload list (SD3.5, Wan, Flux, Qwen-Image)
starts with diffusers' ``GELU`` module: ``proj(x)`` then
``F.gelu(..., approximate="tanh")``. The GEMM already folds its bias add into the
cuBLASLt epilogue; the activation then runs as a separate bandwidth-bound pass
over the ``[tokens, 4 * dim]`` intermediate, the widest activation in the block
(one read plus one write of it per block). ``torch._addmm_activation`` selects
the epilogue that applies the tanh-GELU in the same pass, so the intermediate
leaves the GEMM already activated and the extra HBM round trip disappears. It is
one aten op, so inductor keeps it opaque and the fusion survives a compiled
rollout too.

Scope: rollout only. The fused op has no autograd formula, so the replay forward
(and any forward that runs with gradients enabled) keeps the reference kernel;
the module falls back to it whenever grad mode is on, the input is not on CUDA,
the projection is not a plain biased ``nn.Linear`` (LoRA-wrapped or quantized),
or the dtypes disagree. The rollout/replay difference is a tanh-implementation
one (the epilogue's tanh is within ~1e-5 of ``F.gelu``'s in fp32, below one
bf16 ulp), the same class as a compiled rollout against an eager replay, not a
change of the math. Exact-GELU feed-forwards (Cosmos) have no epilogue
equivalent and are left alone.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

_HAS_EPILOGUE_OP = hasattr(torch, "_addmm_activation")
_EPILOGUE_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


class FusedGELUProjection(nn.Module):
    """Drop-in for a diffusers ``GELU``: same ``proj`` submodule, one launch."""

    def __init__(self, proj: nn.Module, approximate: str) -> None:
        super().__init__()
        # The ORIGINAL projection module, so state_dict names, optimizer
        # membership and weight-sync ``load_state_dict`` targets are unchanged.
        self.proj = proj
        self.approximate = approximate

    @classmethod
    def from_module(cls, module: nn.Module) -> FusedGELUProjection:
        return cls(module.proj, module.approximate)

    def _epilogue_applies(self, hidden_states: torch.Tensor) -> bool:
        proj = self.proj
        # ``type`` rather than ``isinstance``: a LoRA or quantized wrapper
        # carries extra math the epilogue would skip.
        if type(proj) is not nn.Linear or proj.bias is None:
            return False
        return (
            _HAS_EPILOGUE_OP
            and self.approximate == "tanh"
            and hidden_states.is_cuda
            and not torch.is_grad_enabled()
            and hidden_states.dtype in _EPILOGUE_DTYPES
            and hidden_states.dtype == proj.weight.dtype == proj.bias.dtype
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._epilogue_applies(hidden_states):
            return F.gelu(self.proj(hidden_states), approximate=self.approximate)
        proj = self.proj
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        out = torch._addmm_activation(proj.bias, flat, proj.weight.t(), use_gelu=True)
        return out.view(*hidden_states.shape[:-1], out.shape[-1])


def _diffusers_gelu_cls() -> type[nn.Module] | None:
    try:
        from diffusers.models.activations import GELU
    except ImportError:  # pragma: no cover - diffusers is a hard dependency of every DiT family
        return None
    return GELU


def fuse_gelu_projections(root: nn.Module) -> int:
    """Replace every fusable diffusers ``GELU`` under ``root`` in place; return the count.

    Fusable means tanh-approximate with a biased projection: that is the only
    form the GEMM epilogue computes. Exact-GELU or bias-free modules stay, so
    zero is a valid result for a family such as Cosmos.
    """

    target = _diffusers_gelu_cls()
    if target is None:
        return 0
    # Materialize first: replacing while iterating ``named_modules`` would
    # mutate the tree under the walk.
    matches = [
        (name, module)
        for name, module in root.named_modules()
        if type(module) is target
        and module.approximate == "tanh"
        and getattr(module.proj, "bias", None) is not None
    ]
    for name, module in matches:
        parent_name, _, attr = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        setattr(parent, attr, FusedGELUProjection.from_module(module))
    return len(matches)
