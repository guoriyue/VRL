"""The LoRA delta lands in the base output in place, with no fp32 round trips.

Every policy core on the workload list trains through peft's ``lora.Linear``.
With an fp32 adapter over a bf16 base (peft's default upcast for Cosmos, SD3.5,
Flux and Qwen-Image; ``model.lora.parameter_dtype: float32`` for Wan) its
forward is, per LoRA site::

    result = base(x)                          # bf16 [tokens, out]
    x32 = x.to(fp32)                          # full-width cast of the input
    delta = lora_B(lora_A(x32)) * scaling     # fp32 [tokens, out], then a second
                                              #   fp32 pass just for ``scaling``
    result = (result + delta).to(bf16)        # a THIRD fp32 [tokens, out] for the
                                              #   promoted sum, a fourth pass back

Nothing here is compute: the rank is 4-32, so the site is bound by the
``[tokens, out]`` fp32 intermediates it materializes, and an eager DiT with six
to ten LoRA sites per block pays them every step. The fused forward keeps
peft's math but removes the intermediates:

* ``scaling`` is applied to the rank-wide ``lora_A`` output (``[tokens, rank]``)
  when it is a power of two, where the fold is bit-exact (scaling by ``2**k``
  commutes exactly with every product and partial sum of the ``lora_B`` GEMM);
  any other scaling multiplies the delta in place, the same kernel peft runs.
* the delta is added into ``result`` in place: the add kernel promotes both
  operands to fp32, sums once and stores in ``result``'s dtype -- the same two
  roundings as ``(result + delta).to(bf16)``, without the fp32 sum and the
  cast pass.

Numerics: bit-identical to peft's forward for one active adapter (the only
case taken; several active adapters accumulate in fp32 before the final
rounding, which the in-place store would split, so they keep the reference
path). Scope: rollout only. The fast path is taken with gradients disabled;
under grad, with adapters disabled or merged, with a mixed-batch
``adapter_names`` call or a peft LoRA variant, the module IS peft's forward.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import torch
from torch import nn


def _lora_linear_cls() -> type[nn.Module] | None:
    try:
        from peft.tuners.lora.layer import Linear
    except ImportError:  # pragma: no cover - peft is a hard dependency of every LoRA family
        return None
    return Linear


def _is_power_of_two(scaling: Any) -> bool:
    return isinstance(scaling, (int, float)) and scaling > 0 and math.frexp(scaling)[0] == 0.5


@functools.cache
def _fused_lora_linear_cls() -> type[nn.Module]:
    from peft.tuners.lora.layer import VARIANT_KWARG_KEYS

    base = _lora_linear_cls()
    assert base is not None

    class FusedLoraLinear(base):  # type: ignore[misc,valid-type]
        """peft ``lora.Linear`` whose no-grad forward skips the fp32 round trips.

        Installed by re-classing the peft instance, not by rebuilding it: the
        adapter registry (``lora_A``/``lora_B`` ModuleDicts, dropout, scaling,
        variants) and the object identity that ``set_adapter``,
        ``disable_adapter`` and weight-sync ``load_state_dict`` rely on stay
        exactly as peft left them.
        """

        def _single_vanilla_adapter(self) -> str | None:
            active = [name for name in self.active_adapters if name in self.lora_A]
            if len(active) != 1 or active[0] in self.lora_variant:
                return None
            return active[0]

        def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            if (
                torch.is_grad_enabled()
                or self.disable_adapters
                or self.merged
                or kwargs.get("adapter_names") is not None
            ):
                return super().forward(x, *args, **kwargs)
            adapter = self._single_vanilla_adapter()
            if adapter is None:
                return super().forward(x, *args, **kwargs)
            # peft strips its own routing kwargs before calling the base layer.
            kwargs.pop("adapter_names", None)
            for key in VARIANT_KWARG_KEYS:
                kwargs.pop(key, None)
            lora_A = self.lora_A[adapter]
            lora_B = self.lora_B[adapter]
            scaling = self.scaling[adapter]

            result = self.base_layer(x, *args, **kwargs)
            x = self._cast_input_dtype(x, lora_A.weight.dtype)
            hidden = lora_A(self.lora_dropout[adapter](x))
            if scaling == 1:
                delta = lora_B(hidden)
            elif _is_power_of_two(scaling):
                delta = lora_B(hidden * scaling)
            else:
                delta = lora_B(hidden)
                delta.mul_(scaling)
            result.add_(delta)
            return result

    return FusedLoraLinear


def fuse_lora_branches(root: nn.Module) -> int:
    """Re-class every peft ``lora.Linear`` under ``root`` in place; return the count.

    Zero is a valid result: a policy trained without LoRA (full fine-tune) has
    no adapter branch to fuse.
    """

    target = _lora_linear_cls()
    if target is None:
        return 0
    fused = _fused_lora_linear_cls()
    matches = [module for module in root.modules() if type(module) is target]
    for module in matches:
        module.__class__ = fused
    return len(matches)
