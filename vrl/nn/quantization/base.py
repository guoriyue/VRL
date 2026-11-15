"""Shared ownership contract for rollout-side quantized linear schemes.

Every rollout quantization scheme subclasses :class:`QuantizedLinear`. Runtime
guards identify the exact requested scheme through ``quantization_scheme`` while
master cleanup and device moves remain scheme-neutral.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from vrl.nn.quantization.targeting import DEFAULT_EXCLUDE


class _FrozenGemm(torch.autograd.Function):
    """Differentiate a frozen quantized linear with respect to its input.

    The forward is the scheme's own low-precision GEMM, so a replay under
    autograd computes exactly what the no-grad rollout computed. The backward is
    the Jacobian of that GEMM in the quantized activation, ``grad @ Q(W)``, with
    the activation's own rounding passed straight through. It runs in the
    activation dtype on the dequantized cache: the low-precision kernels take
    their block scales along the forward contraction, which is the wrong axis
    for the transposed product.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, module: QuantizedLinear) -> torch.Tensor:
        ctx.module = module
        ctx.input_dtype = x.dtype
        return module.quantized_forward(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        weight = ctx.module.dequantized_weight()
        grad = grad_output.reshape(-1, weight.shape[0]).to(weight.dtype) @ weight
        return grad.reshape(*grad_output.shape[:-1], weight.shape[1]).to(ctx.input_dtype), None


class QuantizedLinear(nn.Module):
    """Base for a low-precision drop-in ``nn.Linear`` replacement.

    Subclasses declare their derived caches through ``cache_buffer_names``.
    ``Module.to(dtype=...)`` must never cast packed FP4/FP8 caches into the model's
    base dtype: when a source master exists, caches are rebuilt from the moved
    master; master-free rollout caches move as raw bytes and preserve their exact
    format.

    They also own their module-tree swap policy: ``can_replace`` rejects shapes
    the scheme's kernels cannot take, and ``swap_linears`` is the one traversal.
    """

    quantization_scheme: str
    cache_buffer_names: tuple[str, ...] = ()

    def _requantize_weight(self) -> None:
        """Rebuild derived low-precision buffers from ``self.weight``."""

        raise NotImplementedError

    def quantized_forward(self, x: torch.Tensor) -> torch.Tensor:
        """The scheme's low-precision GEMM on ``x``, bias included."""

        raise NotImplementedError

    def dequantized_weight(self) -> torch.Tensor:
        """The cached weight the GEMM multiplies by, ``[out, in]`` in bf16."""

        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only a frozen base is differentiated here (LoRA training): the cache
        # is derived from the master at load time and no optimizer step
        # refreshes it, so the weight takes no gradient.
        if torch.is_grad_enabled() and x.requires_grad:
            return _FrozenGemm.apply(x, self)
        return self.quantized_forward(x)

    @classmethod
    def can_replace(cls, linear: nn.Linear) -> bool:
        """Whether this scheme's kernels accept ``linear``'s shape.

        Traversal skips a rejected Linear rather than failing, so a policy with a
        few unsupported shapes still quantizes everything else.
        """

        return True

    @classmethod
    def swap_linears(
        cls,
        root: nn.Module,
        *,
        exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
        min_features: int = 1024,
        **init_kwargs: Any,
    ) -> list[str]:
        """Replace eligible ``nn.Linear`` modules under ``root`` with ``cls`` in place.

        A Linear is quantized when its dotted path matches no ``exclude``
        substring, meets ``min_features`` on both dimensions, and passes
        ``cls.can_replace``. ``init_kwargs`` reach the
        scheme constructor (``keep_master``, ``device``, fp8's ``recipe``). Returns the dotted paths swapped.
        """

        swapped: list[str] = []
        # Replaced in place as we go: collecting every target first would keep all
        # source Linears alive beside their quantized copies, roughly doubling the
        # swap's peak memory on a rollout-sized policy.
        for parent_path, parent in root.named_modules():
            for child_name, child in list(parent.named_children()):
                if not isinstance(child, nn.Linear):
                    continue
                path = f"{parent_path}.{child_name}" if parent_path else child_name
                if any(token in path for token in exclude):
                    continue
                if child.in_features < min_features or child.out_features < min_features:
                    continue
                if not cls.can_replace(child):
                    continue
                setattr(parent, child_name, cls(child, **init_kwargs))
                swapped.append(path)
        return swapped

    def _load_from_state_dict(self, state_dict, prefix, *args) -> None:
        """Reject base-weight loads into a master-free module; refresh cache otherwise."""

        if self.weight is None and f"{prefix}weight" in state_dict:
            raise RuntimeError(
                f"cannot load base weights into a master-free {type(self).__name__} "
                f"({prefix.rstrip('.')}): it was built with keep_master=False. "
                "Master-free is for a frozen base; full-finetune weight-sync "
                "must keep the master.",
            )
        super()._load_from_state_dict(state_dict, prefix, *args)
        # Full-parameter sync overwrites `weight`, so refresh the derived cache.
        # Adapter-only sync still recurses through every child module even though
        # its state dict contains only LoRA keys. Master-free rollout linears have
        # no `weight` by design; their frozen packed cache must remain untouched.
        if self.weight is not None:
            self._requantize_weight()

    @staticmethod
    def _apply_preserving_dtype(
        tensor: torch.Tensor,
        fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Apply a module move while shielding a packed cache from dtype casts."""

        original_shape = tensor.shape
        original_dtype = tensor.dtype
        raw = tensor.reshape(-1).view(torch.uint8)
        return fn(raw).view(original_dtype).reshape(original_shape)

    def _apply(self, fn, recurse: bool = True):
        """Move/cast source masters normally and keep derived caches format-stable."""

        caches = {name: self._buffers[name] for name in self.cache_buffer_names}
        # ``nn.Module._apply`` skips ``None`` buffers. Hiding caches prevents an
        # FP8 -> BF16 cast and avoids unsupported copy kernels for packed FP4.
        for name in caches:
            self._buffers[name] = None
        try:
            result = super()._apply(fn, recurse=recurse)
        except Exception:
            for name, cache in caches.items():
                self._buffers[name] = cache
            raise

        weight = getattr(self, "weight", None)
        if weight is not None:
            self._requantize_weight()
        else:
            for name, cache in caches.items():
                self._buffers[name] = (
                    None if cache is None else self._apply_preserving_dtype(cache, fn)
                )
        return result


__all__ = ["QuantizedLinear"]
