"""Shared base for quantized linear schemes.

Every rollout quantization scheme — ``Fp8Linear`` / ``Fp4Linear`` today and
future ``Int8Linear`` siblings — subclasses :class:`QuantizedLinear`. This is the
single contract the rest of the stack keys off: the rollout backstop guard reads
each module's :attr:`quantization_scheme` instead of maintaining a per-scheme type
list, so a newly added scheme is covered when it implements this contract.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class QuantizedLinear(nn.Module):
    """Marker base for a low-precision drop-in ``nn.Linear`` replacement.

    Subclasses implement the quantized forward and declare their derived cache
    buffers through :attr:`cache_buffer_names`. The base owns ``Module._apply``
    because packed FP4/FP8 caches must move devices without being cast by
    ``module.to(dtype=...)``. When a master exists, caches are rebuilt from the
    moved/cast master; master-free rollout caches preserve their exact dtypes.

    Every scheme also implements ``drop_master() -> int`` (free the
    high-precision master, return bytes freed), so
    :func:`drop_quantized_masters` covers it without a per-scheme type list.
    """

    quantization_scheme: str
    cache_buffer_names: tuple[str, ...] = ()

    def _requantize_weight(self) -> None:
        """Rebuild derived low-precision buffers from ``self.weight``."""
        raise NotImplementedError

    def drop_master(self) -> int:
        """Free the high-precision master and return the released bytes."""
        raise NotImplementedError

    @staticmethod
    def _apply_preserving_dtype(
        tensor: torch.Tensor,
        fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Apply a module move while shielding a packed cache from dtype casts.

        ``Module.to(device, dtype=...)`` casts floating buffers. Viewing the
        cache as bytes makes the same closure move its storage without changing
        its format; restoring the original view preserves FP4/FP8/FP32 exactly.
        """

        original_shape = tensor.shape
        original_dtype = tensor.dtype
        raw = tensor.reshape(-1).view(torch.uint8)
        return fn(raw).view(original_dtype).reshape(original_shape)

    def _apply(self, fn, recurse: bool = True):
        """Move/cast masters normally and keep quantized caches format-stable."""

        caches = {name: self._buffers[name] for name in self.cache_buffer_names}
        # ``nn.Module._apply`` skips ``None`` buffers. This prevents shell FP4
        # copy failures and the more dangerous silent FP8 -> bf16 cache cast.
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


def drop_quantized_masters(root: nn.Module) -> int:
    """Free every quantized linear's high-precision master under ``root``.

    Returns the bytes freed. Valid whenever weight-sync never loads base weights
    into these modules (adapter-only or sync-free rollouts) — see the per-scheme
    ``drop_master`` docstrings.
    """
    return sum(
        module.drop_master() for module in root.modules() if isinstance(module, QuantizedLinear)
    )


__all__ = ["QuantizedLinear", "drop_quantized_masters"]
