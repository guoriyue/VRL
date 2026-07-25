"""Shared ownership contract for rollout-side quantized linear schemes.

Every rollout quantization scheme subclasses :class:`QuantizedLinear`. Runtime
guards identify the exact requested scheme through ``quantization_scheme`` while
master cleanup and device moves remain scheme-neutral.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class QuantizedLinear(nn.Module):
    """Base for a low-precision drop-in ``nn.Linear`` replacement.

    Subclasses declare their derived caches through ``cache_buffer_names``.
    ``Module.to(dtype=...)`` must never cast packed FP4/FP8 caches into the model's
    base dtype: when a source master exists, caches are rebuilt from the moved
    master; master-free rollout caches move as raw bytes and preserve their exact
    format.
    """

    quantization_scheme: str
    cache_buffer_names: tuple[str, ...] = ()

    def _requantize_weight(self) -> None:
        """Rebuild derived low-precision buffers from ``self.weight``."""

        raise NotImplementedError

    def drop_master(self) -> int:
        """Free the source-dtype master, keeping only the derived cache.

        Valid whenever weight-sync never loads base weights into this module:
        LoRA rollouts sync adapters only, and probes/inference sync nothing.
        A subsequent state-dict load fails loud (see ``_load_from_state_dict``)
        instead of silently skipping the (gone) master. Returns the bytes freed.
        """

        if self.weight is None:
            return 0
        freed = self.weight.numel() * self.weight.element_size()
        self.weight = None
        return freed

    def _load_from_state_dict(self, state_dict, prefix, *args) -> None:
        """Reject base-weight loads after ``drop_master``; refresh cache otherwise."""

        if self.weight is None and f"{prefix}weight" in state_dict:
            raise RuntimeError(
                f"cannot load base weights into a master-free {type(self).__name__} "
                f"({prefix.rstrip('.')}): the source master was dropped "
                "(drop_master). Master-free is for adapter-only/frozen "
                "rollouts; full-finetune weight-sync must keep the master.",
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
