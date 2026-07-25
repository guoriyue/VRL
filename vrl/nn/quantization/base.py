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

from vrl.nn.quantization.targeting import (
    DEFAULT_EXCLUDE,
    LinearTargetProfile,
    matches_linear_target,
)


class QuantizedLinear(nn.Module):
    """Base for a low-precision drop-in ``nn.Linear`` replacement.

    Subclasses declare their derived caches through ``cache_buffer_names``.
    ``Module.to(dtype=...)`` must never cast packed FP4/FP8 caches into the model's
    base dtype: when a source master exists, caches are rebuilt from the moved
    master; master-free rollout caches move as raw bytes and preserve their exact
    format.

    They also own their module-tree swap policy: ``default_target_profile`` is the
    validated production scope for the scheme, and ``can_replace`` rejects shapes
    its kernels cannot take. ``swap_linears`` is the one traversal both feed.
    """

    quantization_scheme: str
    cache_buffer_names: tuple[str, ...] = ()
    # The scheme's validated production scope. Annotation-only so a new scheme
    # that forgets it fails loudly on the first default-profile swap instead of
    # silently inheriting another scheme's accuracy/perf trade-off.
    default_target_profile: LinearTargetProfile

    def _requantize_weight(self) -> None:
        """Rebuild derived low-precision buffers from ``self.weight``."""

        raise NotImplementedError

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
        target_profile: LinearTargetProfile | str | None = None,
        **init_kwargs: Any,
    ) -> list[str]:
        """Replace eligible ``nn.Linear`` modules under ``root`` with ``cls`` in place.

        A Linear is quantized when its dotted path matches no ``exclude``
        substring, belongs to ``target_profile``, meets ``min_features`` on both
        dimensions, and passes ``cls.can_replace``. ``init_kwargs`` reach the
        scheme constructor (fp8's ``recipe``). Returns the dotted paths swapped.
        """

        # Resolved before the traversal on purpose: an invalid profile must raise
        # with the model untouched. It cannot be a signature default either --
        # ``cls`` does not exist while the ``def`` is evaluated.
        profile = (
            cls.default_target_profile
            if target_profile is None
            else LinearTargetProfile(target_profile)
        )
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
                if not matches_linear_target(path, profile):
                    continue
                if child.in_features < min_features or child.out_features < min_features:
                    continue
                if not cls.can_replace(child):
                    continue
                setattr(parent, child_name, cls(child, **init_kwargs))
                swapped.append(path)
        return swapped

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

    Deliberately a free function while ``swap_linears`` is a classmethod: this
    walk is scheme-agnostic (it visits every scheme under ``root`` at once),
    reads no class attribute, and constructs nothing, so it has no subject.
    """
    return sum(
        module.drop_master() for module in root.modules() if isinstance(module, QuantizedLinear)
    )


__all__ = ["QuantizedLinear", "drop_quantized_masters"]
