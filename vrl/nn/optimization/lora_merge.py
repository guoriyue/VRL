"""Fold LoRA adapters into the rollout policy's base weights.

Rollout replays the policy ``steps x CFG`` times per chunk. With adapters
active every targeted projection costs three GEMMs (base, ``lora_A``,
``lora_B``) plus a scaling multiply; folded, it costs one. That is the
"LoRA plumbing elementwise storm" recorded in
``docs/sprints/info/SPRINT_cross_model_performance.md`` -- measured here at
5-12% of denoise wall time depending on sequence length, and it survives
``torch.compile`` because inductor can fuse the scaling but cannot remove the
two extra matmuls.

Safe only on the rollout worker: the trainer's replay bundle is a separate
model instance (``assemble_replay_bundle`` vs ``build_denoise_runtime_bundle``),
which is why weight sync exists at all, so folding here never touches the
weights gradients flow through.

WHY A PRISTINE COPY: PEFT's ``unmerge`` subtracts ``B @ A`` back out, and in
bf16 that round trip is lossy. Weight sync re-merges every optimizer step, so
the error compounds -- measured at 4.8e-3 relative after one cycle and 2.4e-1
after 1000, i.e. the base weights are destroyed over a normal run. This module
therefore never relies on unmerge arithmetic: it restores the base from an
exact copy captured before the first merge and folds the new adapter onto that.
The copy lives on CPU because the rollout worker's GPU budget is the scarce
one, and it is read once per weight sync, not once per denoise step.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch

from vrl.utils.logging import init_logger

logger = init_logger(__name__)


def lora_layers(root: Any) -> Iterator[tuple[str, Any]]:
    """Yield ``(path, layer)`` for every PEFT LoRA layer under ``root``.

    Walks for the layer type rather than asking the model, because the two
    adapter surfaces in this repo attach the same layers by different routes:
    ``get_peft_model`` wraps the tree in a ``PeftModel``, while diffusers'
    ``PeftAdapterMixin.add_adapter`` (Wan, cosmos-predict2) does not. Both end
    up with ``peft.tuners.lora`` layers, so the layer type is the one seam that
    covers both -- the same reason ``swap_linears`` walks for ``nn.Linear``.
    """

    from peft.tuners.lora.layer import LoraLayer

    modules = getattr(root, "named_modules", None)
    if not callable(modules):
        return
    for path, module in modules():
        if isinstance(module, LoraLayer):
            yield path, module


def capture_pristine_base(model: Any) -> dict[str, torch.Tensor]:
    """Snapshot every LoRA-targeted base weight, before any merge.

    Keyed ``"<core>.<layer path>"`` so a multi-expert family (Wan) cannot
    collide two experts' identically-named projections. Copies are detached,
    CPU-resident, and cloned: a view would alias the live weight the merge is
    about to overwrite.
    """

    pristine: dict[str, torch.Tensor] = {}
    for core_name, core in model.policy_cores.items():
        core = getattr(core, "_orig_mod", core)  # unwrap torch.compile
        for path, layer in lora_layers(core):
            if layer.merged:
                raise RuntimeError(
                    f"LoRA layer {core_name}.{path} is already merged; the "
                    "pristine base must be captured before the first merge.",
                )
            weight = layer.get_base_layer().weight
            pristine[f"{core_name}.{path}"] = weight.detach().to("cpu", copy=True)
    return pristine


def merge_lora_into_base(model: Any, pristine: dict[str, torch.Tensor]) -> list[str]:
    """Fold the active adapter into every LoRA-targeted base weight.

    Idempotent across weight syncs: an already-merged layer is first unmerged
    (so PEFT's own bookkeeping stays truthful about what the forward should
    skip) and then overwritten from ``pristine``, which discards the lossy
    unmerge arithmetic entirely. Returns the merged layer keys for logging.
    """

    merged: list[str] = []
    for core_name, core in model.policy_cores.items():
        core = getattr(core, "_orig_mod", core)
        for path, layer in lora_layers(core):
            key = f"{core_name}.{path}"
            base = pristine.get(key)
            if base is None:
                raise RuntimeError(
                    f"no pristine base weight captured for LoRA layer {key}; "
                    "capture_pristine_base must run before the first merge.",
                )
            if layer.merged:
                # Bookkeeping only -- the arithmetic result is discarded below.
                layer.unmerge()
            weight = layer.get_base_layer().weight
            with torch.no_grad():
                weight.copy_(base.to(device=weight.device, dtype=weight.dtype))
            # safe_merge=True makes PEFT compute into a copy and reject NaN/Inf
            # rather than publishing a corrupted projection to the rollout.
            layer.merge(safe_merge=True)
            merged.append(key)
    return merged


def require_every_core_merged(model: Any) -> None:
    """Every declared policy core must be fully folded, not just the first.

    Checks the LAYERS, not the pass's own account of what it did, for the same
    reason ``require_every_core_quantized`` does: the Wan two-expert bug left
    one expert untouched while the pass believed it had covered everything, and
    a half-merged policy splits one trajectory across two sets of weights.
    """

    unmerged: list[str] = []
    for core_name, core in model.policy_cores.items():
        core = getattr(core, "_orig_mod", core)
        found = False
        for path, layer in lora_layers(core):
            found = True
            if not layer.merged:
                unmerged.append(f"{core_name}.{path}")
        if not found:
            unmerged.append(f"{core_name} (no LoRA layers found)")
    if unmerged:
        preview = ", ".join(unmerged[:5])
        suffix = " ..." if len(unmerged) > 5 else ""
        raise RuntimeError(
            f"rollout LoRA merge left {len(unmerged)} layer(s) unfolded: "
            f"{preview}{suffix}; every projection the policy samples through "
            "must carry the same adapter or one trajectory spans two policies.",
        )


__all__ = [
    "capture_pristine_base",
    "lora_layers",
    "merge_lora_into_base",
    "require_every_core_merged",
]
