"""VAE decode memory knobs (tiling/slicing) co-located with latent decoding.

VAE tiling and slicing trade latency for peak memory during decode. They live
here next to ``latent_decode.py`` because applying them requires executing on a
concrete diffusers VAE object (``enable_tiling()`` / ``enable_slicing()``), which
is decode-path execution — not a pure config view.
"""

from __future__ import annotations

from typing import Any

from vrl.models.interfaces.generation_memory import (
    GenerationMemoryPolicy,
    VaeDecodeMemory,
)


def configure_vae_decode_memory(
    target: Any,
    mem: VaeDecodeMemory,
) -> None:
    """Apply resolved VAE decode switches; fail on unsupported requests."""

    if mem.tiling:
        target.enable_tiling()
    if mem.slicing:
        target.enable_slicing()


def apply_generation_memory_policy(
    model: Any,
    *,
    memory: GenerationMemoryPolicy | None,
    owner: str,
) -> None:
    """Apply the resolved memory policy to declared generation targets.

    Family models declare WHAT can be configured; this policy owns HOW and
    WHEN. Runtime builders call it once after model construction. Each policy
    field is dispatched explicitly so adding a resolved field also requires a
    real behavior consumer instead of inheriting VAE-specific handling.
    """

    if memory is not None and not isinstance(memory, GenerationMemoryPolicy):
        raise TypeError("memory must be a resolved GenerationMemoryPolicy or None")
    targets = model.generation_memory_targets()
    vae_decode = None if memory is None else memory.vae_decode
    if vae_decode is None:
        return
    if "vae_decode" not in targets:
        exposed = ", ".join(sorted(targets)) or "<none>"
        raise ValueError(
            f"{owner} configures unsupported model.memory section(s) "
            "vae_decode; model exposes generation memory "
            f"target(s): {exposed}",
        )
    configure_vae_decode_memory(
        targets["vae_decode"],
        vae_decode,
    )


__all__ = [
    "apply_generation_memory_policy",
    "configure_vae_decode_memory",
]
