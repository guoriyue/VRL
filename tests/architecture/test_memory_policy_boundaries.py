"""Architecture checks for runtime memory policy boundaries."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VRL_ROOT = ROOT / "vrl"


def test_family_model_loaders_use_vae_decode_memory_boundary() -> None:
    """VAE tiling/slicing is applied by the vae_decode_memory optimization pass, never inline."""

    # Only VAE tiling/slicing has a policy boundary (the VaeDecodeMemory pass). Diffusers
    # pipeline-level offload through model.offload_mode is a legitimate single-GPU
    # inference strategy — e.g. Wan I2V 14B on a 32 GB card — so the underlying
    # accelerate calls are intentionally not forbidden here.
    violations: list[str] = []
    for path in sorted((VRL_ROOT / "models" / "families").rglob("model.py")):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for snippet in ("enable_tiling(", "enable_slicing("):
                if snippet in line:
                    violations.append(
                        f"{path.relative_to(ROOT)}:{lineno}: inline memory policy call {snippet!r}"
                    )
    assert not violations, "\n".join(violations)
