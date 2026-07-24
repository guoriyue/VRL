"""Configuration owned by the continuous denoise-step loop."""

from __future__ import annotations

from dataclasses import dataclass

from vrl.generation.steps.denoise.teacache import TeaCacheConfig


@dataclass(frozen=True, slots=True)
class DenoiseSDEParams:
    """Parsed SDE knobs used to sample and score denoise transitions."""

    noise_level: float
    sde_type: str
    same_latent: bool
    return_kl: bool
    return_prev_sample_mean: bool = False
    cache_ref_noise_pred: bool = False


@dataclass(frozen=True, slots=True)
class DenoiseLoopConfig:
    """Runtime inputs for one sample chunk's denoise loop."""

    prompt_index: int
    sample_start: int
    sample_count: int
    seed: int | None
    sde: DenoiseSDEParams
    sde_window: tuple[int, int] | None
    denoise_mode: str = "sde"
    teacache: TeaCacheConfig | None = None
    execute_steps: int | None = None


__all__ = ["DenoiseLoopConfig", "DenoiseSDEParams"]
