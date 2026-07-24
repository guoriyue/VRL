"""§3 data-link: opt-in storage of the rollout proposal mean.

sampling.return_prev_sample_mean -> DenoiseLoopConfig.return_prev_sample_mean
gates a per-step, full-latent buffer that the trust-region losses (Flow-DPPO /
GRPO-Guard) read back as old_prev_sample_mean. Off by default so non-trust-region
runs pay nothing.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vrl.generation.steps.denoise import (
    DenoiseLoopConfig,
    DenoiseSDEParams,
    preallocate_denoise_buffers,
)


def _config(*, return_prev_sample_mean: bool) -> DenoiseLoopConfig:
    return DenoiseLoopConfig(
        prompt_index=0,
        sample_start=0,
        sample_count=2,
        seed=None,
        sde=DenoiseSDEParams(
            noise_level=1.0,
            sde_type="flow_grpo",
            same_latent=False,
            return_kl=False,
            return_prev_sample_mean=return_prev_sample_mean,
        ),
        sde_window=None,
    )


def _state() -> SimpleNamespace:
    # 2 samples, 3 denoise steps, latent shape (4, 8, 8).
    return SimpleNamespace(latents=torch.zeros(2, 4, 8, 8), timesteps=[0.9, 0.5, 0.1])


def test_buffer_allocated_with_step_and_latent_shape_when_opted_in() -> None:
    buffers = preallocate_denoise_buffers(
        state=_state(), config=_config(return_prev_sample_mean=True)
    )
    assert buffers.prev_sample_means is not None
    # [chunk_batch, num_steps, *latent_shape]
    assert tuple(buffers.prev_sample_means.shape) == (2, 3, 4, 8, 8)
    assert buffers.prev_sample_means.dtype == buffers.observations.dtype


def test_buffer_is_none_by_default() -> None:
    buffers = preallocate_denoise_buffers(
        state=_state(), config=_config(return_prev_sample_mean=False)
    )
    assert buffers.prev_sample_means is None
    # The always-on tensors are unaffected.
    assert buffers.observations is not None
    assert buffers.log_probs is not None
