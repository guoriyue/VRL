"""Tests for diffusion denoise replay-buffer preallocation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vrl.generation.diffusion import (
    DiffusionDenoiseConfig,
    DiffusionPipelineExecutorBase,
    preallocate_denoise_buffers,
)


def test_preallocate_denoise_buffers_matches_latent_shape_dtype_and_device() -> None:
    state = _state(batch=2, steps=3, latent_shape=(4, 5), dtype=torch.float16)

    buffers = preallocate_denoise_buffers(state=state, config=_config(sample_count=2))

    assert buffers.observations.shape == (2, 3, 4, 5)
    assert buffers.actions.shape == (2, 3, 4, 5)
    assert buffers.observations.dtype == torch.float16
    assert buffers.actions.dtype == torch.float16
    assert buffers.log_probs.shape == (2, 3)
    assert buffers.log_probs.dtype == torch.float32
    assert buffers.timesteps.shape == (2, 3)
    assert buffers.timesteps.dtype == state.timesteps.dtype
    assert buffers.kl.shape == (2, 3)
    assert buffers.kl.dtype == torch.float32
    assert buffers.observations.device == state.latents.device


def test_preallocate_denoise_buffers_rejects_sample_count_mismatch() -> None:
    with pytest.raises(ValueError, match="expected 3"):
        preallocate_denoise_buffers(
            state=_state(batch=2, steps=1),
            config=_config(sample_count=3),
        )


@pytest.mark.parametrize("return_kl", [False, True])
def test_run_denoise_steps_writes_preallocated_buffers(return_kl: bool) -> None:
    executor = _Executor()
    config = _config(sample_count=2, return_kl=return_kl)

    result = executor.run_denoise_steps(
        state=_state(batch=2, steps=2, latent_shape=(3,)),
        encoded={},
        config=config,
    )

    assert result.observations.shape == (2, 2, 3)
    assert result.actions.shape == (2, 2, 3)
    assert result.log_probs.shape == (2, 2)
    assert result.timesteps.shape == (2, 2)
    assert result.kl.shape == (2, 2)
    assert torch.equal(result.timesteps[0], torch.tensor([0.0, 1.0]))
    if return_kl:
        assert torch.count_nonzero(result.kl).item() > 0
    else:
        assert torch.count_nonzero(result.kl).item() == 0
    assert result.engine_counters["diffusion_num_denoise_steps"] == 2
    assert result.engine_counters["diffusion_sample_batch_size"] == 2
    assert result.engine_counters["diffusion_observation_bytes"] == (
        result.observations.numel() * result.observations.element_size()
    )


def _config(*, sample_count: int = 2, return_kl: bool = False) -> DiffusionDenoiseConfig:
    return DiffusionDenoiseConfig(
        prompt_index=0,
        sample_start=0,
        sample_count=sample_count,
        seed=None,
        same_latent=False,
        sde_window=(0, 0),
        return_kl=return_kl,
    )


def _state(
    *,
    batch: int,
    steps: int,
    latent_shape: tuple[int, ...] = (2,),
    dtype: torch.dtype = torch.float32,
) -> SimpleNamespace:
    return SimpleNamespace(
        latents=torch.arange(batch * int(torch.tensor(latent_shape).prod()), dtype=dtype).view(
            batch,
            *latent_shape,
        ),
        timesteps=torch.arange(steps, dtype=torch.float32),
        scheduler=_Scheduler(steps),
    )


class _Scheduler:
    def __init__(self, steps: int) -> None:
        self.sigmas = torch.linspace(1.0, 0.1, steps + 1)

    def index_for_timestep(self, timestep: torch.Tensor) -> int:
        return int(timestep.item())


class _Model:
    def forward_step(self, state: SimpleNamespace, step_idx: int) -> dict[str, torch.Tensor]:
        del step_idx
        return {"noise_pred": torch.full_like(state.latents, 0.25)}


class _Executor(DiffusionPipelineExecutorBase):
    family = "test"
    task = "t2i"
    model = _Model()
