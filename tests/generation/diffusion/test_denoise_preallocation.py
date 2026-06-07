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
    """Checks preallocate denoise buffers matches latent shape dtype and device."""
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
    """Checks preallocate denoise buffers rejects sample count mismatch."""
    with pytest.raises(ValueError, match="expected 3"):
        preallocate_denoise_buffers(
            state=_state(batch=2, steps=1),
            config=_config(sample_count=3),
        )


@pytest.mark.parametrize("return_kl", [False, True])
def test_run_denoise_steps_writes_preallocated_buffers(return_kl: bool) -> None:
    """Checks run denoise steps writes preallocated buffers."""
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


def test_run_denoise_steps_records_rollout_transformer_dtype() -> None:
    """Checks denoise metadata records the rollout forward dtype."""
    executor = _Executor()

    result = executor.run_denoise_steps(
        state=_state(batch=2, steps=1),
        encoded={"prompt_embeds": torch.zeros(2, 1, dtype=torch.float16)},
        config=_config(sample_count=2),
    )

    assert result.engine_counters["diffusion_rollout_transformer_dtype"] == "float16"
    assert result.engine_counters["diffusion_rollout_autocast_enabled"] is False


def test_decode_denoise_result_threads_rollout_dtype_into_context() -> None:
    """Checks rollout dtype is visible in trajectory context."""
    executor = _Executor()
    state = _state(batch=2, steps=1)
    denoise = executor.run_denoise_steps(
        state=state,
        encoded={"prompt_embeds": torch.zeros(2, 1, dtype=torch.float16)},
        config=_config(sample_count=2),
    )

    chunk = executor.decode_denoise_result(
        request=SimpleNamespace(),
        config=_config(sample_count=2),
        denoise_result=denoise,
    )

    assert chunk.context["rollout_transformer_dtype"] == "float16"
    assert chunk.context["rollout_autocast_enabled"] is False


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

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return latents

    def export_replay_tensors(self, state: SimpleNamespace) -> dict[str, torch.Tensor]:
        return {"prompt_embeds": torch.zeros_like(state.latents[:, :1])}

    def export_batch_context(self, state: SimpleNamespace) -> dict[str, object]:
        del state
        return {"model_family": "test"}


class _Executor(DiffusionPipelineExecutorBase):
    family = "test"
    task = "t2i"
    model = _Model()
