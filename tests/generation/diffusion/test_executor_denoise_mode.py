"""Tests for diffusion denoise step selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vrl.generation.diffusion.executor import (
    DiffusionChunkExecutorBase,
    DiffusionDenoiseConfig,
)


def test_native_denoise_mode_uses_scheduler_step() -> None:
    """Checks native denoise mode uses scheduler step."""
    scheduler = _Scheduler()
    state = _State(
        latents=torch.ones(1, 1),
        timesteps=torch.tensor([1.0]),
        scheduler=scheduler,
    )
    executor = _Executor(_Model())

    result = executor.run_denoise_steps(
        state=state,
        encoded={},
        config=DiffusionDenoiseConfig(
            prompt_index=0,
            sample_start=0,
            sample_count=1,
            seed=None,
            same_latent=False,
            sde_window=None,
            return_kl=False,
            noise_level=0.7,
            sde_type="sde",
            denoise_mode="native",
        ),
    )

    assert scheduler.step_calls == 1
    assert torch.equal(state.latents, torch.full((1, 1), 13.0))
    assert torch.equal(result.actions[:, 0], torch.full((1, 1), 13.0))
    assert result.log_probs.shape == (1, 1)


class _Executor(DiffusionChunkExecutorBase):
    family = "fake"
    task = "t2i"

    def __init__(self, model: object) -> None:
        self.model = model


class _Model:
    def forward_step(self, state: object, step_idx: int) -> dict[str, torch.Tensor]:
        del state, step_idx
        return {"noise_pred": torch.full((1, 1), 2.0)}


@dataclass
class _State:
    latents: torch.Tensor
    timesteps: torch.Tensor
    scheduler: object


class _Scheduler:
    def __init__(self) -> None:
        self.sigmas = torch.tensor([1.0, 0.5, 0.0])
        self.step_calls = 0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        *,
        return_dict: bool,
    ) -> tuple[torch.Tensor]:
        del timestep
        assert return_dict is False
        self.step_calls += 1
        return (sample + model_output + 10.0,)

    def index_for_timestep(self, timestep: torch.Tensor) -> int:
        del timestep
        return 0
