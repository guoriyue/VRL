"""Tests for diffusion denoise step selection."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from vrl.generation.bindings.full_sequence_denoise.executor import DenoiseBatchExecutorBase
from vrl.generation.protocols import BatchSizeProbeExecutor
from vrl.generation.steps.denoise.config import DenoiseLoopConfig, DenoiseSDEParams
from vrl.generation.types import DenoiseRequest


@pytest.mark.parametrize("seed", [None, 7])
def test_initial_noise_uses_batch_offset_without_mutating_request(seed: int | None) -> None:
    request = DenoiseRequest(
        width=128, height=128, frame_count=9, num_steps=1, guidance_scale=1.0, seed=seed
    )

    class PreparingModel:
        def prepare_sampling(self, batch_request, encoded, *, marker):
            assert encoded == {"prompt": "test"}
            assert marker == "forwarded"
            self.seed = batch_request.seed
            generator = torch.Generator().manual_seed(self.seed or 0)
            return _State(
                torch.randn(1, 8, generator=generator), torch.tensor([1.0]), _Scheduler()
            )

    model = PreparingModel()
    executor = _Executor(model)
    states = []
    for start in (0, 1, 1):
        config = DenoiseLoopConfig(
            sample_start=start,
            sample_count=1,
            seed=seed,
            sde=DenoiseSDEParams(noise_level=0.7, sde_type="flow_grpo"),
            sde_window=None,
            denoise_mode="native",
        )
        states.append(
            executor.prepare_denoise_state(
                request=request,
                encoded={"prompt": "test"},
                config=config,
                prepare_kwargs={"marker": "forwarded"},
            )
        )
        assert model.seed == (None if seed is None else seed + start)
        assert request.seed == seed
        assert config.seed == seed
    assert torch.equal(states[1].latents, states[2].latents)
    if seed is not None:
        assert not torch.equal(states[0].latents, states[1].latents)


def test_group_shared_initial_noise_is_one_latent_per_prompt_across_batches() -> None:
    """With ``initial_noise_seed`` set, every row of every batch of the prompt starts
    from the same latent — the seed is not offset by the batch position and row 0
    is broadcast over the batch — while a different prompt seed gives a different one."""
    request = DenoiseRequest(
        width=128, height=128, frame_count=1, num_steps=1, guidance_scale=1.0, seed=None
    )

    class PreparingModel:
        def __init__(self) -> None:
            self.rows = 4

        def prepare_sampling(self, batch_request, encoded):
            del encoded
            generator = torch.Generator().manual_seed(batch_request.seed)
            return _State(
                torch.randn(self.rows, 8, generator=generator), torch.tensor([1.0]), _Scheduler()
            )

    executor = _Executor(PreparingModel())

    def prepare(start: int, count: int, seed: int) -> torch.Tensor:
        executor.model.rows = count
        config = DenoiseLoopConfig(
            sample_start=start,
            sample_count=count,
            seed=None,
            sde=DenoiseSDEParams(noise_level=0.7, sde_type="flow_grpo"),
            sde_window=None,
            initial_noise_seed=seed,
        )
        return executor.prepare_denoise_state(request=request, encoded={}, config=config).latents

    first, second = prepare(0, 4, 21), prepare(4, 2, 21)
    assert all(torch.equal(row, first[0]) for row in first)
    assert torch.equal(second[0], first[0]) and torch.equal(second[1], first[0])
    assert not torch.equal(prepare(0, 4, 22)[0], first[0])


def test_diffusion_executor_base_satisfies_probe_protocol() -> None:
    """The probe's ``samples_per_generation_batch: auto`` diffusion-only gate keys off this."""
    assert issubclass(DenoiseBatchExecutorBase, BatchSizeProbeExecutor)
    assert isinstance(_Executor(_Model()), BatchSizeProbeExecutor)
    assert not isinstance(object(), BatchSizeProbeExecutor)


def test_native_denoise_mode_uses_scheduler_step() -> None:
    """``denoise_mode="native"`` steps through the scheduler's own ``step`` (one call per step)
    and records the stepped latents as the action, so the trajectory still carries a log-prob
    column without an SDE.
    """
    scheduler = _Scheduler()
    state = _State(
        latents=torch.ones(1, 1),
        timesteps=torch.tensor([1.0]),
        scheduler=scheduler,
    )
    executor = _Executor(_Model())

    result = executor.run_denoise_steps(
        state=state,
        config=DenoiseLoopConfig(
            sample_start=0,
            sample_count=1,
            seed=None,
            sde=DenoiseSDEParams(
                noise_level=0.7,
                sde_type="flow_grpo",
            ),
            sde_window=None,
            denoise_mode="native",
        ),
    )

    assert scheduler.step_calls == 1
    assert torch.equal(state.latents, torch.full((1, 1), 13.0))
    assert torch.equal(result.actions[:, 0], torch.full((1, 1), 13.0))
    assert result.log_probs.shape == (1, 1)


class _Executor(DenoiseBatchExecutorBase):
    family = "fake"
    task = "t2i"


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
