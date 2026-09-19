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
        def prepare_sampling(self, batch_request, encoded, *, initial_latents=None, marker):
            assert encoded == {"prompt": "test"}
            assert marker == "forwarded"
            assert initial_latents is None
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


def test_group_shared_initial_latent_is_drawn_once_per_prompt_and_expanded() -> None:
    """With ``initial_noise_seed`` set, the executor draws ONE row through the
    family with the group seed and one row of conditioning, expands it to the
    batch, and hands it to the batch's own preparation as ``initial_latents``,
    whose seed stays the per-batch one. Every batch of the prompt, whatever its
    width or offset, starts from that row; a different group seed differs."""
    request = DenoiseRequest(
        width=128, height=128, frame_count=1, num_steps=1, guidance_scale=1.0, seed=5
    )

    class PreparingModel:
        def __init__(self) -> None:
            self.calls: list[tuple[int | None, int, bool]] = []

        def prepare_sampling(self, batch_request, encoded, *, initial_latents=None):
            rows = encoded["prompt_embeds"].shape[0]
            self.calls.append((batch_request.seed, rows, initial_latents is not None))
            if initial_latents is not None:
                latents = initial_latents.clone()
            else:
                generator = torch.Generator().manual_seed(batch_request.seed)
                latents = torch.randn(rows, 8, generator=generator)
            return _State(latents, torch.tensor([1.0]), _Scheduler())

    model = PreparingModel()
    executor = _Executor(model)
    single = {"prompt_embeds": torch.zeros(1, 3)}

    def prepare(start: int, count: int, group_seed: int) -> torch.Tensor:
        config = DenoiseLoopConfig(
            sample_start=start,
            sample_count=count,
            seed=5,
            sde=DenoiseSDEParams(noise_level=0.7, sde_type="flow_grpo"),
            sde_window=None,
            initial_noise_seed=group_seed,
        )
        initial = executor.draw_group_initial_latents(
            request=request, encoded=single, config=config
        )
        assert initial is not None and initial.shape == (count, 8)
        return executor.prepare_denoise_state(
            request=request,
            encoded={"prompt_embeds": torch.zeros(count, 3)},
            config=config,
            initial_latents=initial,
        ).latents

    first, second, child = prepare(0, 4, 21), prepare(4, 2, 21), prepare(6, 1, 21)
    assert all(torch.equal(row, first[0]) for row in first)
    assert all(torch.equal(row, first[0]) for row in second)
    assert torch.equal(child[0], first[0])
    assert not torch.equal(prepare(0, 4, 22)[0], first[0])
    # Group draws: one row, group seed, no initial latent; batch preparations:
    # the batch's rows, the per-batch seed (request seed + offset), the latent.
    assert model.calls[0] == (21, 1, False)
    assert model.calls[1] == (5, 4, True)
    assert model.calls[2] == (21, 1, False)
    assert model.calls[3] == (5 + 4, 2, True)

    # Without a group seed nothing is drawn and the batch prepares on its own.
    config = DenoiseLoopConfig(
        sample_start=0,
        sample_count=2,
        seed=5,
        sde=DenoiseSDEParams(noise_level=0.7, sde_type="flow_grpo"),
        sde_window=None,
    )
    assert (
        executor.draw_group_initial_latents(request=request, encoded=single, config=config) is None
    )


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
