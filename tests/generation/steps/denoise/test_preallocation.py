"""Tests for diffusion denoise replay-buffer preallocation."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vrl.generation.bindings.full_sequence_denoise import DenoiseBatchExecutorBase
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.steps.denoise.config import DenoiseLoopConfig, DenoiseSDEParams
from vrl.generation.steps.denoise.loop import DenoiseTrajectoryBuffers
from vrl.generation.types import GenerationRequest
from vrl.math.denoise.flow_matching import SDEStepResult
from vrl.trajectory.storage import TrajectoryStoragePolicy


def test_preallocate_denoise_buffers_matches_latent_shape_dtype_and_device() -> None:
    """Preallocated buffers take the latent's shape, dtype and device for observations and
    actions, fp32 for log-probs and KL, and the state's timestep dtype.
    """
    state = _state(batch=2, steps=3, latent_shape=(4, 5), dtype=torch.float16)

    buffers = DenoiseTrajectoryBuffers.allocate(state=state, config=_config(sample_count=2))

    assert buffers.latents.shape == (2, 4, 4, 5)
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_record_step_casts_into_allocated_buffers_without_gradients(dtype, device) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    state = _state(batch=2, steps=1, dtype=dtype)
    state.latents = state.latents.to(device)
    state.timesteps = state.timesteps.to(device)
    config = _config(return_kl=True)
    config = replace(
        config,
        sde=replace(config.sde, return_prev_sample_mean=True),
    )
    buffers = DenoiseTrajectoryBuffers.allocate(state=state, config=config)
    values = torch.tensor(
        [[0.1234, -0.5678], [1.2345, -2.3456]], device=device, requires_grad=True
    )
    log_prob = values[:, 0]
    buffers.record_initial_latents(values)
    buffers.record_step(
        0,
        action=values,
        timestep=state.timesteps[0],
        sde_result=SDEStepResult(values, log_prob, values, None),
        return_kl=True,
    )
    for output in (
        buffers.observations,
        buffers.actions,
        buffers.prev_sample_means,
    ):
        assert output.dtype == dtype
        torch.testing.assert_close(output[:, 0], values.detach().to(dtype), rtol=0, atol=0)
        assert not output.requires_grad
    torch.testing.assert_close(buffers.log_probs[:, 0], log_prob.detach())
    torch.testing.assert_close(buffers.kl[:, 0], log_prob.detach().abs())
    assert not buffers.log_probs.requires_grad
    assert not buffers.kl.requires_grad


def test_observations_and_actions_are_adjacent_views_of_one_latent_path() -> None:
    """Step ``t`` consumes what step ``t - 1`` produced, so the replay pair is stored
    once: ``observations[:, t + 1]`` aliases ``actions[:, t]`` and only the initial
    latent and each step's action are ever written.
    """
    state = _state(batch=2, steps=3, latent_shape=(4,))
    buffers = DenoiseTrajectoryBuffers.allocate(state=state, config=_config(sample_count=2))

    buffers.record_initial_latents(state.latents)
    for step_idx in range(3):
        action = torch.full_like(state.latents, float(step_idx + 1))
        buffers.record_step(
            step_idx,
            action=action,
            timestep=state.timesteps[step_idx],
            sde_result=SDEStepResult(action, action[:, 0], action, None),
            return_kl=False,
        )

    assert buffers.observations.data_ptr() == buffers.latents.data_ptr()
    assert buffers.actions.data_ptr() == buffers.latents[:, 1].data_ptr()
    assert torch.equal(buffers.observations[:, 0], state.latents)
    assert torch.equal(buffers.observations[:, 1:], buffers.actions[:, :-1])
    assert torch.equal(buffers.actions[:, -1], torch.full_like(state.latents, 3.0))


def test_run_denoise_loop_observation_is_previous_action() -> None:
    """The loop's trajectory obeys the same aliasing end to end: the sample each
    forward consumed is the sample the previous step produced, and the first
    observation is the prepared noise.
    """
    state = _state(batch=2, steps=3, latent_shape=(3,))
    initial = state.latents.clone()

    result = _Executor().run_denoise_steps(state=state, config=_config(sample_count=2))

    assert torch.equal(result.observations[:, 0], initial)
    assert torch.equal(result.observations[:, 1:], result.actions[:, :-1])
    assert torch.equal(result.actions[:, -1], result.state.latents.to(result.actions.dtype))


def test_preallocate_denoise_buffers_rejects_sample_count_mismatch() -> None:
    with pytest.raises(ValueError, match="expected 3"):
        DenoiseTrajectoryBuffers.allocate(
            state=_state(batch=2, steps=1),
            config=_config(sample_count=3),
        )


@pytest.mark.parametrize("return_kl", [False, True])
def test_run_denoise_steps_writes_preallocated_buffers(return_kl: bool) -> None:
    """``run_denoise_steps`` fills the preallocated buffers in place: per-step timesteps, KL only
    when requested (zeros otherwise), and engine counters for steps, batch width and
    observation bytes.
    """
    executor = _Executor()
    config = _config(sample_count=2, return_kl=return_kl)

    result = executor.run_denoise_steps(
        state=_state(batch=2, steps=2, latent_shape=(3,)),
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
    assert result.engine_counters["diffusion_samples_per_generation_batch"] == 2
    assert result.engine_counters["diffusion_observation_bytes"] == (
        result.observations.numel() * result.observations.element_size()
    )


@pytest.mark.parametrize("execute_steps, expected_steps", [(1, 1), (5, 3), (None, 3)])
def test_denoise_step_counter_reports_execution_with_full_buffer_capacity(
    execute_steps, expected_steps
) -> None:
    config = replace(_config(), execute_steps=execute_steps)
    result = _Executor().run_denoise_steps(state=_state(batch=2, steps=3), config=config)
    assert result.engine_counters["diffusion_num_denoise_steps"] == expected_steps
    assert result.observations.shape[1] == 3
    assert result.engine_counters["diffusion_observation_bytes"] == (
        result.observations.numel() * result.observations.element_size()
    )


def test_decode_denoise_result_does_not_serialize_model_precision() -> None:
    """Execution policy stays on the model instead of entering trajectories."""
    executor = _Executor()

    result = executor.run_denoise_steps(
        state=_state(batch=2, steps=1),
        config=_config(sample_count=2),
    )

    batch = executor.decode_denoise_result(
        batch=_chunk(),
        config=_config(sample_count=2),
        denoise_result=result,
    )

    assert batch.context == {"model_family": "test"}


def test_executor_adds_no_autocast_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model owns its forward contract; the executor must not add autocast.

    ``DenoiseModelBase`` wraps ``forward_step`` with the stamped contract
    (covered in tests/models/steps/denoise/common/test_model_base.py); scheduler
    and SDE math always run outside any autocast scope.
    """
    from vrl.generation.steps.denoise import loop as loop_module

    forward_states: list[bool] = []
    sde_states: list[bool] = []
    real_sde_step = loop_module.sde_step_with_logprob

    class _AutocastModel(_Model):
        def forward_step(
            self,
            state: SimpleNamespace,
            step_idx: int,
        ) -> dict[str, torch.Tensor]:
            forward_states.append(torch.is_autocast_enabled("cpu"))
            return super().forward_step(state, step_idx)

    def tracked_sde_step(*args: Any, **kwargs: Any) -> Any:
        sde_states.append(torch.is_autocast_enabled("cpu"))
        return real_sde_step(*args, **kwargs)

    monkeypatch.setattr(loop_module, "sde_step_with_logprob", tracked_sde_step)
    executor = _Executor()
    executor.model = _AutocastModel()

    executor.run_denoise_steps(
        state=_state(batch=2, steps=1),
        config=_config(sample_count=2),
    )

    assert forward_states == [False]
    assert sde_states == [False]


def test_decode_denoise_result_uses_only_model_exported_context() -> None:
    """The executor does not append process-local execution policy."""
    executor = _Executor()
    state = _state(batch=2, steps=1)
    denoise = executor.run_denoise_steps(
        state=state,
        config=_config(sample_count=2),
    )

    batch = executor.decode_denoise_result(
        batch=_chunk(),
        config=_config(sample_count=2),
        denoise_result=denoise,
    )

    assert batch.context == {"model_family": "test"}


def test_forward_probe_batch_uses_canonical_flow_with_truncated_steps() -> None:
    """The probe reuses the production batch flow with an explicit step bound."""
    executor = _StageTrackingExecutor()
    request = GenerationRequest(
        request_id="req-1",
        family="test",
        task="t2i",
        inputs=["prompt"],
        samples_per_prompt=2,
        sampling={
            "num_steps": 2,
            "guidance_scale": 1.0,
            "height": 8,
            "width": 8,
        },
    )
    batch = _chunk()

    result = executor.forward_probe_batch(
        request,
        batch,
        execute_steps=1,
    )

    assert result.batch is batch
    assert executor.calls == ["encode", "prepare", "denoise:1", "decode"]


@pytest.mark.parametrize("field", ["sample_start", "sample_count"])
@pytest.mark.parametrize("value", [1.9, "2", True, -1])
def test_denoise_config_rejects_invalid_sample_identity(field, value) -> None:
    with pytest.raises(ValueError, match=field):
        replace(_config(), **{field: value})


@pytest.mark.parametrize("mode", ["Native", "unknown", "", None])
def test_denoise_config_rejects_unknown_mode(mode) -> None:
    with pytest.raises(ValueError, match="denoise_mode"):
        replace(_config(), denoise_mode=mode)


def test_denoise_config_rejects_empty_sample_batch() -> None:
    with pytest.raises(ValueError, match="sample_count"):
        replace(_config(), sample_count=0)


def _config(*, sample_count: int = 2, return_kl: bool = False) -> DenoiseLoopConfig:
    return DenoiseLoopConfig(
        sample_start=0,
        sample_count=sample_count,
        seed=None,
        sde=DenoiseSDEParams(
            noise_level=1.0,
            sde_type="flow_grpo",
            return_kl=return_kl,
        ),
        sde_window=(0, 0),
    )


def _chunk(*, sample_count: int = 2) -> GenerationSampleBatch:
    return GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=sample_count)


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
    def encode_prompt(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        del args, kwargs
        return {"prompt_embeds": torch.ones(1, 1)}

    def prepare_sampling(
        self,
        request: Any,
        encoded: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> SimpleNamespace:
        del kwargs
        return _state(
            batch=int(encoded["prompt_embeds"].shape[0]),
            steps=request.num_steps,
        )

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


class _Executor(DenoiseBatchExecutorBase):
    family = "test"
    task = "t2i"

    def __init__(self) -> None:
        super().__init__(_Model())


class _StageTrackingExecutor(DenoiseBatchExecutorBase):
    family = "test"
    task = "t2i"

    def __init__(self) -> None:
        super().__init__(_Model())
        self.calls: list[str] = []

    def encode_prompt_for_batch(
        self,
        *,
        generation_request: GenerationRequest,
        model_request: Any,
        params: Any,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any]:
        self.calls.append("encode")
        return super().encode_prompt_for_batch(
            generation_request=generation_request,
            model_request=model_request,
            params=params,
            batch=batch,
        )

    def prepare_denoise_state(
        self,
        *,
        request: Any,
        encoded: dict[str, Any],
        config: DenoiseLoopConfig,
        prepare_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        self.calls.append("prepare")
        return super().prepare_denoise_state(
            request=request,
            encoded=encoded,
            config=config,
            prepare_kwargs=prepare_kwargs,
        )

    def run_denoise_steps(
        self,
        *,
        state: Any,
        config: DenoiseLoopConfig,
    ) -> Any:
        self.calls.append(f"denoise:{config.execute_steps}")
        return super().run_denoise_steps(state=state, config=config)

    def decode_denoise_result(self, **kwargs: Any) -> Any:
        self.calls.append("decode")
        return super().decode_denoise_result(**kwargs)


def test_decode_denoise_result_packs_video_as_uint8() -> None:
    """Checks decoded video crosses the wire as uint8 (wire diet T1).

    Training tensors (observations/actions/log_probs) must stay untouched;
    only the decoded video is quantized, with the canonical to_uint8 formula.
    """

    class _UnitVideoModel(_Model):
        def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
            del latents
            return torch.linspace(0.0, 1.0, 16, dtype=torch.float32).view(1, 1, 4, 4)

    class _UnitVideoExecutor(DenoiseBatchExecutorBase):
        family = "test"
        task = "t2i"

        def __init__(self) -> None:
            super().__init__(_UnitVideoModel())

    executor = _UnitVideoExecutor()
    denoise = executor.run_denoise_steps(
        state=_state(batch=2, steps=1),
        config=_config(sample_count=2),
    )

    batch = executor.decode_denoise_result(
        batch=_chunk(),
        config=_config(sample_count=2),
        denoise_result=denoise,
    )

    from vrl.utils.media import to_uint8

    assert batch.video.dtype == torch.uint8
    expected = to_uint8(torch.linspace(0.0, 1.0, 16, dtype=torch.float32).view(1, 1, 4, 4))
    assert torch.equal(batch.video, expected)
    # One byte per element on the wire.
    assert batch.engine_counters["diffusion_video_bytes"] == batch.video.numel()
    # Training tensors keep their dtype.
    assert batch.latents.is_floating_point()
    assert batch.log_probs.dtype == torch.float32


def test_apply_wire_storage_policy_downcasts_before_wire() -> None:
    """Checks rollout.trajectory_storage applies at the worker boundary.

    Downcasting only saves transfer bytes if it happens BEFORE the
    worker->driver wire; the driver-side application stays as an idempotent
    fallback. The default preserve policy must be an identity (GRPO baseline).
    """
    executor = _Executor()
    denoise = executor.run_denoise_steps(
        state=_state(batch=2, steps=1),
        config=_config(sample_count=2),
    )
    batch = executor.decode_denoise_result(
        batch=_chunk(),
        config=_config(sample_count=2),
        denoise_result=denoise,
    )

    request = GenerationRequest(
        request_id="req-policy",
        family="test",
        task="t2i",
        inputs=["p"],
        samples_per_prompt=2,
        trajectory_storage=TrajectoryStoragePolicy(dtype="float16"),
    )
    out = executor.apply_wire_storage_policy(request, batch)

    assert out.latents.dtype == torch.float16
    assert out.replay_tensors["prompt_embeds"].dtype == torch.float16

    # Default policy is a strict identity: same tensor objects, no copies.
    chunk2 = executor.decode_denoise_result(
        batch=_chunk(),
        config=_config(sample_count=2),
        denoise_result=executor.run_denoise_steps(
            state=_state(batch=2, steps=1),
            config=_config(sample_count=2),
        ),
    )
    plain_request = GenerationRequest(
        request_id="req-plain",
        family="test",
        task="t2i",
        inputs=["p"],
        samples_per_prompt=2,
    )
    before = chunk2.latents
    untouched = executor.apply_wire_storage_policy(plain_request, chunk2)
    assert untouched.latents is before


@pytest.mark.parametrize("timesteps", [[0.9, 0.5], [torch.tensor(0.9)], None])
def test_preallocation_requires_tensor_timestep_schedule(timesteps: object) -> None:
    state = _state(batch=2, steps=2)
    state.timesteps = timesteps
    with pytest.raises(TypeError, match=r"state\.timesteps must be a torch\.Tensor"):
        DenoiseTrajectoryBuffers.allocate(state=state, config=_config(sample_count=2))


@pytest.mark.parametrize("steps", [0, -1, True, 1.5, "1"])
def test_probe_rejects_invalid_step_limit_before_encoding(steps) -> None:
    executor = _StageTrackingExecutor()
    request = GenerationRequest(
        request_id="invalid-probe",
        family="test",
        task="t2i",
        inputs=["prompt"],
        samples_per_prompt=2,
    )
    with pytest.raises(ValueError, match="execute_steps"):
        executor.forward_probe_batch(request, _chunk(), execute_steps=steps)
    assert executor.calls == []
