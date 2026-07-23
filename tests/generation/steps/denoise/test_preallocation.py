"""Tests for diffusion denoise replay-buffer preallocation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vrl.generation.bindings.full_sequence_denoise import DiffusionChunkExecutorBase
from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.steps.denoise import (
    DenoiseLoopConfig,
    DenoiseSDEParams,
    preallocate_denoise_buffers,
)
from vrl.generation.types import GenerationRequest


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
    assert result.engine_counters["diffusion_samples_per_chunk"] == 2
    assert result.engine_counters["diffusion_observation_bytes"] == (
        result.observations.numel() * result.observations.element_size()
    )


def test_decode_denoise_result_does_not_serialize_model_precision() -> None:
    """Execution policy stays on the model instead of entering trajectories."""
    executor = _Executor()

    result = executor.run_denoise_steps(
        state=_state(batch=2, steps=1),
        encoded={"prompt_embeds": torch.zeros(2, 1, dtype=torch.float16)},
        config=_config(sample_count=2),
    )

    chunk = executor.decode_denoise_result(
        config=_config(sample_count=2),
        denoise_result=result,
    )

    assert chunk.context == {"model_family": "test"}


def test_executor_adds_no_autocast_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model owns its forward contract; the executor must not add autocast.

    ``DiffusionModelBase`` wraps ``forward_step`` with the stamped contract
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
        encoded={},
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
        encoded={"prompt_embeds": torch.zeros(2, 1, dtype=torch.float16)},
        config=_config(sample_count=2),
    )

    chunk = executor.decode_denoise_result(
        config=_config(sample_count=2),
        denoise_result=denoise,
    )

    assert chunk.context == {"model_family": "test"}


def test_forward_chunk_plan_routes_through_stage_boundary_methods() -> None:
    """Checks fused chunk execution runs through the typed stage boundary."""
    executor = _StageTrackingExecutor()
    request = GenerationRequest(
        request_id="req-1",
        family="test",
        task="t2i",
        inputs=["prompt"],
        samples_per_prompt=2,
    )
    chunk = SampleChunk(
        prompt_index=0,
        prompt="prompt",
        sample_start=0,
        sample_count=2,
    )

    result = executor.forward_chunk_plan(
        request,
        chunk,
    )

    assert result == {"decoded": True, "stage_durations": {"encode": 1.0, "denoise": 2.0}}
    assert executor.calls == [
        "build_prompt_stage_input",
        "run_prompt_encode_stage",
        "run_prepare_stage",
        "run_denoise_stage",
        "run_decode_stage",
    ]


def _config(*, sample_count: int = 2, return_kl: bool = False) -> DenoiseLoopConfig:
    return DenoiseLoopConfig(
        prompt_index=0,
        sample_start=0,
        sample_count=sample_count,
        seed=None,
        sde=DenoiseSDEParams(
            noise_level=1.0,
            sde_type="flow_grpo",
            same_latent=False,
            return_kl=return_kl,
        ),
        sde_window=(0, 0),
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


class _Executor(DiffusionChunkExecutorBase):
    family = "test"
    task = "t2i"
    model = _Model()


class _StageTrackingExecutor(DiffusionChunkExecutorBase):
    family = "test"
    task = "t2i"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def build_prompt_stage_input(
        self,
        request: GenerationRequest,
        chunk: SampleChunk,
    ) -> SimpleNamespace:
        assert request.request_id == "req-1"
        assert chunk.chunk_key == "prompt:0:samples:0:2"
        self.calls.append("build_prompt_stage_input")
        return SimpleNamespace(stage="prompt")

    def run_prompt_encode_stage(
        self,
        payload: Any,
        *,
        stage_durations: dict[str, float],
        record_function: Any,
    ) -> SimpleNamespace:
        assert payload.stage == "prompt"
        assert callable(record_function)
        self.calls.append("run_prompt_encode_stage")
        stage_durations["encode"] = 1.0
        return SimpleNamespace(stage="encoded")

    def run_prepare_stage(
        self,
        payload: Any,
        *,
        stage_durations: dict[str, float],
    ) -> SimpleNamespace:
        assert payload.stage == "encoded"
        assert stage_durations == {"encode": 1.0}
        self.calls.append("run_prepare_stage")
        return SimpleNamespace(stage="prepared")

    def run_denoise_stage(
        self,
        payload: Any,
        *,
        stage_durations: dict[str, float],
    ) -> SimpleNamespace:
        assert payload.stage == "prepared"
        self.calls.append("run_denoise_stage")
        stage_durations["denoise"] = 2.0
        return SimpleNamespace(stage="denoised", stage_durations=dict(stage_durations))

    def run_decode_stage(self, payload: Any) -> dict[str, Any]:
        assert payload.stage == "denoised"
        self.calls.append("run_decode_stage")
        return {"decoded": True, "stage_durations": payload.stage_durations}


def test_decode_denoise_result_packs_video_as_uint8() -> None:
    """Checks decoded video crosses the wire as uint8 (wire diet T1).

    Training tensors (observations/actions/log_probs) must stay untouched;
    only the decoded video is quantized, with the canonical to_uint8 formula.
    """

    class _UnitVideoModel(_Model):
        def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
            del latents
            return torch.linspace(0.0, 1.0, 16, dtype=torch.float32).view(1, 1, 4, 4)

    class _UnitVideoExecutor(DiffusionChunkExecutorBase):
        family = "test"
        task = "t2i"
        model = _UnitVideoModel()

    executor = _UnitVideoExecutor()
    denoise = executor.run_denoise_steps(
        state=_state(batch=2, steps=1),
        encoded={},
        config=_config(sample_count=2),
    )

    chunk = executor.decode_denoise_result(
        config=_config(sample_count=2),
        denoise_result=denoise,
    )

    from vrl.utils.media import to_uint8

    assert chunk.video.dtype == torch.uint8
    expected = to_uint8(torch.linspace(0.0, 1.0, 16, dtype=torch.float32).view(1, 1, 4, 4))
    assert torch.equal(chunk.video, expected)
    # One byte per element on the wire.
    assert chunk.engine_counters["diffusion_video_bytes"] == chunk.video.numel()
    # Training tensors keep their dtype.
    assert chunk.observations.is_floating_point()
    assert chunk.log_probs.dtype == torch.float32


def test_apply_wire_storage_policy_downcasts_before_wire() -> None:
    """Checks rollout.trajectory_storage applies at the worker boundary.

    Downcasting only saves transfer bytes if it happens BEFORE the
    worker->driver wire; the driver-side application stays as an idempotent
    fallback. The default preserve policy must be an identity (GRPO baseline).
    """
    executor = _Executor()
    denoise = executor.run_denoise_steps(
        state=_state(batch=2, steps=1),
        encoded={},
        config=_config(sample_count=2),
    )
    chunk = executor.decode_denoise_result(
        config=_config(sample_count=2),
        denoise_result=denoise,
    )

    request = GenerationRequest(
        request_id="req-policy",
        family="test",
        task="t2i",
        inputs=["p"],
        samples_per_prompt=2,
        sampling={"trajectory_storage": {"dtype": "float16"}},
    )
    out = executor.apply_wire_storage_policy(request, chunk)

    assert out.observations.dtype == torch.float16
    assert out.actions.dtype == torch.float16
    assert out.replay_tensors["prompt_embeds"].dtype == torch.float16

    # Default policy is a strict identity: same tensor objects, no copies.
    chunk2 = executor.decode_denoise_result(
        config=_config(sample_count=2),
        denoise_result=executor.run_denoise_steps(
            state=_state(batch=2, steps=1),
            encoded={},
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
    before = chunk2.observations
    untouched = executor.apply_wire_storage_policy(plain_request, chunk2)
    assert untouched.observations is before
