"""Lever D data-link: opt-in cache of the frozen reference noise_pred.

sampling.cache_ref_noise_pred -> DenoiseLoopConfig.cache_ref_noise_pred gates
a per-step, full-latent buffer that generation fills with the LoRA-disabled ref
forward. The SDE evaluator reads it back as the denoise replay tensor
``ref_noise_pred`` and runs the same sde_step_with_logprob on it instead of
rerunning the ref forward every ppo_epoch. Off by default so non-KL runs pay
nothing, and bit-identical to the fresh ref path when on.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import torch

from vrl.config.precision import RolePrecision
from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.generation.bindings.full_sequence_denoise.layout import DiffusionRequestLayout
from vrl.generation.steps.denoise.config import (
    DenoiseLoopConfig,
    DenoiseRequestOptions,
    DenoiseSDEParams,
)
from vrl.generation.steps.denoise.loop import preallocate_denoise_buffers
from vrl.models.interfaces import ReplayResult, ReplaySegmentResult
from vrl.models.steps.denoise import DiffusionModelBase
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.denoise.sde_logprob import DiffusionSDELogProbEvaluator
from vrl.rollouts.evaluators.types import SignalRequest
from vrl.trajectory import build_diffusion_trajectory

_PRECISION = RolePrecision(
    dtype="fp32",
    float32_precision="ieee",
    outer_autocast=False,
)

# -- generation-side buffer + config ----------------------------------------


def _config(*, cache_ref_noise_pred: bool) -> DenoiseLoopConfig:
    return DenoiseLoopConfig(
        sample_start=0,
        sample_count=2,
        seed=None,
        sde=DenoiseSDEParams(
            noise_level=1.0,
            sde_type="flow_grpo",
            return_kl=False,
            cache_ref_noise_pred=cache_ref_noise_pred,
        ),
        sde_window=None,
    )


def _state() -> SimpleNamespace:
    # 2 samples, 3 denoise steps, latent shape (4, 8, 8).
    return SimpleNamespace(latents=torch.zeros(2, 4, 8, 8), timesteps=[0.9, 0.5, 0.1])


def test_ref_buffer_allocated_with_step_and_latent_shape_when_opted_in() -> None:
    buffers = preallocate_denoise_buffers(
        state=_state(), config=_config(cache_ref_noise_pred=True)
    )
    assert buffers.ref_noise_preds is not None
    # [chunk_batch, num_steps, *latent_shape]
    assert tuple(buffers.ref_noise_preds.shape) == (2, 3, 4, 8, 8)
    assert buffers.ref_noise_preds.dtype == buffers.observations.dtype


def test_ref_buffer_is_none_by_default() -> None:
    buffers = preallocate_denoise_buffers(
        state=_state(), config=_config(cache_ref_noise_pred=False)
    )
    assert buffers.ref_noise_preds is None
    # The always-on tensors are unaffected.
    assert buffers.observations is not None
    assert buffers.log_probs is not None


def test_layout_parses_cache_ref_noise_pred_flag() -> None:
    layout = DiffusionRequestLayout(
        default_num_frames=1,
        default_fps=None,
        default_max_sequence_length=512,
        sde_type="flow_grpo",
    )
    sampling = {"num_steps": 3, "guidance_scale": 1.0, "height": 16, "width": 16}
    request = GenerationRequest(
        request_id="req",
        family="sd3_5",
        task="t2i",
        inputs=["p"],
        samples_per_prompt=1,
        sampling=sampling,
        denoise=DenoiseRequestOptions(cache_ref_noise_pred=True),
    )
    params = layout.parse_sampling_params(request)
    assert params.sde.cache_ref_noise_pred is True
    # Default stays off when the request carries no options.
    default_request = GenerationRequest(
        request_id="req-default",
        family="sd3_5",
        task="t2i",
        inputs=["p"],
        samples_per_prompt=1,
        sampling=sampling,
    )
    assert layout.parse_sampling_params(default_request).sde.cache_ref_noise_pred is False


# -- replay-side consumption ------------------------------------------------


def _batch(*, ref_noise_pred: torch.Tensor | None) -> RolloutBatch:
    observations = torch.ones(2, 2, 3)
    actions = torch.ones(2, 2, 3) * 0.5
    replay_tensors: dict[str, Any] = {}
    if ref_noise_pred is not None:
        replay_tensors["ref_noise_pred"] = ref_noise_pred
    request = GenerationRequest(
        request_id="req",
        family="sd3_5",
        task="t2i",
        inputs=["p0", "p1"],
        samples_per_prompt=1,
    )
    sample_rows = [
        GenerationSampleRow(
            prompt_index=index,
            sample_index=0,
            prompt=f"p{index}",
            sample_id=f"s{index}",
        )
        for index in range(2)
    ]
    trajectory = build_diffusion_trajectory(
        request=request,
        sample_rows=sample_rows,
        observations=observations,
        actions=actions,
        old_log_prob=torch.zeros(2, 2),
        timesteps=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        kl=torch.zeros(2, 2),
        replay_tensors=replay_tensors,
        context={"guidance_scale": 1.0, "cfg": False, "model_family": "sd3_5"},
    )
    return RolloutBatch(
        rewards=torch.ones(2),
        group_ids=torch.arange(2),
        trajectory=trajectory,
    )


def test_cached_ref_skips_the_ref_forward() -> None:
    """With a cached ref_noise_pred, evaluate never reruns the ref forward."""
    model = _CountingReplayModel()
    evaluator = DiffusionSDELogProbEvaluator(_Scheduler())
    batch = _batch(ref_noise_pred=torch.full((2, 2, 3), 0.25))

    signals = evaluator.evaluate(
        model,
        batch,
        timestep_idx=1,
        ref_model=model,
        signal_request=SignalRequest(need_ref=True),
    )

    assert signals.primary.ref_log_prob is not None
    # One forward for the policy replay; the ref came from the cache, not a
    # second replay_forward.
    assert model.replay_forward_calls == 1


def test_cached_ref_matches_fresh_ref_forward() -> None:
    """Cache holding the same value the ref forward produces is bit-identical."""
    # The model's ref forward (disable_adapter) returns zeros_like; store zeros.
    cached = _CountingReplayModel()
    fresh = _CountingReplayModel()
    evaluator = DiffusionSDELogProbEvaluator(_Scheduler())

    cached_signals = evaluator.evaluate(
        cached,
        _batch(ref_noise_pred=torch.zeros(2, 2, 3)),
        timestep_idx=1,
        ref_model=cached,
        signal_request=SignalRequest(need_ref=True),
    )
    fresh_signals = evaluator.evaluate(
        fresh,
        _batch(ref_noise_pred=None),
        timestep_idx=1,
        ref_model=fresh,
        signal_request=SignalRequest(need_ref=True),
    )

    assert cached.replay_forward_calls == 1  # policy only
    assert fresh.replay_forward_calls == 2  # policy + ref
    torch.testing.assert_close(
        cached_signals.primary.ref_log_prob,
        fresh_signals.primary.ref_log_prob,
    )


class _Scheduler:
    sigmas = torch.tensor([1.0, 0.5, 0.1])

    def index_for_timestep(self, timestep: torch.Tensor) -> int:
        return int(timestep.item())


class _CountingReplayModel(DiffusionModelBase):
    family = "test"
    precision = _PRECISION
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.replay_forward_calls = 0

    def encode_prompt(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def prepare_sampling(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def forward_step(self, state: Any, step_idx: int) -> dict[str, torch.Tensor]:
        del step_idx
        return {"noise_pred": torch.zeros_like(state.latents)}

    def decode_latents(self, latents: Any) -> Any:
        return latents

    def export_batch_context(self, state: Any) -> dict[str, Any]:
        return {}

    def export_replay_tensors(self, state: Any) -> dict[str, Any]:
        return {}

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> Any:
        del batch_context, step_idx
        return SimpleNamespace(latents=latents, timesteps=replay_tensors["timesteps"])

    def disable_adapter(self) -> Any:
        return nullcontext()

    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int,
        *,
        request: Any | None = None,
    ) -> ReplayResult:
        del request
        self.replay_forward_calls += 1
        replay_tensors, batch_context, latents = self._replay_inputs_for_step(
            batch,
            timestep_idx,
        )
        state = self.restore_eval_state(replay_tensors, batch_context, latents, timestep_idx)
        return ReplayResult(
            segments={
                "denoise": ReplaySegmentResult(
                    segment="denoise",
                    values=self.forward_step(state, 0),
                ),
            },
        )
