"""The reference forward runs before the live forward in every replay evaluator.

A full-parameter policy stands in for its own reference by swapping its weights
in place (``TrainableWeightsSnapshot.active``), which bumps the parameters'
version counters. If that swap happened between the live forward and its
backward, autograd would refuse the graph. These tests run each evaluator with
a policy whose reference is a real in-place snapshot and then backpropagate the
live log-prob, which is exactly what the trainer does.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from vrl.generation import GenerationRequest
from vrl.models.interfaces import ReplayResult, ReplaySegmentResult
from vrl.models.weight_snapshot import TrainableWeightsSnapshot
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.denoise.chunk_autoregressive_logprob import (
    ChunkAutoregressiveDenoiseLogProbEvaluator,
)
from vrl.rollouts.evaluators.denoise.sde_logprob import DenoiseSDELogProbEvaluator
from vrl.rollouts.evaluators.types import SignalRequest
from vrl.trajectory.builders import (
    build_chunk_autoregressive_denoise_trajectory,
    build_diffusion_trajectory,
)


class _Scheduler:
    sigmas = torch.tensor([1.0, 0.5, 0.1])

    def index_for_timestep(self, timestep: torch.Tensor) -> int:
        return int(timestep.item())


class _SnapshotPolicy:
    """A trainable linear whose reference is an in-place snapshot of its weights."""

    def __init__(self, value_key: str) -> None:
        self.linear = nn.Linear(3, 3)
        self.value_key = value_key
        self.reference = TrainableWeightsSnapshot(list(self.linear.parameters()))
        with torch.no_grad():
            self.linear.weight.add_(1.0)  # the live policy has moved on

    def reference_policy(self) -> Any:
        return self.reference.active()

    def replay_forward(self, batch: RolloutBatch, timestep_idx: int = 0, *, request=None):
        del request
        segment = batch.trajectory.segments["denoise"]
        observations = segment.role_tensor("observation").value
        value = self.linear(observations)
        if self.value_key == "log_probs":
            value = value.mean(dim=-1)
        else:
            value = value[:, timestep_idx]
        return ReplayResult(
            segments={
                "denoise": ReplaySegmentResult(segment="denoise", values={self.value_key: value})
            }
        )


def _request() -> GenerationRequest:
    return GenerationRequest(
        request_id="req", family="sd3_5", task="t2i", inputs=["p0", "p1"], samples_per_prompt=1
    )


def test_sde_evaluator_backpropagates_after_an_in_place_reference_forward() -> None:
    request = _request()
    trajectory = build_diffusion_trajectory(
        request=request,
        sample_rows=request.sample_rows(),
        observations=torch.ones(2, 2, 3),
        actions=torch.ones(2, 2, 3) * 0.5,
        old_log_prob=torch.zeros(2, 2),
        timesteps=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        replay_tensors={},
        context={"model_family": "sd3_5"},
    )
    batch = RolloutBatch(rewards=torch.ones(2), group_ids=torch.arange(2), trajectory=trajectory)
    model = _SnapshotPolicy("noise_pred")

    signals = (
        DenoiseSDELogProbEvaluator(_Scheduler())
        .evaluate(model, batch, 0, ref_model=model, signal_request=SignalRequest(need_ref=True))
        .primary
    )
    (signals.log_prob - signals.ref_log_prob).sum().backward()

    assert model.linear.weight.grad is not None
    assert not torch.equal(signals.log_prob, signals.ref_log_prob)


def test_chunk_evaluator_backpropagates_after_an_in_place_reference_forward() -> None:
    request = _request()
    shape = (2, 1, 2)  # sample, temporal_chunk, denoise_transition
    trajectory = build_chunk_autoregressive_denoise_trajectory(
        request=request,
        sample_rows=request.sample_rows(),
        observations=torch.ones(*shape, 3),
        actions=torch.ones(*shape, 3),
        old_log_prob=torch.zeros(shape),
        mask=torch.ones(shape),
        timesteps=torch.zeros(shape),
        finalized_chunk_latents=torch.zeros(2, 1, 3),
        replay_tensors={},
        context={"model_family": "causvid"},
    )
    batch = RolloutBatch(rewards=torch.ones(2), group_ids=torch.arange(2), trajectory=trajectory)
    model = _SnapshotPolicy("log_probs")

    signals = (
        ChunkAutoregressiveDenoiseLogProbEvaluator()
        .evaluate(model, batch, 0, ref_model=model, signal_request=SignalRequest(need_ref=True))
        .primary
    )
    (signals.log_prob - signals.ref_log_prob).sum().backward()

    assert model.linear.weight.grad is not None
