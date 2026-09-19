"""Grouped evaluator tests for causal temporal-batch denoise policies."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import torch

from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.models.interfaces import ReplayResult, ReplaySegmentResult
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.denoise import (
    ChunkAutoregressiveDenoiseLogProbEvaluator,
)
from vrl.rollouts.evaluators.types import SignalRequest
from vrl.trajectory.builders import build_chunk_autoregressive_denoise_trajectory


def test_grouped_evaluator_flattens_policy_axes_and_replays_reference_once() -> None:
    batch = _batch()
    model = _ReplayModel()
    evaluator = ChunkAutoregressiveDenoiseLogProbEvaluator()

    signals = evaluator.evaluate(
        model,
        batch,
        timestep_idx=99,
        ref_model=model,
        signal_request=SignalRequest(need_ref=True),
    )

    assert evaluator.replay_granularity == "trajectory"
    assert signals.primary.log_prob.shape == (2, 6)
    assert signals.primary.old_log_prob.shape == (2, 6)
    assert signals.primary.mask.shape == (2, 6)
    assert torch.all(signals.primary.log_prob == 3.0)
    assert torch.all(signals.primary.ref_log_prob == 2.0)
    assert model.adapter_disable_count == 1
    assert model.requests == [("denoise",), ("denoise",)]


@pytest.mark.parametrize("shared_reference", [False, True])
def test_reference_replay_does_not_build_a_gradient_graph(shared_reference) -> None:
    class DifferentiableReplay(_ReplayModel):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.grad_modes = []

        def replay_forward(self, *args, **kwargs):
            self.grad_modes.append(torch.is_grad_enabled())
            result = super().replay_forward(*args, **kwargs)
            segment = result.require_segment("denoise")
            segment.values["log_probs"] = segment.values["log_probs"] * self.weight
            return result

    model = DifferentiableReplay()
    reference = model if shared_reference else DifferentiableReplay()
    signal = (
        ChunkAutoregressiveDenoiseLogProbEvaluator()
        .evaluate(
            model, _batch(), 0, ref_model=reference, signal_request=SignalRequest(need_ref=True)
        )
        .primary
    )

    assert model.grad_modes[0] is True
    assert reference.grad_modes[-1] is False
    assert signal.log_prob.requires_grad
    assert not signal.ref_log_prob.requires_grad
    (signal.log_prob - signal.ref_log_prob).sum().backward()
    assert model.weight.grad.item() == 36.0
    if not shared_reference:
        assert reference.weight.grad is None


class _ReplayModel:
    def __init__(self) -> None:
        self.adapter_disabled = False
        self.adapter_disable_count = 0
        self.requests: list[tuple[str, ...] | None] = []

    def replay_forward(
        self,
        batch: RolloutBatch,
        timestep_idx: int = 0,
        *,
        request: Any | None = None,
    ) -> ReplayResult:
        del timestep_idx
        self.requests.append(request.segment_names if request is not None else None)
        value = 2.0 if self.adapter_disabled else 3.0
        shape = batch.trajectory.segments["denoise"].tensors["old_log_prob"].value.shape
        return ReplayResult(
            segments={
                "denoise": ReplaySegmentResult(
                    segment="denoise",
                    values={"log_probs": torch.full(shape, value)},
                ),
            },
        )

    @contextmanager
    def reference_policy(self) -> Iterator[None]:
        self.adapter_disable_count += 1
        self.adapter_disabled = True
        try:
            yield
        finally:
            self.adapter_disabled = False


def _batch() -> RolloutBatch:
    request = GenerationRequest(
        request_id="batch-replay",
        family="causvid",
        task="t2v",
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
    shape = (2, 2, 3)
    trajectory = build_chunk_autoregressive_denoise_trajectory(
        request=request,
        sample_rows=sample_rows,
        observations=torch.zeros(*shape, 1),
        actions=torch.ones(*shape, 1),
        old_log_prob=torch.arange(12, dtype=torch.float32).reshape(shape),
        mask=torch.tensor(
            [
                [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
                [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            ]
        ),
        timesteps=torch.zeros(shape),
        finalized_chunk_latents=torch.zeros(2, 2, 1),
        replay_tensors={},
        context={},
    )
    return RolloutBatch(
        rewards=torch.ones(2),
        group_ids=torch.arange(2),
        trajectory=trajectory,
    )
