"""Shared OnlineTrainer test helpers: algorithm-input unpacking and trajectory-signal construction."""

from __future__ import annotations

from dataclasses import asdict

from vrl.models.interfaces import ResolvedForwardPrecision

DEFAULT_FORWARD_PRECISION = ResolvedForwardPrecision(
    autocast="off",
    float32_precision="ieee",
)


def _rollout_context(
    precision: ResolvedForwardPrecision = DEFAULT_FORWARD_PRECISION,
) -> dict[str, object]:
    return {"rollout_forward_precision": asdict(precision)}


def _algorithm_inputs(inputs):
    if inputs.signals is None:
        raise AssertionError("test algorithm requires trajectory signals")
    signals = inputs.signals.primary
    return signals, inputs.advantages, signals.old_log_prob


def _trajectory_signals(batch, log_prob, timestep_idx: int = 0):
    import torch

    from vrl.rollouts.evaluators.types import SegmentSignal, TrajectorySignalBatch

    old_log_prob = torch.full_like(log_prob, float(timestep_idx))
    mask = torch.ones_like(log_prob)
    return TrajectorySignalBatch(
        segments={
            "default": SegmentSignal(
                name="default",
                distribution="flow_matching",
                log_prob=log_prob,
                old_log_prob=old_log_prob,
                mask=mask,
            ),
        },
        group_ids=batch.group_ids,
        primary_segment="default",
    )
