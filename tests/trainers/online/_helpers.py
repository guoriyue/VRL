"""Shared OnlineTrainer test helpers: algorithm-input unpacking and trajectory-signal construction."""

from __future__ import annotations


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
    axes = ("sample",) if int(getattr(log_prob, "ndim", 1)) <= 1 else ("sample", "denoise")
    return TrajectorySignalBatch(
        segments={
            "default": SegmentSignal(
                name="default",
                axis="denoise",
                axes=axes,
                distribution="flow_matching",
                log_prob=log_prob,
                old_log_prob=old_log_prob,
                mask=mask,
            ),
        },
        group_ids=batch.group_ids,
        primary_segment="default",
    )
