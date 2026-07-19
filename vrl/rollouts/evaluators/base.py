"""Evaluator protocol — extract training signals from model forward results."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vrl.models.interfaces import ReplayModel
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.types import SignalRequest, TrajectorySignalBatch


@runtime_checkable
class Evaluator(Protocol):
    """Extract training signals from model forward results.

    Uses ``model.replay_forward`` for the train-time forward pass and
    extracts trajectory-native signals (log_prob, KL, masks, etc.). The
    selected role precision is stamped on the model at RuntimeBundle assembly.

    Evaluators may publish ``replay_granularity='trajectory'`` when causal
    state requires one ordered replay over every policy action. The default is
    step replay, preserving existing full-sequence and token behavior.

    Replay ownership lives on the model. Evaluators must not route train-time
    replay through collectors.
    """

    def evaluate(
        self,
        model: ReplayModel,
        batch: RolloutBatch,
        timestep_idx: int,
        ref_model: ReplayModel | None = None,
        signal_request: SignalRequest | None = None,
    ) -> TrajectorySignalBatch:
        """Run model.replay_forward() -> extract log_prob, KL, etc."""
        ...
