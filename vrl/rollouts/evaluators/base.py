"""Evaluator base — extract training signals from model forward results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

from vrl.models.interfaces import ReplayModel, require_replay_model
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.types import SignalRequest, TrajectorySignalBatch


class Evaluator(ABC):
    """Extract training signals from a ``ReplayModel``'s replay forward.

    ``evaluate`` runs ``model.replay_forward`` for the train-time forward pass
    and extracts trajectory-native signals (log_prob, KL, masks, ...). The
    selected role precision is stamped on the model at RuntimeBundle assembly.
    Replay ownership lives on the model; evaluators never route train-time
    replay through collectors.

    ``replay_granularity`` is ``'step'`` for evaluators that recompute one
    denoise transition per call and ``'trajectory'`` when causal state requires
    one ordered replay over every policy action.
    ``supports_deferred_replay_tensor_move`` says the evaluator moves the replay
    tensors it reads to the device itself, so the trainer can skip the eager
    whole-batch move.
    """

    replay_granularity: ClassVar[Literal["step", "trajectory"]] = "step"
    supports_deferred_replay_tensor_move: ClassVar[bool] = False

    @abstractmethod
    def evaluate(
        self,
        model: ReplayModel,
        batch: RolloutBatch,
        timestep_idx: int,
        ref_model: ReplayModel | None = None,
        signal_request: SignalRequest | None = None,
    ) -> TrajectorySignalBatch:
        """Run replay and extract the requested trajectory signals."""

        raise NotImplementedError

    def _require_models(
        self,
        model: ReplayModel,
        ref_model: ReplayModel | None = None,
    ) -> tuple[ReplayModel, ReplayModel | None]:
        """Validate the replay pair, attributing failures to this evaluator."""

        owner = type(self).__name__
        model = require_replay_model(model, owner=f"{owner}.model")
        if ref_model is not None:
            ref_model = require_replay_model(ref_model, owner=f"{owner}.ref_model")
        return model, ref_model


__all__ = ["Evaluator"]
