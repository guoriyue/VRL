"""Evaluator protocol — extract training signals from model forward results."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vrl.models.interfaces import ReplayModel, require_replay_model
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
    step replay for evaluators that recompute one denoise transition per call.

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


class ReplayEvaluatorBase:
    """Opt-in preamble for evaluators that replay through a ``ReplayModel``.

    Every evaluator opened ``evaluate()`` by re-checking the model — and the
    optional reference model — against the ReplayModel contract, naming itself
    as the failing boundary. That owner string was literally
    ``f"{type(self).__name__}.model"`` at every call site, so the evaluator is
    the subject and derives it here instead of spelling it out per family.

    List this **before** ``Evaluator`` in the bases: ``Evaluator`` is a
    ``Protocol``, and with the wrong order its ``...`` stubs win the MRO
    silently rather than failing at class creation.
    """

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


__all__ = ["Evaluator", "ReplayEvaluatorBase"]
