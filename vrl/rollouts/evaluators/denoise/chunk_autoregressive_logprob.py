"""Grouped replay signals for chunk-autoregressive denoise policies."""

from __future__ import annotations

import contextlib
from typing import Any

import torch

from vrl.models.interfaces import ReplayModel
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.base import Evaluator
from vrl.rollouts.evaluators.trajectory import TrajectorySignalBuilder
from vrl.rollouts.evaluators.types import TrajectorySignalBatch
from vrl.trajectory.reader import TrajectoryReader


class ChunkAutoregressiveDenoiseLogProbEvaluator(Evaluator):
    """Replay a complete causal-chunk trajectory in one ordered model pass.

    A scalar denoise-step replay would repeatedly rebuild the temporal prefix
    and, more importantly, lose the family-owned cache finalization order. The
    trainer recognizes ``replay_granularity='trajectory'`` and invokes this
    evaluator once per replay batch. Semantic ``[sample, batch, transition]``
    axes remain intact in ``TrajectoryBatch``; only the algorithm-facing signal
    is flattened to ``[sample, action]`` for the existing masked GRPO reduction.
    """

    replay_granularity = "trajectory"
    supports_deferred_replay_tensor_move = True

    def evaluate(
        self,
        model: ReplayModel,
        batch: RolloutBatch,
        timestep_idx: int,
        ref_model: ReplayModel | None = None,
        *,
        need_ref: bool = False,
    ) -> TrajectorySignalBatch:
        del timestep_idx
        model, ref_model = self._require_models(model, ref_model)

        # Reference first: a policy standing in for its own reference swaps its
        # weights in place, which must precede the live forward's graph.
        ref_log_prob = None
        if need_ref and ref_model is not None:
            same_model = ref_model is model
            with (
                torch.no_grad(),
                model.reference_policy() if same_model else contextlib.nullcontext(),
            ):
                reference = ref_model.replay_forward(batch)
            ref_log_prob = self._flatten_actions(reference["log_probs"])

        log_prob = self._flatten_actions(model.replay_forward(batch)["log_probs"])

        reader = TrajectoryReader.from_batch(batch)
        old_log_prob = self._flatten_actions(
            reader.role_value("denoise", "old_log_prob"),
        )
        mask = self._flatten_actions(reader.role_value("denoise", "mask"))

        return TrajectorySignalBuilder(batch).single_segment(
            segment_name="denoise",
            log_prob=log_prob,
            old_log_prob=old_log_prob,
            mask=mask,
            ref_log_prob=ref_log_prob,
        )

    @staticmethod
    def _flatten_actions(value: Any) -> Any:
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) < 2:
            raise ValueError(
                "chunk-autoregressive replay values must be sample-aligned with "
                "at least one policy axis",
            )
        return value.reshape(int(shape[0]), -1)


__all__ = ["ChunkAutoregressiveDenoiseLogProbEvaluator"]
