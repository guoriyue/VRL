"""Grouped replay signals for chunk-autoregressive denoise policies."""

from __future__ import annotations

import contextlib
from typing import Any

from vrl.models.interfaces import ReplayModel, ReplayRequest, require_replay_model
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.base import Evaluator
from vrl.rollouts.evaluators.trajectory import TrajectorySignalBuilder
from vrl.rollouts.evaluators.types import SignalRequest, TrajectorySignalBatch
from vrl.trajectory import TrajectoryResolver


class ChunkAutoregressiveDenoiseLogProbEvaluator(Evaluator):
    """Replay a complete causal-chunk trajectory in one ordered model pass.

    A scalar denoise-step replay would repeatedly rebuild the temporal prefix
    and, more importantly, lose the family-owned cache finalization order. The
    trainer recognizes ``replay_granularity='trajectory'`` and invokes this
    evaluator once per replay batch. Semantic ``[sample, chunk, transition]``
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
        signal_request: SignalRequest | None = None,
    ) -> TrajectorySignalBatch:
        del timestep_idx
        model = require_replay_model(
            model,
            owner="ChunkAutoregressiveDenoiseLogProbEvaluator.model",
        )
        if signal_request is None:
            signal_request = SignalRequest()
        if ref_model is not None:
            ref_model = require_replay_model(
                ref_model,
                owner="ChunkAutoregressiveDenoiseLogProbEvaluator.ref_model",
            )

        request = ReplayRequest(segment_names=("denoise",))
        current = model.replay_forward(batch, request=request).require_segment("denoise")
        log_prob = self._flatten_actions(current.require_value("log_probs"))

        resolver = TrajectoryResolver.from_batch(batch)
        old_log_prob = self._flatten_actions(
            resolver.role_value("denoise", "old_log_prob"),
        )
        mask = self._flatten_actions(resolver.role_value("denoise", "mask"))

        ref_log_prob = None
        if signal_request.need_ref and ref_model is not None:
            same_model = ref_model is model
            with model.disable_adapter() if same_model else contextlib.nullcontext():
                reference = ref_model.replay_forward(
                    batch,
                    request=request,
                ).require_segment("denoise")
            ref_log_prob = self._flatten_actions(reference.require_value("log_probs"))

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
