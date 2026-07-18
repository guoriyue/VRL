"""Token-level log-probability evaluator for AR replay models.

Wraps ``model.replay_forward`` (which returns full logits) and gathers the
log-probabilities of the *sampled* tokens — both under the current model
and, when needed, under the LoRA-off reference model.

The returned ``TrajectorySignalBatch`` is consumed by the trainer/algorithm
adapter.

Distribution family is ``"categorical"`` — flow-matching latent-space KL
intermediates are unused and stay ``None``.
"""

from __future__ import annotations

import torch

from vrl.math.token.logprob import gather_categorical_log_probs
from vrl.models.interfaces import ReplayModel, require_replay_model
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.base import Evaluator
from vrl.rollouts.evaluators.trajectory import TrajectorySignalBuilder
from vrl.rollouts.evaluators.types import SignalRequest, TrajectorySignalBatch


class TokenLogProbEvaluator(Evaluator):
    """Recompute per-token log-probs of sampled tokens under the replay model.

    Two-pass when ``need_ref=True``:
      1. forward through ``model`` with the LoRA adapter ON  → log_prob
      2. forward through ``ref_model`` if provided, otherwise through
         ``model`` inside ``ReplayModel.disable_adapter()`` → ref_log_prob

    ``disable_adapter()`` is part of the trainer-facing ReplayModel contract. If a
    model has no adapter, it may return ``contextlib.nullcontext()``; callers
    that need a distinct frozen reference must pass ``ref_model``.
    """

    def __init__(self, mask_key: str = "token_mask") -> None:
        self.mask_key = mask_key

    def evaluate(
        self,
        model: ReplayModel,
        batch: RolloutBatch,
        timestep_idx: int = 0,
        ref_model: ReplayModel | None = None,
        signal_request: SignalRequest | None = None,
    ) -> TrajectorySignalBatch:
        model = require_replay_model(model, owner="TokenLogProbEvaluator.model")
        if ref_model is not None:
            ref_model = require_replay_model(ref_model, owner="TokenLogProbEvaluator.ref_model")
        request = signal_request or SignalRequest()
        action_ids: torch.Tensor = batch.actions  # [B, L_img]

        builder = TrajectorySignalBuilder(batch)
        # Rollout scoring divides logits by the sampling temperature; replay
        # must renormalize with the recorded temperature for old/new parity.
        temperature = float(builder.context.get("temperature", 1.0))

        new_lp = self._compute_logprobs(
            model,
            batch,
            action_ids,
            temperature,
        )

        ref_lp = None
        if request.need_ref:
            if ref_model is not None:
                ref_lp = self._compute_logprobs(
                    ref_model,
                    batch,
                    action_ids,
                    temperature,
                )
            else:
                with torch.no_grad(), model.disable_adapter():
                    ref_lp = self._compute_logprobs(
                        model,
                        batch,
                        action_ids,
                        temperature,
                    )

        return builder.single_segment(
            segment_name="image_tokens",
            log_prob=new_lp,
            old_log_prob=None,
            ref_log_prob=ref_lp,
            distribution="categorical",
            timestep_idx=timestep_idx,
            mask_key=self.mask_key,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_logprobs(
        model: ReplayModel,
        batch: RolloutBatch,
        action_ids: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Forward + gather. Always returns ``[B, L]`` float32 log-probs."""
        out = model.replay_forward(batch, timestep_idx=0)
        result = out.require_segment("image_tokens")
        logits: torch.Tensor = result.require_value("logits")  # [B, L, V_img]
        return gather_categorical_log_probs(
            logits,
            action_ids,
            temperature=temperature,
        )  # [B, L]
