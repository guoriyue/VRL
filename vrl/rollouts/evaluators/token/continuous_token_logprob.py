"""Continuous-token log-probability evaluator for NextStep-1.

Drop-in replacement for ``TokenLogProbEvaluator`` when tokens are
continuous (NextStep-1 family). The LLM produces hidden states; the
flow-matching head turns those into per-token Gaussian log-probabilities
of the *previously sampled* continuous tokens.

Distribution family is ``"gaussian"`` — algorithms see the same
``[B, L]`` log-prob tensor as the categorical path, so ``TokenGRPO``
needs no change.
"""

from __future__ import annotations

import torch

from vrl.models.interfaces import ReplayModel, require_replay_model
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.base import Evaluator
from vrl.rollouts.evaluators.trajectory import TrajectorySignalBuilder
from vrl.rollouts.evaluators.types import SignalRequest, TrajectorySignalBatch


class ContinuousTokenLogProbEvaluator(Evaluator):
    """Recompute Gaussian log-probs of sampled continuous tokens.

    Two-pass when ``need_ref=True`` — use ``ref_model`` when provided, otherwise
    use the model's required ``disable_adapter()`` context.
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
        model = require_replay_model(model, owner="ContinuousTokenLogProbEvaluator.model")
        if ref_model is not None:
            ref_model = require_replay_model(
                ref_model,
                owner="ContinuousTokenLogProbEvaluator.ref_model",
            )
        request = signal_request or SignalRequest()

        new_lp = self._compute_logprobs(model, batch)

        ref_lp: torch.Tensor | None = None
        if request.need_ref:
            if ref_model is not None:
                ref_lp = self._compute_logprobs(ref_model, batch)
            else:
                with torch.no_grad(), model.disable_adapter():
                    ref_lp = self._compute_logprobs(model, batch)

        return TrajectorySignalBuilder(batch).single_segment(
            segment_name="image_tokens",
            log_prob=new_lp,
            ref_log_prob=ref_lp,
            distribution="gaussian",
            timestep_idx=timestep_idx,
            mask_key=self.mask_key,
        )

    @staticmethod
    def _compute_logprobs(
        model: ReplayModel,
        batch: RolloutBatch,
    ) -> torch.Tensor:
        """Forward through ``model.replay_forward`` — return ``[B, L]`` float32 log-probs.

        The model's ``replay_forward`` re-primes the LLM with text, replays
        the AR loop with stashed noise, and re-evaluates the flow head's
        Gaussian density at the sampled tokens. This evaluator just unwraps
        the returned ``log_probs`` and casts to float32.
        """
        out = model.replay_forward(batch, timestep_idx=0)
        result = out.require_segment("image_tokens")
        log_probs: torch.Tensor = result.require_value("log_probs")  # [B, L]
        return log_probs.float()
