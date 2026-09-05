"""Evaluator implementation ownership and structural protocol boundaries."""

from __future__ import annotations

import pytest

from vrl.rollouts.evaluators.base import Evaluator, ReplayEvaluatorBase
from vrl.rollouts.evaluators.denoise.chunk_autoregressive_logprob import (
    ChunkAutoregressiveDenoiseLogProbEvaluator,
)
from vrl.rollouts.evaluators.denoise.sde_logprob import DiffusionSDELogProbEvaluator
from vrl.rollouts.evaluators.token.continuous_token_logprob import (
    ContinuousTokenLogProbEvaluator,
)
from vrl.rollouts.evaluators.token.multi_segment_token_logprob import (
    MultiSegmentTokenLogProbEvaluator,
)
from vrl.rollouts.evaluators.token.token_logprob import TokenLogProbEvaluator


class _IncompleteReplayEvaluator(ReplayEvaluatorBase):
    pass


def _evaluators() -> tuple[ReplayEvaluatorBase, ...]:
    return (
        TokenLogProbEvaluator(),
        ContinuousTokenLogProbEvaluator(),
        MultiSegmentTokenLogProbEvaluator(enabled_segments=()),
        ChunkAutoregressiveDenoiseLogProbEvaluator(),
        DiffusionSDELogProbEvaluator(scheduler=object()),
    )


@pytest.mark.parametrize("evaluator", _evaluators(), ids=lambda value: type(value).__name__)
def test_concrete_evaluator_satisfies_structural_protocol(
    evaluator: ReplayEvaluatorBase,
) -> None:
    assert Evaluator not in type(evaluator).__mro__
    assert isinstance(evaluator, Evaluator)
