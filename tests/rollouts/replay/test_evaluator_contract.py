"""Every evaluator is an ``Evaluator``, and one without ``evaluate`` cannot be built."""

from __future__ import annotations

import pytest

from vrl.rollouts.evaluators.base import Evaluator
from vrl.rollouts.evaluators.denoise.chunk_autoregressive_logprob import (
    ChunkAutoregressiveDenoiseLogProbEvaluator,
)
from vrl.rollouts.evaluators.denoise.sde_logprob import DenoiseSDELogProbEvaluator


class _IncompleteEvaluator(Evaluator):
    pass


@pytest.mark.parametrize(
    "evaluator",
    [
        ChunkAutoregressiveDenoiseLogProbEvaluator(),
        DenoiseSDELogProbEvaluator(scheduler=object()),
    ],
    ids=lambda value: type(value).__name__,
)
def test_concrete_evaluators_declare_their_replay_shape(evaluator: Evaluator) -> None:
    assert isinstance(evaluator, Evaluator)
    assert evaluator.replay_granularity in {"step", "trajectory"}


def test_evaluator_without_evaluate_cannot_be_built() -> None:
    with pytest.raises(TypeError, match="evaluate"):
        _IncompleteEvaluator()  # type: ignore[abstract]
