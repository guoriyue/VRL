"""GenEval compositional reward binding, scored in-process with OWLv2 + CLIP.

Registered as ``geneval_owl``. The scoring logic lives in
``vrl.rewards.models.geneval_owl``; the runtime builds it from the
``model_factory`` dotted path so a reward sharing the trainer GPU can park its
pages at each phase handoff (CuMem sleep/wake), exactly like ``pickscore``.
"""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import CumemRewardFunction
from vrl.rewards.runtime import build_reward_scorer


class GenEvalOwlReward(CumemRewardFunction):
    """Satisfied-condition fraction of the prompt's GenEval spec in ``[0, 1]``."""

    def __init__(self, device: str = "cuda", **kwargs: Any) -> None:
        super().__init__(
            reward_name="geneval_owl",
            score_key="geneval_owl",
            scorer=build_reward_scorer(
                {
                    "device": device,
                    **kwargs,
                    "model_factory": "vrl.rewards.models.geneval_owl:GenEvalOwlRewardModel",
                },
            ),
        )


__all__ = ["GenEvalOwlReward"]
