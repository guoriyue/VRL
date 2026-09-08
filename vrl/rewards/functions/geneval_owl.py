"""GenEval compositional reward binding, scored in-process with OWLv2 + CLIP.

Registered as ``geneval_owl``. The scoring logic lives in
``vrl.rewards.models.geneval_owl``; the runtime builds it from the
``model_factory`` dotted path so a reward sharing the trainer GPU can park its
pages at each phase handoff (CuMem sleep/wake), exactly like ``pickscore``.

``score_key`` selects which of the model's three readings drives training.
``geneval_owl_dense`` is the default because GRPO consumes the ordering of
rewards inside a prompt group and the verdict-derived readings barely order
anything: measured on 8 prompts x 16 samples, ``geneval_owl_partial`` took 2-4
distinct values per group and the policy did not move over 10 updates, while a
continuous reward on the identical setup moved +7.7 SEM
(SPRINT_anima_geneval_spatial_rl 7.3-7.5).
"""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import CumemRewardFunction
from vrl.rewards.runtime import build_reward_scorer

GENEVAL_SCORE_KEYS = ("geneval_owl_dense", "geneval_owl_partial", "geneval_owl_strict")


class GenEvalOwlReward(CumemRewardFunction):
    """Continuous GenEval condition satisfaction in ``[0, 1]`` (dense by default)."""

    def __init__(
        self,
        device: str = "cuda",
        score_key: str = "geneval_owl_dense",
        **kwargs: Any,
    ) -> None:
        if score_key not in GENEVAL_SCORE_KEYS:
            raise ValueError(
                f"geneval_owl score_key must be one of {list(GENEVAL_SCORE_KEYS)}, got {score_key!r}"
            )
        super().__init__(
            reward_name="geneval_owl",
            score_key=score_key,
            scorer=build_reward_scorer(
                {
                    "device": device,
                    **kwargs,
                    "model_factory": "vrl.rewards.models.geneval_owl:GenEvalOwlRewardModel",
                },
            ),
        )


__all__ = ["GENEVAL_SCORE_KEYS", "GenEvalOwlReward"]
