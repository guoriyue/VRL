from __future__ import annotations

import pytest

from vrl.rewards.functions.geneval import GenEvalReward
from vrl.rewards.functions.registry import MultiReward
from vrl.rewards.types import RewardSample


def _sample(metadata: dict) -> RewardSample:
    return RewardSample(
        prompt="a photo of a yellow bus",
        output=None,
        sample_id="sample-0",
        metadata=metadata,
    )


@pytest.mark.asyncio
async def test_geneval_reward_uses_injected_scorer_metadata() -> None:
    """Checks GenEval reward uses injected scorer metadata."""

    def scorer(**kwargs):
        assert kwargs["geneval"]["tag"] == "colors"
        assert kwargs["geneval"]["include"][0]["class"] == "bus"
        return {"score": 0.75}

    reward = GenEvalReward(device="cpu", scorer=scorer)
    score = await reward.score(
        _sample(
            {
                "geneval": {
                    "tag": "colors",
                    "include": [{"class": "bus", "count": 1, "color": "yellow"}],
                },
            },
        ),
    )

    assert score == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_geneval_reward_requires_metadata() -> None:
    """Checks GenEval reward requires metadata."""
    with pytest.raises(ValueError, match=r"metadata\.geneval"):
        GenEvalReward._extract_geneval_metadata(_sample({}))


@pytest.mark.asyncio
async def test_geneval_reward_rejects_unknown_kwarg() -> None:
    """The removed ``evaluator`` knob (and any typo) fails loud, not silently.

    GenEvalReward has an explicit signature (no catch-all): an unknown
    reward.kwargs key is a config typo and must raise at construction rather
    than be silently swallowed.
    """
    with pytest.raises(TypeError):
        GenEvalReward(device="cpu", evaluator="constant", scorer=lambda **_: 0.25)


def test_geneval_reward_rejects_removed_timeout_knob() -> None:
    with pytest.raises(TypeError, match="timeout_s"):
        GenEvalReward(device="cpu", timeout_s=0.0, scorer=lambda **_: 0.25)


@pytest.mark.asyncio
async def test_geneval_reward_registered_in_multi_reward() -> None:
    """Checks GenEval reward registered in multi reward."""

    def scorer(**kwargs):
        assert kwargs["geneval"]["tag"] == "colors"
        return 0.25

    reward = MultiReward.from_dict(
        {"geneval": 1.0},
        device="cpu",
        reward_kwargs={"geneval": {"scorer": scorer}},
    )

    output = await reward.score_batch(
        [
            _sample(
                {
                    "geneval": {
                        "tag": "colors",
                        "include": [{"class": "bus", "count": 1, "color": "yellow"}],
                    },
                },
            ),
        ],
    )

    assert output.scores == pytest.approx([0.25])
