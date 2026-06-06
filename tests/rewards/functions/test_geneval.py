from __future__ import annotations

import pytest

from vrl.rewards.functions.geneval import GenEvalReward
from vrl.rewards.functions.registry import MultiReward
from vrl.rewards.types import RewardRollout, RewardTrajectory


def _rollout(metadata: dict) -> RewardRollout:
    return RewardRollout(
        request=None,
        trajectory=RewardTrajectory(
            prompt="a photo of a yellow bus", seed=0, steps=[], output=None
        ),
        metadata=metadata,
    )


@pytest.mark.asyncio
async def test_geneval_reward_uses_injected_scorer_metadata() -> None:
    def scorer(**kwargs):
        assert kwargs["geneval"]["tag"] == "colors"
        assert kwargs["geneval"]["include"][0]["class"] == "bus"
        return {"score": 0.75}

    reward = GenEvalReward(device="cpu", scorer=scorer)
    score = await reward.score(
        _rollout(
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
async def test_geneval_reward_reads_manifest_row_metadata() -> None:
    def scorer(**kwargs):
        assert kwargs["geneval"]["tag"] == "single_object"
        return 0.5

    reward = GenEvalReward(device="cpu", scorer=scorer)
    score = await reward.score(
        _rollout(
            {
                "manifest_row": {
                    "metadata": {
                        "geneval": {
                            "tag": "single_object",
                            "include": [{"class": "bench", "count": 1}],
                        },
                    },
                },
            },
        ),
    )

    assert score == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_geneval_reward_requires_metadata() -> None:
    with pytest.raises(ValueError, match=r"metadata\.geneval"):
        GenEvalReward._extract_geneval_metadata(_rollout({}))


@pytest.mark.asyncio
async def test_geneval_reward_rejects_unsupported_evaluator() -> None:
    reward = GenEvalReward(device="cpu", evaluator="constant", scorer=lambda **_: 0.25)

    with pytest.raises(ValueError, match="unsupported GenEval evaluator"):
        await reward.score(
            _rollout(
                {
                    "geneval": {
                        "tag": "colors",
                        "include": [{"class": "bus", "count": 1, "color": "yellow"}],
                    },
                },
            ),
        )


@pytest.mark.asyncio
async def test_geneval_reward_registered_in_multi_reward() -> None:
    def scorer(**kwargs):
        assert kwargs["geneval"]["tag"] == "colors"
        return 0.25

    reward = MultiReward.from_dict(
        {"geneval": 1.0},
        device="cpu",
        reward_kwargs={"geneval": {"scorer": scorer}},
    )

    scores = await reward.score_batch(
        [
            _rollout(
                {
                    "geneval": {
                        "tag": "colors",
                        "include": [{"class": "bus", "count": 1, "color": "yellow"}],
                    },
                },
            ),
        ],
    )

    assert scores == pytest.approx([0.25])
