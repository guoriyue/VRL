from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from vrl.rewards.functions.anime_anatomy import AnimeAnatomyStructureReward
from vrl.rewards.functions.registry import MultiReward
from vrl.rewards.types import RewardRollout, RewardTrajectory


def _rollout(
    prompt: str = "single anime girl, full body, both hands visible, feet visible",
    metadata: dict[str, object] | None = None,
) -> RewardRollout:
    return RewardRollout(
        request=None,
        trajectory=RewardTrajectory(
            prompt=prompt,
            seed=0,
            steps=[],
            output=Image.new("RGB", (8, 8), color=(128, 128, 128)),
        ),
        metadata=metadata or {},
    )


def _pose_result(*, hands: bool = True, collapsed_elbow: bool = False) -> dict[str, np.ndarray]:
    keypoints = np.zeros((1, 134, 2), dtype=float)
    scores = np.zeros((1, 134), dtype=float)
    body = {
        0: (0.50, 0.10),
        1: (0.50, 0.22),
        2: (0.42, 0.25),
        3: (0.35, 0.42),
        4: (0.31, 0.58),
        5: (0.58, 0.25),
        6: (0.65, 0.42),
        7: (0.69, 0.58),
        8: (0.45, 0.55),
        9: (0.43, 0.74),
        10: (0.42, 0.92),
        11: (0.55, 0.55),
        12: (0.57, 0.74),
        13: (0.58, 0.92),
    }
    if collapsed_elbow:
        body[3] = body[2]
        body[4] = body[2]
    for idx, point in body.items():
        keypoints[0, idx] = point
        scores[0, idx] = 0.95
    for idx, point in enumerate(
        (
            (0.39, 0.93),
            (0.42, 0.96),
            (0.45, 0.93),
            (0.55, 0.93),
            (0.58, 0.96),
            (0.61, 0.93),
        ),
        start=18,
    ):
        keypoints[0, idx] = point
        scores[0, idx] = 0.9
    if hands:
        for start, center in ((92, (0.31, 0.58)), (113, (0.69, 0.58))):
            for offset in range(21):
                keypoints[0, start + offset] = (
                    center[0] + (offset % 5 - 2) * 0.01,
                    center[1] + (offset // 5) * 0.01,
                )
                scores[0, start + offset] = 0.9
    return {"keypoints": keypoints, "scores": scores}


@pytest.mark.asyncio
async def test_anime_anatomy_reward_scores_openpose_style_keypoints() -> None:
    reward = AnimeAnatomyStructureReward(
        device="cpu",
        detector=lambda images: [_pose_result() for _ in images],
    )

    scores = await reward.score_batch([_rollout()])

    assert scores[0] > 0.9


@pytest.mark.asyncio
async def test_anime_anatomy_reward_penalizes_prompt_required_missing_hands() -> None:
    reward = AnimeAnatomyStructureReward(
        device="cpu",
        detector=lambda images: [_pose_result(hands=False) for _ in images],
    )

    score = await reward.score(_rollout())

    assert score == pytest.approx(0.70)


@pytest.mark.asyncio
async def test_anime_anatomy_reward_prefers_manifest_constraints_over_prompt_guessing() -> None:
    reward = AnimeAnatomyStructureReward(
        device="cpu",
        detector=lambda images: [_pose_result(hands=False) for _ in images],
    )

    score = await reward.score(
        _rollout(
            "single anime girl, portrait, elegant boots",
            metadata={"constraints": ["both hands visible", "feet visible"]},
        ),
    )

    assert score == pytest.approx(0.70)


@pytest.mark.asyncio
async def test_anime_anatomy_reward_penalizes_degenerate_joint_geometry() -> None:
    reward = AnimeAnatomyStructureReward(
        device="cpu",
        detector=lambda images: [_pose_result(collapsed_elbow=True) for _ in images],
    )

    score = await reward.score(_rollout("single anime girl, full body, feet visible"))

    assert score < 0.95


def test_anime_anatomy_reward_is_registered() -> None:
    reward = MultiReward.from_dict(
        {"anime_anatomy_structure": 1.0},
        device="cpu",
        reward_kwargs={
            "anime_anatomy_structure": {"detector": lambda images: [_pose_result() for _ in images]},
        },
    )

    assert [name for name, _, _ in reward.rewards] == ["anime_anatomy_structure"]
