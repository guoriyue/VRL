"""Constructor defaults owned by concrete disk-artifact rewards."""

from __future__ import annotations

import pytest

from vrl.rewards.functions.unified_reward_video import UnifiedRewardVideoReward


class _Runtime:
    scoring_is_nonblocking = False
    external_accelerator_isolation_verified = False

    async def score_batch(self, request):
        return []

    async def shutdown(self) -> None:
        return None


def test_explicit_empty_request_identity_is_rejected() -> None:
    with pytest.raises(ValueError, match="score_key"):
        UnifiedRewardVideoReward(
            reward_name="unified_reward_video",
            score_key="",
            scorer=_Runtime(),
        )
