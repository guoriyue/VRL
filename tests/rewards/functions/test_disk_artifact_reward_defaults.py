"""Constructor defaults owned by concrete disk-artifact rewards."""

from __future__ import annotations

import pytest

from vrl.rewards.functions.videoscore2 import VideoScore2Reward


class _Runtime:
    scoring_is_nonblocking = False
    external_accelerator_isolation_verified = False

    async def score_batch(self, request):
        return []

    async def shutdown(self) -> None:
        return None


def test_explicit_empty_request_identity_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="score_key"):
        VideoScore2Reward(
            reward_name="videoscore2",
            score_key="",
            artifact_dir=str(tmp_path),
            scorer=_Runtime(),
        )
