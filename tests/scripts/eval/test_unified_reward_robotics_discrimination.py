from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vrl.rewards.inference import RewardInferenceResult
from vrl.scripts.eval.unified_reward_robotics_discrimination_probe import (
    _score_anchor,
    build_parser,
)
from vrl.utils.artifacts import sha256_file


def test_http_service_is_the_default_deployment() -> None:
    args = build_parser().parse_args(
        ["--manifest", "eval.jsonl", "--out", "outputs/gate.json"],
    )
    assert args.endpoint == "http://127.0.0.1:8300"
    assert args.expected_model == "unified-reward-robotics"


@pytest.mark.asyncio
async def test_anchor_is_materialized_as_one_integrity_checked_batch(tmp_path: Path) -> None:
    class FakeScorer:
        request = None

        async def score_batch(self, request):
            self.request = request
            raw = {"alignment": 4.0, "physics": 3.5, "style": 3.0, "overall": 3.5}
            return [
                RewardInferenceResult(
                    artifact_id=artifact.artifact_id,
                    scores=raw,
                )
                for artifact in request.artifacts
            ]

    scorer = FakeScorer()
    candidates = {
        "exact": torch.rand(4, 8, 8, 3),
        "static_frozen": torch.rand(1, 8, 8, 3).repeat(4, 1, 1, 1),
    }
    scores = await _score_anchor(
        scorer,
        candidates,
        anchor_index=0,
        prompt="Move the block into the tray",
        fps=15.0,
        candidate_root=tmp_path,
    )

    assert len(scorer.request.artifacts) == 2
    assert set(scores) == set(candidates)
    for artifact in scorer.request.artifacts:
        path = Path(artifact.path)
        assert path.is_absolute()
        assert artifact.size_bytes == path.stat().st_size
        assert artifact.sha256 == sha256_file(path)
        assert artifact.prompt == "Move the block into the tray"
