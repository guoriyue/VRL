"""CPU tests for the UnifiedReward robotics anti-reward-hacking gate."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vrl.rewards.inference import RewardInferenceResult, sha256_file
from vrl.scripts.eval.unified_reward_robotics_discrimination_probe import (
    _score_anchor,
    _validated_scores,
    aggregate_axis_scores,
    build_discrimination_candidates,
    build_parser,
    robotics_reward_verdict,
)


def test_probe_builds_full_candidate_battery() -> None:
    frames = torch.rand(8, 16, 16, 3)
    other = torch.rand(8, 16, 16, 3)
    candidates = build_discrimination_candidates(frames, other, seed=0)
    assert {
        "exact",
        "perceptual_blur",
        "temporal_mean",
        "static_frozen",
        "frame_shuffle",
        "reverse",
        "random",
        "color_blob",
        "wrong_clip",
    } <= set(candidates)
    # static_frozen / temporal_mean are constant over time by construction.
    assert torch.allclose(candidates["static_frozen"][0], candidates["static_frozen"][-1])
    assert torch.allclose(candidates["temporal_mean"][0], candidates["temporal_mean"][-1])
    assert torch.equal(candidates["exact"], frames)
    assert candidates["color_blob"].shape == frames.shape
    assert candidates["color_blob"].amin() >= 0.0
    assert candidates["color_blob"].amax() <= 1.0


def _candidate(alignment: float, physics: float) -> dict[str, float]:
    return {
        "alignment": alignment,
        "physics": physics,
        "alignment+physics": alignment + physics,
    }


def _passing_anchor() -> dict[str, dict[str, float]]:
    return {
        "exact": _candidate(4.2, 4.0),
        "perceptual_blur": _candidate(2.0, 1.8),
        "temporal_mean": _candidate(2.0, 1.8),
        "static_frozen": _candidate(2.1, 2.0),
        "random": _candidate(1.0, 1.0),
        "color_blob": _candidate(1.5, 1.5),
        "frame_shuffle": _candidate(3.8, 2.7),
        "reverse": _candidate(3.6, 2.8),
        "wrong_clip": _candidate(2.8, 3.5),
    }


def _verdict(anchors: list[dict[str, dict[str, float]]]) -> dict:
    return robotics_reward_verdict(anchors, aggregate_axis_scores(anchors))


def test_robotics_gate_passes_discriminating_axes() -> None:
    anchors = [_passing_anchor() for _ in range(4)]
    verdict = _verdict(anchors)
    assert verdict["passed"] is True
    assert verdict["metrics"]["exact_alignment+physics"] == pytest.approx(8.2)
    assert all(check["passed"] for check in verdict["checks"])


@pytest.mark.parametrize(
    ("candidate", "axis", "value", "failed_check"),
    [
        ("static_frozen", "alignment", 3.0, "low_information_axes_ceiling"),
        ("color_blob", "physics", 3.0, "low_information_axes_ceiling"),
        ("wrong_clip", "alignment", 3.9, "wrong_clip_alignment_gap"),
        ("frame_shuffle", "physics", 3.8, "frame_shuffle_physics_gap"),
    ],
)
def test_robotics_gate_rejects_reward_hack_leaks(
    candidate: str,
    axis: str,
    value: float,
    failed_check: str,
) -> None:
    anchors = [_passing_anchor() for _ in range(4)]
    for anchor in anchors:
        anchor[candidate] = _candidate(
            value if axis == "alignment" else anchor[candidate]["alignment"],
            value if axis == "physics" else anchor[candidate]["physics"],
        )
    verdict = _verdict(anchors)
    checks = {check["name"]: check for check in verdict["checks"]}
    assert verdict["passed"] is False
    assert checks[failed_check]["passed"] is False


def test_robotics_gate_requires_most_individual_anchors_to_pass() -> None:
    anchors = [_passing_anchor() for _ in range(4)]
    for anchor in anchors[:2]:
        anchor["wrong_clip"] = _candidate(4.0, 3.5)
    verdict = _verdict(anchors)
    checks = {check["name"]: check for check in verdict["checks"]}
    assert verdict["passed"] is False
    assert checks["anchor_pass_rate"]["passed"] is False


def test_robotics_gate_checks_the_actual_training_compound() -> None:
    anchors = [_passing_anchor() for _ in range(4)]
    for anchor in anchors:
        # Alignment still drops enough, but a spuriously high physics score makes
        # this wrong clip nearly tie exact on the score training would optimize.
        anchor["wrong_clip"] = _candidate(3.0, 5.0)
    verdict = _verdict(anchors)
    checks = {check["name"]: check for check in verdict["checks"]}
    assert verdict["passed"] is False
    assert checks["wrong_clip_alignment_gap"]["passed"] is True
    assert checks["exact_adversary_compound_gap"]["passed"] is False


def test_validated_scores_derives_training_compound() -> None:
    assert _validated_scores({"alignment": 3.25, "physics": 2.75, "overall": 9.0}) == {
        "alignment": 3.25,
        "physics": 2.75,
        "alignment+physics": 6.0,
    }
    with pytest.raises(ValueError, match=r"in \[1, 5\]"):
        _validated_scores({"alignment": 5.1, "physics": 2.0})


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
