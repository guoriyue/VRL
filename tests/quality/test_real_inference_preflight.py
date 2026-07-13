"""Opt-in artifact test run before launching a costly training experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.quality.contracts import InferencePhase
from tests.quality.preflight import evaluate_inference_preflight
from vrl.config.loading import load_config


@pytest.mark.quality_gate
def test_real_inference_matches_native_reference(request: pytest.FixtureRequest) -> None:
    """Fail on degenerate, mismatched, reward-hacked, or replay-divergent evidence."""

    config_name = request.config.getoption("--quality-config")
    evidence_path = request.config.getoption("--quality-evidence")
    if not config_name or not evidence_path:
        pytest.skip("pass --quality-config and --quality-evidence to run the real preflight")
    checkpoint = request.config.getoption("--quality-checkpoint")
    phase = InferencePhase.CHECKPOINT if checkpoint else InferencePhase.BASE
    report = evaluate_inference_preflight(
        load_config(config_name),
        Path(evidence_path),
        phase=phase,
        checkpoint_file=Path(checkpoint) if checkpoint else None,
    )
    failures = [item for item in report["assertions"] if not item["passed"]]
    assert report["verdict"] == "pass", json.dumps(failures, indent=2, sort_keys=True)
