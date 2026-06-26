"""PI0.5 flow-logprob contract (P3) — replay-exactness + differentiability.

Pins the RL-eligibility property the probe establishes: every train-mode flow
noise method (flow_sde/flow_cps/flow_noise) yields a per-step Gaussian transition
whose summed log-prob is (a) replay-exact -- recomputing it on the stored
trajectory reproduces the collection-time value bit-for-bit (the RL old_log_prob
contract) -- and (b) differentiable into the policy. This is what makes PI0.5
RL-eligible where OpenVLA-OFT's deterministic L1 head is not.
"""

from __future__ import annotations

import pytest

from vrl.scripts.eval.pi05_flow_logprob_probe import (
    _METHODS,
    _probe_eval_mode_degenerate,
    _probe_method,
)


@pytest.mark.parametrize("method", _METHODS)
def test_flow_method_is_replayable_and_differentiable(method: str) -> None:
    result = _probe_method(method, seed=0)
    assert result["replayable"] is True
    assert result["replay_max_abs_diff"] == 0.0
    assert result["grad_flows_into_policy"] is True
    assert result["std_is_nonzero"] is True


def test_eval_mode_has_no_usable_logprob() -> None:
    """Deterministic eval mode (std=0) cannot form a Normal log-prob."""
    assert _probe_eval_mode_degenerate()["logprob_usable"] is False
