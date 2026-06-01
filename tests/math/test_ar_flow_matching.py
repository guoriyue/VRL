"""Velocity-contract + sample/replay consistency for NextStep-1 flow-matching.

Guards the contract verified against upstream ``FlowMatchingHead``: the velocity
predictor is ``image_head.net(x, t, c)`` and the head's ``forward`` is the
training-loss path, NOT velocity. ``_FakeHead.__call__`` raises, so any code that
reaches for ``image_head(...)`` instead of ``.net`` fails loudly here.
"""

from __future__ import annotations

import torch

from vrl.math.ar.flow_matching import flow_logprob_at, flow_sample_with_logprob


class _FakeHead:
    """Minimal stand-in for NextStep-1 FlowMatchingHead.

    Exposes ``.net(x, t, c)`` (a deterministic linear velocity) and
    ``input_dim``; calling the head directly raises so the logprob paths must
    use ``.net``.
    """

    def __init__(self, input_dim: int) -> None:
        self.input_dim = input_dim

    def net(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Deterministic velocity field of the right shape ([B, input_dim]).
        return 0.1 * x

    def __call__(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError(
            "flow-matching logprob must call image_head.net, not the head's "
            "forward (which is the training-loss path)"
        )


def test_sample_and_replay_go_through_net_and_agree() -> None:
    b, token_dim, hidden = 2, 4, 6
    head = _FakeHead(token_dim)
    cond = torch.randn(b, hidden)
    gen = torch.Generator().manual_seed(0)

    token, log_prob, x0 = flow_sample_with_logprob(
        head, cond, num_flow_steps=5, generator=gen
    )

    assert token.shape == (b, token_dim)
    assert log_prob.shape == (b,)
    assert x0.shape == (b, token_dim)
    assert torch.isfinite(log_prob).all()

    # Replaying the same token with the saved prior and the same (unchanged)
    # velocity field must reproduce the exact collection-time log-prob — this is
    # the GRPO ratio == 1 invariant the divergent velocity resolver would break.
    replay_log_prob = flow_logprob_at(
        head, cond, token, saved_noise=x0, num_flow_steps=5
    )

    assert replay_log_prob.shape == (b,)
    assert torch.isfinite(replay_log_prob).all()
    assert torch.allclose(log_prob, replay_log_prob, atol=1e-5)
