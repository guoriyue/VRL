from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vrl.models.steps.denoise.common.cfg import combine_cfg, predict_with_cfg


class _RecordingTransformer(nn.Module):
    """Predicts ``hidden + timestep + pooled.sum`` per row and records each call."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        hidden = kwargs["hidden_states"]
        shift = kwargs["timestep"] + kwargs["pooled_projections"].sum(dim=1)
        return (hidden + shift.unsqueeze(-1),)


def _branches() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    cond = {
        "hidden_states": torch.ones(2, 1),
        "timestep": torch.tensor([2.0, 3.0]),
        "pooled_projections": torch.ones(2, 3),
    }
    return cond, {**cond, "pooled_projections": torch.zeros(2, 3)}


def test_batched_cfg_is_one_call_on_uncond_first_rows_and_matches_separate_calls() -> None:
    """Batched CFG concatenates every tensor kwarg uncond-first into one 2B-row call and
    combines the same branch predictions that two separate calls produce.
    """
    cond, uncond = _branches()
    batched = _RecordingTransformer()
    separate = _RecordingTransformer()

    out = predict_with_cfg(
        batched, cond, uncond, guidance_scale=2.0, batched=True, dtype=torch.float32
    )
    expected = predict_with_cfg(
        separate, cond, uncond, guidance_scale=2.0, batched=False, dtype=torch.float32
    )

    assert len(batched.calls) == 1
    assert torch.equal(batched.calls[0]["pooled_projections"][:, 0], torch.tensor([0, 0, 1, 1.0]))
    assert len(separate.calls) == 2
    # cond = 1 + t + 3, uncond = 1 + t: uncond + 2 * (cond - uncond) = 7 + t.
    assert torch.equal(out, torch.tensor([[9.0], [10.0]]))
    assert torch.equal(out, expected)


def test_no_uncond_is_one_conditional_call() -> None:
    cond, _ = _branches()
    transformer = _RecordingTransformer()

    out = predict_with_cfg(
        transformer, cond, None, guidance_scale=2.0, batched=True, dtype=torch.float64
    )

    assert len(transformer.calls) == 1
    assert out.dtype == torch.float64
    assert torch.equal(out, torch.tensor([[6.0], [7.0]], dtype=torch.float64))


def test_batched_cfg_rejects_unequal_branch_rows() -> None:
    cond, uncond = _branches()
    uncond["pooled_projections"] = torch.zeros(3, 3)

    with pytest.raises(ValueError, match="pooled_projections"):
        predict_with_cfg(
            _RecordingTransformer(),
            cond,
            uncond,
            guidance_scale=2.0,
            batched=True,
            dtype=torch.float32,
        )


def test_combine_cfg_bases_and_norm_rescale() -> None:
    """The SD form ``uncond + g*(cond - uncond)``, the Cosmos form ``cond + g*(cond - uncond)``,
    and the norm-preserving rescale back to the conditional branch's norm.
    """
    cond = torch.tensor([[3.0, 4.0]])
    uncond = torch.tensor([[1.0, 0.0]])

    assert torch.equal(combine_cfg(cond, uncond, guidance_scale=2.0), torch.tensor([[5.0, 8.0]]))
    assert torch.equal(
        combine_cfg(cond, uncond, guidance_scale=2.0, base="cond"), torch.tensor([[7.0, 12.0]])
    )
    normalized = combine_cfg(cond, uncond, guidance_scale=2.0, normalize=True)
    torch.testing.assert_close(normalized.norm(dim=-1), cond.norm(dim=-1))
    torch.testing.assert_close(
        normalized / normalized.norm(), torch.tensor([[5.0, 8.0]]) / 89**0.5
    )
