"""The separable vocab-head contract: split declaration and payload builder."""

from __future__ import annotations

import torch

from vrl.models.steps.token.base import ARModelBase
from vrl.models.steps.token.vocab_head import VocabHeadSplit


class _Model(ARModelBase):
    """Base-default (None) unless a split is injected."""

    def __init__(self, split: VocabHeadSplit | None = None) -> None:
        super().__init__()
        self._split = split

    def vocab_head_split(self) -> VocabHeadSplit | None:
        return self._split if self._split is not None else super().vocab_head_split()


def test_none_split_falls_back_to_eager_logits() -> None:
    calls = []

    def eager():
        calls.append(1)
        return torch.zeros(2, 3, 7)

    values = _Model()._head_replay_values(torch.zeros(2, 3, 4), eager)
    assert set(values) == {"logits"}
    assert calls == [1]


def test_split_builds_fused_payload_without_calling_eager() -> None:
    weight = torch.randn(7, 4)
    bias = torch.randn(7)
    split = VocabHeadSplit(weight=weight, bias=bias, prefix=lambda h: h + 1.0)
    hidden = torch.zeros(2, 3, 4)

    def eager():
        raise AssertionError("fused path must not materialize logits")

    values = _Model(split)._head_replay_values(hidden, eager)
    assert set(values) == {"head_hidden", "head_weight", "head_bias"}
    torch.testing.assert_close(values["head_hidden"], hidden + 1.0)
    assert values["head_weight"] is weight
    assert values["head_bias"] is bias


def test_none_prefix_feeds_hidden_straight_to_the_projection() -> None:
    weight = torch.randn(7, 4)
    hidden = torch.zeros(2, 3, 4)
    values = _Model(VocabHeadSplit(weight=weight))._head_replay_values(
        hidden,
        lambda: torch.zeros(1),
    )
    assert values["head_hidden"] is hidden
    assert values["head_bias"] is None


def test_from_linear_defaults_to_no_prefix() -> None:
    import torch.nn as nn

    linear = nn.Linear(4, 7)
    split = VocabHeadSplit.from_linear(linear)
    assert split is not None
    assert split.prefix is None
    assert split.weight is linear.weight
    assert split.bias is linear.bias


def test_from_linear_rejects_subclasses_lora_guard() -> None:
    """PEFT's lora.Linear can subclass nn.Linear; .weight would drop the delta."""
    import torch.nn as nn

    class _WrappedLinear(nn.Linear):
        pass

    assert VocabHeadSplit.from_linear(_WrappedLinear(4, 7)) is None
    assert VocabHeadSplit.from_linear(nn.Sequential()) is None
    assert VocabHeadSplit.from_linear(None) is None


def test_from_linear_rows_slices_views() -> None:
    import torch.nn as nn

    linear = nn.Linear(4, 7)
    split = VocabHeadSplit.from_linear(linear, rows=5)
    assert split is not None
    assert split.weight.shape == (5, 4)
    assert split.bias.shape == (5,)
    # Views, not copies: the split must track the live parameter.
    assert split.weight.data_ptr() == linear.weight.data_ptr()
