"""Exercise the real legacy pruning contract, including repeated pruning."""

import torch
import transformers.pytorch_utils as utils

from vrl.rewards.models.videocon_compat import (
    _find_pruneable_heads_and_indices,
    prepare_videocon_imports,
)


def test_pruning_indices_skip_already_removed_heads():
    heads, index = _find_pruneable_heads_and_indices([1, 3, 3], 3, 2, {1})
    assert heads == {3}
    assert torch.equal(index, torch.tensor([0, 1, 2, 3]))
    heads, index = _find_pruneable_heads_and_indices([], 3, 2, set())
    assert heads == set()
    assert torch.equal(index, torch.arange(6))


def test_import_compat_installs_only_missing_symbol(monkeypatch):
    monkeypatch.setattr(utils, "find_pruneable_heads_and_indices", None, raising=False)
    monkeypatch.delattr(utils, "find_pruneable_heads_and_indices", raising=False)
    prepare_videocon_imports()
    assert utils.find_pruneable_heads_and_indices is _find_pruneable_heads_and_indices
    sentinel = object()
    monkeypatch.setattr(utils, "find_pruneable_heads_and_indices", sentinel)
    prepare_videocon_imports()
    assert utils.find_pruneable_heads_and_indices is sentinel
