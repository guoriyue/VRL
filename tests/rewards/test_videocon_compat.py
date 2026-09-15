"""Exercise the real legacy pruning contract, including repeated pruning."""

import torch
import transformers.pytorch_utils as utils

from vrl.rewards.models.videocon_compat import (
    _find_pruneable_heads_and_indices,
    prepare_videocon_imports,
    prepare_videocon_model_class,
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


def test_vendor_head_mask_contract():
    class Vendor:
        dtype = torch.bfloat16

    prepare_videocon_model_class(Vendor)
    model = Vendor()
    assert model.get_head_mask(None, 2) == [None, None]
    mask = model.get_head_mask(torch.tensor([1, 0, 1]), 2)
    assert mask.shape == (2, 1, 3, 1, 1)
    assert mask.dtype == torch.bfloat16
    assert torch.equal(mask[:, 0, :, 0, 0], torch.tensor([[1, 0, 1], [1, 0, 1]]))
    layered = torch.tensor([[1, 0], [0, 1]])
    mask = model.get_head_mask(layered, 2, True)
    assert mask.shape == (2, 1, 2, 1, 1, 1)
    assert torch.equal(mask[:, 0, :, 0, 0, 0], layered)
    existing = Vendor.get_head_mask
    prepare_videocon_model_class(Vendor)
    assert Vendor.get_head_mask is existing
