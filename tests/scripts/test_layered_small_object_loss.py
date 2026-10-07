"""Spatial balancing prevents a sparse object from losing to empty background."""

import pytest
import torch

from vrl.scripts.generation.qwen_layered_supervised_probe import velocity_loss


def test_small_object_gradient_is_not_diluted_by_canvas_area():
    balanced_gradients = []
    for pixels in (10, 100):
        target = torch.zeros(1, pixels, 1)
        target[:, 0] = 1
        value = torch.tensor(0.25, requires_grad=True)
        prediction = value.expand_as(target)
        ordinary = velocity_loss(prediction, target)
        ordinary_gradient = torch.autograd.grad(ordinary, value, retain_graph=True)[0]
        weights = torch.full_like(target, pixels / (2 * (pixels - 1)))
        weights[:, 0] = pixels / 2
        balanced = velocity_loss(prediction, target, spatial_weights=weights)
        balanced_gradient = torch.autograd.grad(balanced, value)[0]
        # Ordinary descent suppresses the small object; balanced descent restores it.
        assert ordinary_gradient > 0
        assert balanced_gradient < 0
        balanced_gradients.append(float(balanced_gradient))
    assert balanced_gradients[0] == pytest.approx(balanced_gradients[1])
