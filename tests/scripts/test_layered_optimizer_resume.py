"""Restored layered Adam state reproduces uninterrupted optimization."""

import hashlib
import json

import pytest
import torch

from vrl.scripts.generation.qwen_layered_rl_update import restore_optimizer


def test_resume_preserves_moments_and_rejects_wrong_policy_or_parameter_order(tmp_path):
    parameters = {
        "image": torch.nn.Parameter(torch.tensor([0.2, -0.3])),
        "text": torch.nn.Parameter(torch.tensor([0.4, 0.1])),
        "unused": torch.nn.Parameter(torch.tensor([0.7])),
    }
    uninterrupted = torch.optim.AdamW(parameters.values(), lr=0.01, weight_decay=0)
    for target in (0.0, 1.0, -0.5):
        uninterrupted.zero_grad(set_to_none=True)
        ((parameters["image"] - target).square().sum() + parameters["text"].sum()).backward()
        uninterrupted.step()
    torch.save(
        {name: value.detach().clone() for name, value in parameters.items()},
        tmp_path / "adapter.pt",
    )
    torch.save(uninterrupted.state_dict(), tmp_path / "optimizer.pt")
    adapter_hash = hashlib.sha256((tmp_path / "adapter.pt").read_bytes()).hexdigest()
    optimizer_hash = hashlib.sha256((tmp_path / "optimizer.pt").read_bytes()).hexdigest()
    (tmp_path / "result.json").write_text(
        json.dumps({"adapter_sha256": adapter_hash, "optimizer_updates": 3})
    )
    resumed_parameters = {
        name: torch.nn.Parameter(value.detach().clone()) for name, value in parameters.items()
    }
    resumed = torch.optim.AdamW(resumed_parameters.values(), lr=0.01, weight_decay=0)
    restore_optimizer(
        resumed,
        resumed_parameters,
        tmp_path,
        adapter_sha256=adapter_hash,
        optimizer_sha256=optimizer_hash,
    )
    # New-batch gradients change direction. Resetting Adam would diverge here.
    for target in (-2.0, 0.7):
        for optimizer, values in ((uninterrupted, parameters), (resumed, resumed_parameters)):
            optimizer.zero_grad(set_to_none=True)
            ((values["image"] - target).square().sum() - values["text"].sum()).backward()
            optimizer.step()
    for name in parameters:
        assert torch.equal(parameters[name], resumed_parameters[name])
    for index in uninterrupted.state_dict()["state"]:
        for key, value in uninterrupted.state_dict()["state"][index].items():
            assert torch.equal(value, resumed.state_dict()["state"][index][key])

    with pytest.raises(ValueError, match="initial adapter"):
        restore_optimizer(
            resumed,
            resumed_parameters,
            tmp_path,
            adapter_sha256="another-policy",
            optimizer_sha256=optimizer_hash,
        )
    # Equal-shaped parameters are still unsafe to swap: Adam uses positional IDs.
    swapped = {name: resumed_parameters[name] for name in ("text", "image", "unused")}
    with pytest.raises(ValueError, match="parameter order"):
        restore_optimizer(
            resumed,
            swapped,
            tmp_path,
            adapter_sha256=adapter_hash,
            optimizer_sha256=optimizer_hash,
        )
    with pytest.raises(ValueError, match="checkpoint hash"):
        restore_optimizer(
            resumed,
            resumed_parameters,
            tmp_path,
            adapter_sha256=adapter_hash,
            optimizer_sha256="another-optimizer",
        )
