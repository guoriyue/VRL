"""A parameter without Adam state may be skipped only if it stayed unchanged."""

import json

import pytest
import torch

from vrl.scripts.eval.layered_policy_audit import audit_optimizer


def test_unconnected_parameter_must_remain_unchanged(tmp_path):
    active = torch.nn.Parameter(torch.tensor([0.2, -0.3]))
    unused = torch.nn.Parameter(torch.tensor([0.7]))
    parameters = {"image.lora_B.weight": active, "text.lora_B.weight": unused}
    initial = {name: value.detach().clone() for name, value in parameters.items()}
    optimizer = torch.optim.AdamW(parameters.values(), lr=1e-4, weight_decay=0)
    active.square().sum().backward()
    optimizer.step()
    updated = {name: value.detach().clone() for name, value in parameters.items()}
    output = tmp_path / "update01"
    output.mkdir()
    torch.save(initial, tmp_path / "initial_adapter.pt")
    torch.save(updated, output / "adapter.pt")
    torch.save(optimizer.state_dict(), output / "optimizer.pt")

    audit_optimizer(tmp_path)
    record = json.loads((output / "optimizer_audit.json").read_text())
    assert record["parameters_without_optimizer_state"] == ["text.lora_B.weight"]
    assert record["gradient_dot_parameter_step"] < 0

    updated["text.lora_B.weight"].add_(0.1)
    torch.save(updated, output / "adapter.pt")
    with pytest.raises(AssertionError, match="Parameter changed without optimizer state"):
        audit_optimizer(tmp_path)
