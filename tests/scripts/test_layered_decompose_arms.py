"""The decomposition evaluator's sampler and arm selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vrl.scripts.eval.layered_decompose_eval import adapter_overrides, heun_decompose


@dataclass
class _State:
    latents: torch.Tensor
    timesteps: torch.Tensor
    scheduler: object


def test_heun_integrates_a_constant_velocity_exactly_over_the_sigma_grid() -> None:
    steps = 5
    sigmas = torch.linspace(1.0, 0.0, steps + 1)
    velocity = torch.full((1, 4, 2), 3.0)
    calls: list[int] = []

    def forward_step(state, index):
        calls.append(index)
        return {"noise_pred": velocity.to(torch.bfloat16)}

    model = SimpleNamespace(forward_step=forward_step)
    x0 = torch.zeros(1, 4, 2)
    state = _State(
        latents=x0.clone(), timesteps=sigmas[:-1] * 1000, scheduler=SimpleNamespace(sigmas=sigmas)
    )

    latents = heun_decompose(model, state)

    # sum of dt over the grid is -1, so the sample moves by -velocity.
    torch.testing.assert_close(latents, x0 - velocity)
    # A corrector forward at every step but the last (sigma 0 has no second evaluation).
    assert calls == [0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_base_arm_disables_the_adapter() -> None:
    assert adapter_overrides(None, base=True) == ["model.use_lora=false"]


def test_adapter_arm_takes_a_peft_directory_or_a_checkpoint_directory(tmp_path: Path) -> None:
    peft_dir = tmp_path / "peft"
    peft_dir.mkdir()
    (peft_dir / "adapter_config.json").write_text("{}")
    checkpoint = tmp_path / "checkpoint-500"
    (checkpoint / "lora_weights").mkdir(parents=True)
    (checkpoint / "lora_weights" / "adapter_config.json").write_text("{}")

    assert adapter_overrides(peft_dir, base=False) == [
        "model.use_lora=true",
        f"model.lora.path={peft_dir}",
    ]
    assert adapter_overrides(checkpoint, base=False) == [
        "model.use_lora=true",
        f"model.lora.path={checkpoint / 'lora_weights'}",
    ]
    with pytest.raises(FileNotFoundError, match="PEFT adapter directory"):
        adapter_overrides(tmp_path, base=False)


def test_an_arm_is_exactly_one_of_adapter_or_base() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        adapter_overrides(None, base=False)
