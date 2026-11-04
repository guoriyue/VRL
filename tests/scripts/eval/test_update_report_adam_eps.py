"""The update report tells stalled optimizer updates from ones that move the weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from vrl.scripts.eval.update_report import build_report
from vrl.trainers.checkpointing import TRAINING_CHECKPOINT_NAME, write_checkpoint_meta
from vrl.trainers.optimizer import FP32MasterWeightOptimizer


def _save(checkpoint_dir: Path, *, step: int, layer: torch.nn.Module, trainer: dict[str, Any]):
    checkpoint_dir.mkdir(parents=True)
    identity = {"schema": "test"}
    progress = {"global_step": step}
    trainer = {"step": step, "global_step": step, **trainer}
    torch.save(
        {
            "schema_version": 2,
            "family": "toy",
            "model": {"identity": identity, "owned_state": {"policy": layer.state_dict()}},
            "trainer": trainer,
            "progress": progress,
        },
        checkpoint_dir / TRAINING_CHECKPOINT_NAME,
    )
    write_checkpoint_meta(
        checkpoint_dir,
        family="toy",
        model_identity=identity,
        trainer_state=trainer,
        progress=progress,
        uses_lora=False,
    )


def _train_adamw(run_dir: Path, *, eps: float) -> None:
    """Gradients near 1e-6, checkpoints after steps 2 and 5."""

    torch.manual_seed(0)
    layer = torch.nn.Linear(16, 16)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3, eps=eps, weight_decay=0.0)
    for step in range(1, 6):
        optimizer.zero_grad()
        (layer(torch.randn(8, 16)).sum() * 1e-6).backward()
        optimizer.step()
        if step in (2, 5):
            trainer = {"optimizer": optimizer.state_dict()}
            _save(run_dir / f"checkpoint-{step}", step=step, layer=layer, trainer=trainer)


def test_eps_above_root_second_moment_shows_as_a_stalled_update(tmp_path: Path) -> None:
    _train_adamw(tmp_path / "healthy", eps=1e-12)
    _train_adamw(tmp_path / "stalled", eps=1e-2)

    healthy, stalled = build_report([tmp_path / "healthy", tmp_path / "stalled"])["series"]

    assert [c["global_step"] for c in healthy["checkpoints"]] == [2, 5]
    healthy_adam = healthy["checkpoints"][-1]["adam"]
    stalled_adam = stalled["checkpoints"][-1]["adam"]
    assert healthy_adam["eps_dominated"] == 0.0
    assert stalled_adam["eps_dominated"] == 1.0
    assert healthy_adam["next_step_to_lr"] > 0.1
    assert stalled_adam["next_step_to_lr"] < 1e-3
    assert stalled_adam["eps_damping"] < 1e-3

    (healthy_move,) = healthy["intervals"]
    (stalled_move,) = stalled["intervals"]
    assert healthy_move["optimizer_steps"] == 3
    assert healthy_move["net_update_to_lr"] > 0.1
    assert stalled_move["net_update_to_lr"] < 1e-3
    assert stalled_move["relative_change"] < 1e-5 < healthy_move["relative_change"]


def test_sub_ulp_updates_of_a_bf16_model_show_through_its_fp32_masters(tmp_path: Path) -> None:
    torch.manual_seed(0)
    layer = torch.nn.Linear(16, 16, bias=False).to(torch.bfloat16)
    optimizer = FP32MasterWeightOptimizer(
        layer.parameters(), lambda params: torch.optim.SGD(params, lr=1e-6)
    )
    # A DDP-wrapped policy names its parameters ``<root>.module.*``; the owned
    # state the checkpoint exports does not.
    manifest = [{"name": "policy.module.weight", "shape": [16, 16], "dtype": "torch.bfloat16"}]
    snapshots = []
    for step in (1, 2):
        optimizer.zero_grad()
        layer(torch.randn(4, 16, dtype=torch.bfloat16)).float().square().sum().backward()
        optimizer.prepare_gradients()
        optimizer.step()
        snapshots.append(layer.weight.detach().clone())
        trainer = {"optimizer": optimizer.state_dict(), "optimizer_parameter_manifest": manifest}
        _save(tmp_path / f"checkpoint-{step}", step=step, layer=layer, trainer=trainer)

    ((interval,),) = (series["intervals"] for series in build_report([tmp_path])["series"])

    assert torch.equal(snapshots[0], snapshots[1])
    assert interval["optimizer_steps"] == 1
    assert interval["unmatched_tensors"] == 0
    assert interval["changed_fraction"] > 0.9
    assert interval["relative_change"] > 0
