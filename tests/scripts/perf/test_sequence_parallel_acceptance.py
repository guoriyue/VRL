"""CPU-side contract of the sequence-parallel acceptance script: the fleet
overrides it composes and the dump comparison verdict. The GPU launch path is
the hardware gate itself and is exercised on the multi-GPU host."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from vrl.scripts.perf import sequence_parallel_acceptance as sp


def _dump(path: Path, output: torch.Tensor, *, gpus_per_engine: int, seed: int = 7) -> Path:
    path.mkdir(parents=True)
    torch.save(
        {
            "output": output,
            "prompts": ["a", "b"],
            "seed": seed,
            "gpus_per_engine": gpus_per_engine,
        },
        path / sp.OUTPUT_TENSOR_FILE,
    )
    return path


def _compare_args(reference: Path, candidate: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "reference": reference,
        "candidate": candidate,
        "atol": 0.02,
        "max_mismatch_fraction": 0.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_fleet_overrides_pin_topology_seed_and_eager_policy() -> None:
    args = argparse.Namespace(
        rollout_devices=[1, 2],
        trainer_device=0,
        gpus_per_engine=2,
        seed=11,
        overrides=["sampling.num_steps=4"],
    )
    overrides = sp._fleet_overrides(args)
    assert "distributed.resources.trainer.devices=[0]" in overrides
    assert "distributed.resources.rollout.devices=[1,2]" in overrides
    assert "distributed.resources.rollout.gpus_per_engine=2" in overrides
    assert "model.torch_compile.enable=false" in overrides
    assert "sampling.seed=11" in overrides
    # Caller overrides apply last so they win over the pinned defaults.
    assert overrides[-1] == "sampling.num_steps=4"


def test_compare_passes_within_tolerance_and_scales_uint8(tmp_path: Path, capsys) -> None:
    reference = _dump(
        tmp_path / "n1",
        torch.full((2, 3, 4, 4), 128, dtype=torch.uint8),
        gpus_per_engine=1,
    )
    candidate = _dump(
        tmp_path / "n2",
        torch.full((2, 3, 4, 4), 130, dtype=torch.uint8),
        gpus_per_engine=2,
    )
    result = sp._compare(_compare_args(reference, candidate))
    assert result["passed"] is True
    assert result["max_abs_diff"] == pytest.approx(2 / 255)
    assert result["mismatch_fraction"] == 0.0
    assert result["candidate_gpus_per_engine"] == 2


def test_compare_fails_beyond_tolerance(tmp_path: Path, capsys) -> None:
    reference = _dump(tmp_path / "n1", torch.zeros(1, 3, 2, 2), gpus_per_engine=1)
    candidate = _dump(tmp_path / "n2", torch.full((1, 3, 2, 2), 0.5), gpus_per_engine=2)
    result = sp._compare(_compare_args(reference, candidate))
    assert result["passed"] is False
    assert result["mismatch_fraction"] == 1.0
    with pytest.raises(SystemExit):
        sp.main(["compare", str(reference), str(candidate)])


def test_compare_rejects_mismatched_inputs(tmp_path: Path, capsys) -> None:
    reference = _dump(tmp_path / "n1", torch.zeros(1, 3, 2, 2), gpus_per_engine=1, seed=1)
    candidate = _dump(tmp_path / "n2", torch.zeros(1, 3, 2, 2), gpus_per_engine=2, seed=2)
    with pytest.raises(ValueError, match="seed"):
        sp._compare(_compare_args(reference, candidate))
