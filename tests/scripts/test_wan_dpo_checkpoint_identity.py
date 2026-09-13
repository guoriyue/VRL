"""Wan DPO checkpoint identity must gate and follow the whole training run.

Every test drives ``train_wan_2_1_dpo`` on the real tiny snapshot: the identity
is the real local-directory hash, the checkpoint is a real ``checkpoint.pt``,
and resume goes through the real validate -> restore -> save chain.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tests.scripts._wan_dpo_helpers import install_local_pickapic, spy_wan_from_build, tiny_dpo_run
from tests.scripts.eval.fixtures import write_tiny_wan_snapshot
from vrl.scripts.families.wan_2_1.train_dpo import train_wan_2_1_dpo
from vrl.trainers.checkpointing import TrainingCheckpoint, read_checkpoint_meta


def _metric_steps(run: Path) -> list[str]:
    with (run / "metrics.csv").open(encoding="utf-8") as handle:
        return [row["step"] for row in csv.DictReader(handle)]


def test_matching_identity_restores_progress_and_keeps_saving_it(monkeypatch, tmp_path) -> None:
    install_local_pickapic(monkeypatch)
    train_wan_2_1_dpo(tiny_dpo_run(tmp_path))
    run = tmp_path / "run"
    first = read_checkpoint_meta(run / "checkpoint-1")
    assert first["model_identity"]["sources"]  # the real local snapshot hash

    train_wan_2_1_dpo(
        tiny_dpo_run(
            tmp_path,
            overrides=[
                f"trainer.resume_from={run / 'checkpoint-1'}",
                "trainer.max_train_steps=2",
            ],
        ),
    )

    # The resumed run logged only step 1 and its checkpoints continue the
    # restored trainer: global_step 2 is impossible for a fresh trainer that ran
    # one step, so it proves the restore threaded through the optimizer state.
    assert _metric_steps(run) == ["0", "1"]
    for name in ("checkpoint-2", "checkpoint-final"):
        checkpoint = TrainingCheckpoint.load(run / name)
        assert checkpoint.next_step == 2
        assert checkpoint.payload["trainer"]["global_step"] == 2
        assert checkpoint.meta["model_identity"] == first["model_identity"]


def test_identity_mismatch_stops_before_model_and_dataset(monkeypatch, tmp_path) -> None:
    install_local_pickapic(monkeypatch)
    train_wan_2_1_dpo(tiny_dpo_run(tmp_path))
    checkpoint = tmp_path / "run" / "checkpoint-1"
    other = write_tiny_wan_snapshot(tmp_path / "other-snapshot")
    (other / "provenance.txt").write_text("a different model directory", encoding="utf-8")
    dataset_calls = install_local_pickapic(monkeypatch)
    built = spy_wan_from_build(monkeypatch)

    with pytest.raises(ValueError, match="model identity mismatch"):
        train_wan_2_1_dpo(
            tiny_dpo_run(
                tmp_path,
                overrides=[f"model.path={other}", f"trainer.resume_from={checkpoint}"],
            ),
        )

    assert built == []
    assert dataset_calls == []


def test_source_change_during_model_load_stops_before_dataset(monkeypatch, tmp_path) -> None:
    install_local_pickapic(monkeypatch)
    train_wan_2_1_dpo(tiny_dpo_run(tmp_path))
    checkpoint = tmp_path / "run" / "checkpoint-1"
    snapshot = tmp_path / "wan-snapshot"
    dataset_calls = install_local_pickapic(monkeypatch)

    def drift(_model) -> None:
        (snapshot / "extra-weights.bin").write_bytes(b"drift")

    built = spy_wan_from_build(monkeypatch, on_built=drift)

    with pytest.raises(
        RuntimeError,
        match="model checkpoint source changed during Wan DPO bundle construction",
    ):
        train_wan_2_1_dpo(tiny_dpo_run(tmp_path, overrides=[f"trainer.resume_from={checkpoint}"]))

    assert len(built) == 1
    assert dataset_calls == []
