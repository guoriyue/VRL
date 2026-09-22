from __future__ import annotations

import csv
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.trainers._strategy_policies import free_port
from vrl.algorithms.types import TrainStepMetrics
from vrl.scripts.common.online import OnlineRecipeRun
from vrl.trainers.distributed import DistributedTrainingContext
from vrl.trainers.metrics_io import OnlineMetricsCSV


def _context(*, distributed: bool, primary: bool) -> DistributedTrainingContext:
    return DistributedTrainingContext(
        strategy="ddp" if distributed else "single_process",
        rank=0 if primary else 1,
        world_size=2 if distributed else 1,
        device=torch.device("cpu"),
    )


def test_online_checkpoint_threads_required_model_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    identity = {"schema": "test"}
    calls: list[dict[str, object]] = []
    run = OnlineRecipeRun(
        bundle=object(),
        trainer=SimpleNamespace(state=SimpleNamespace(global_step=7)),
        strategy=SimpleNamespace(
            context=SimpleNamespace(is_primary=True),
        ),
        family="unit",
        adapter_exports=None,
        rng=object(),
        model_identity=identity,
    )
    monkeypatch.setattr(
        "vrl.scripts.common.online.capture_rng_state",
        lambda *, prompt_generator: {"prompt_generator": prompt_generator},
    )
    monkeypatch.setattr(
        "vrl.scripts.common.online.save_training_checkpoint",
        lambda path, **kwargs: calls.append({"path": path, **kwargs}),
    )

    run.save_checkpoint(tmp_path / "checkpoint-3", epoch=3)

    assert calls[0]["model_identity"] is identity
    assert calls[0]["family"] == "unit"
    assert calls[0]["progress"] == {
        "completed_epoch": 3,
        "next_epoch": 3,
        "next_step": 7,
        "global_step": 7,
    }


def _run_metrics_preflight_rank(
    rank: int,
    world_size: int,
    port: int,
    marker_path: str,
    fail: bool,
    queue: mp.Queue,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:

        def _prepare(*args, **kwargs):
            if rank != 0:
                raise AssertionError("only rank 0 may prepare the metrics CSV")
            if fail:
                raise ValueError("different metrics schema")
            Path(marker_path).write_text("prepared\n", encoding="utf-8")
            return object()

        try:
            from unittest.mock import patch

            run = SimpleNamespace(metrics_csv=None)
            with patch("vrl.scripts.common.online.OnlineMetricsCSV", _prepare):
                OnlineRecipeRun.initialize_metrics(
                    run,
                    _context(distributed=True, primary=rank == 0),
                    Path(marker_path).parent,
                    component_names=(),
                    resume_epoch=None,
                )
            assert (run.metrics_csv is not None) == (rank == 0)
        except RuntimeError as exc:
            queue.put((rank, str(exc)))
        else:
            queue.put((rank, "ok"))
    finally:
        dist.destroy_process_group()


def test_metrics_csv_preflight_preserves_single_process_error(monkeypatch, tmp_path) -> None:
    def fail(*args, **kwargs):
        raise ValueError("different metrics schema")

    monkeypatch.setattr("vrl.scripts.common.online.OnlineMetricsCSV", fail)
    with pytest.raises(ValueError, match="different metrics schema"):
        OnlineRecipeRun.initialize_metrics(
            SimpleNamespace(metrics_csv=None),
            _context(distributed=False, primary=True),
            tmp_path,
            component_names=(),
            resume_epoch=None,
        )


def test_online_resume_rejects_changed_reward_component_schema(tmp_path) -> None:
    OnlineMetricsCSV(tmp_path, component_names=("aesthetic",))
    with pytest.raises(ValueError, match="different metrics schema"):
        OnlineMetricsCSV(tmp_path, component_names=("aesthetic", "pickscore"), resume_epoch=0)


def test_metrics_csv_initializes_both_headers_before_any_rows(tmp_path) -> None:
    output_dir = tmp_path / "new_run"
    OnlineMetricsCSV(output_dir, component_names=("ocr",))
    for name in ("metrics.csv", "metrics.full_precision.csv"):
        with (output_dir / name).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert len(rows) == 1
        assert rows[0][0] == "epoch"
        assert rows[0][-1] == "r_ocr"
        assert "loss" in rows[0]


def test_metrics_csv_writes_continuous_request_diagnostics(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    run = OnlineMetricsCSV(tmp_path)
    metrics = TrainStepMetrics(
        phase_times={
            "continuous.ready_groups_at_demand": 1.0,
            "continuous.lookahead_requested": 1.0,
            "continuous.producer_submitted": 2.0,
            "continuous.producer_completed": 1.0,
        },
    )

    run.append(0, metrics)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert None not in rows[0]
    assert rows[0]["continuous_ready_groups_at_demand"] == "1.0"
    assert rows[0]["continuous_lookahead_requested"] == "1.0"
    assert rows[0]["continuous_producer_submitted"] == "2.0"
    assert rows[0]["continuous_producer_completed"] == "1.0"


@pytest.mark.parametrize("fail", [False, True])
def test_metrics_csv_preflight_is_rank_consistent_with_real_gloo(tmp_path, fail) -> None:
    context = mp.get_context("spawn")
    queue = context.Queue()
    marker = tmp_path / "prepared.txt"
    port = free_port()
    processes = [
        context.Process(
            target=_run_metrics_preflight_rank,
            args=(rank, 2, port, str(marker), fail, queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    results = sorted(queue.get(timeout=10) for _ in processes)
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    if fail:
        assert all(
            "rank 0: ValueError: different metrics schema" in result for _, result in results
        )
        assert not marker.exists()
    else:
        assert results == [(0, "ok"), (1, "ok")]
        assert marker.read_text(encoding="utf-8") == "prepared\n"


def test_full_precision_metrics_follow_online_write_and_resume(tmp_path) -> None:
    import csv

    run = OnlineMetricsCSV(tmp_path, component_names=("ocr",))
    for epoch in range(3):
        run.append(
            epoch,
            TrainStepMetrics(loss=0.123456789 + epoch, reward_components={"ocr": 0.987654321}),
        )
    path = tmp_path / "metrics.full_precision.csv"
    rows = list(csv.DictReader(path.open()))
    assert float(rows[0]["loss"]) == 0.123456789
    assert float(rows[0]["r_ocr"]) == 0.987654321
    run = OnlineMetricsCSV(tmp_path, component_names=("ocr",), resume_epoch=2)
    run.append(2, TrainStepMetrics(loss=0.111111111, reward_components={"ocr": 0.222222222}))
    rows = list(csv.DictReader(path.open()))
    assert [row["epoch"] for row in rows] == ["0", "1", "2"]
    assert float(rows[-1]["loss"]) == 0.111111111


def test_reward_observation_axes_survive_fixed_csv_schema_and_checkpoint_resume(tmp_path):
    import json

    run = OnlineMetricsCSV(tmp_path, component_names=("judge",))
    for epoch in range(3):
        run.append(
            epoch,
            TrainStepMetrics(
                reward_components={
                    "judge": 0.8,
                    "judge/alignment": 0.123456789 + epoch,
                    **({"judge/new_axis": 0.0} if epoch == 2 else {}),
                }
            ),
        )
    path = tmp_path / "reward_components.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["components"]["judge/alignment"] == 0.123456789
    assert "judge/new_axis" not in records[0]["components"]
    assert records[2]["components"]["judge/new_axis"] == 0.0
    with path.open("a") as handle:
        handle.write('{"incomplete":')
    resumed = OnlineMetricsCSV(tmp_path, component_names=("judge",), resume_epoch=2)
    resumed.append(2, TrainStepMetrics(reward_components={"judge/alignment": 0.9}))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["epoch"] for record in records] == [0, 1, 2]
    assert records[2]["components"] == {"judge/alignment": 0.9}
    with pytest.raises(ValueError, match="Out of range"):
        resumed.append(3, TrainStepMetrics(reward_components={"judge/hidden": float("nan")}))
    assert len(path.read_text().splitlines()) == 3
