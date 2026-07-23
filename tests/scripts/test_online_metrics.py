from __future__ import annotations

import csv
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from vrl.scripts.common.online import (
    OnlineRecipeRun,
    _export_transformer_lora,
    _prepare_metrics_csv_rank_consistent,
)
from vrl.trainers.checkpointing import AdapterExport
from vrl.trainers.distributed import DistributedTrainingContext


def _context(*, distributed: bool, primary: bool) -> DistributedTrainingContext:
    return DistributedTrainingContext(
        strategy="ddp" if distributed else "single_process",
        rank=0 if primary else 1,
        local_rank=0,
        world_size=2 if distributed else 1,
        device=torch.device("cpu"),
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_dual_transformer_lora_export_is_namespaced() -> None:
    high = SimpleNamespace(save_pretrained=lambda _path: None)
    low = SimpleNamespace(save_pretrained=lambda _path: None)
    bundle = SimpleNamespace(
        trainable_modules={"transformer": high, "transformer_2": low},
    )
    cfg = OmegaConf.create({"model": {"use_lora": True}})

    exported = _export_transformer_lora(bundle, cfg)

    assert exported == {
        "lora_weights/transformer": AdapterExport(high),
        "lora_weights/transformer_2": AdapterExport(low),
    }


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
        component_names=(),
        adapter_exports=None,
        csv_path=tmp_path / "metrics.csv",
        rng=object(),
        resume_epoch=None,
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

        def _prepare() -> None:
            if rank != 0:
                raise AssertionError("only rank 0 may prepare the metrics CSV")
            if fail:
                raise ValueError("different metrics schema")
            Path(marker_path).write_text("prepared\n", encoding="utf-8")

        try:
            _prepare_metrics_csv_rank_consistent(
                SimpleNamespace(prepare_metrics_csv=_prepare),
                _context(distributed=True, primary=rank == 0),
            )
        except RuntimeError as exc:
            queue.put((rank, str(exc)))
        else:
            queue.put((rank, "ok"))
    finally:
        dist.destroy_process_group()


def test_metrics_csv_preflight_preserves_single_process_error() -> None:
    def _raise_schema_error() -> None:
        raise ValueError("different metrics schema")

    run = SimpleNamespace(prepare_metrics_csv=_raise_schema_error)

    with pytest.raises(ValueError, match="different metrics schema"):
        _prepare_metrics_csv_rank_consistent(
            run,
            _context(distributed=False, primary=True),
        )


def test_online_resume_rejects_changed_reward_component_schema(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    OnlineRecipeRun(
        bundle=None,
        trainer=None,
        strategy=None,
        family="unit",
        component_names=("aesthetic",),
        adapter_exports=None,
        csv_path=path,
        rng=None,
        resume_epoch=None,
        model_identity={"schema": "test"},
    ).prepare_metrics_csv()

    resumed = OnlineRecipeRun(
        bundle=None,
        trainer=None,
        strategy=None,
        family="unit",
        component_names=("aesthetic", "pickscore"),
        adapter_exports=None,
        csv_path=path,
        rng=None,
        resume_epoch=0,
        model_identity={"schema": "test"},
    )

    with pytest.raises(ValueError, match="different metrics schema"):
        resumed.prepare_metrics_csv()


def test_metrics_csv_writes_continuous_request_diagnostics(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    run = OnlineRecipeRun(
        bundle=None,
        trainer=None,
        strategy=None,
        family="unit",
        component_names=(),
        adapter_exports=None,
        csv_path=path,
        rng=None,
        resume_epoch=None,
        model_identity={"schema": "test"},
    )
    run.prepare_metrics_csv()
    update = SimpleNamespace(
        clip_fraction=0.0,
        active_clip_fraction=0.0,
        tis_clip_fraction=0.0,
        rs_seq_masked_fraction=0.0,
        approx_kl=0.0,
    )
    initial = SimpleNamespace(
        clip_fraction=0.0,
        active_clip_fraction=0.0,
        logprob_abs_diff_max=0.0,
    )
    mismatch = SimpleNamespace(
        logprob_abs_diff_mean=0.0,
        logprob_abs_diff_max=0.0,
        ratio_abs_dev_mean=0.0,
        ratio_abs_dev_max=0.0,
        mismatch_kl=0.0,
        mismatch_k3_kl=0.0,
    )
    metrics = SimpleNamespace(
        loss=0.0,
        policy_loss=0.0,
        sft_loss=0.0,
        kl_penalty=0.0,
        weighted_kl_loss=0.0,
        reward_mean=1.0,
        reward_std=0.0,
        update=update,
        initial_replay=initial,
        logprob_mismatch=mismatch,
        advantage_mean=0.0,
        grad_norm=1.0,
        adv_saturation=0.0,
        adv_zero_rate=0.0,
        group_size=4.0,
        trained_prompt_num=1,
        reward_components={},
        phase_times={
            "continuous.ready_groups_at_demand": 1.0,
            "continuous.lookahead_requested": 1.0,
            "continuous.producer_submitted": 2.0,
            "continuous.producer_completed": 1.0,
        },
    )

    run.write_metric_row(0, metrics)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert None not in rows[0]
    assert rows[0]["continuous_ready_groups_at_demand"] == "1.0"
    assert rows[0]["continuous_lookahead_requested"] == "1.0"
    assert rows[0]["continuous_producer_submitted"] == "2.0"
    assert rows[0]["continuous_producer_completed"] == "1.0"


def test_metrics_csv_preflight_broadcasts_primary_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _broadcast(payload, *, src, device) -> None:
        assert payload == ["ValueError: different metrics schema"]
        assert src == 0
        assert device == torch.device("cpu")

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", _broadcast)

    def _raise_schema_error() -> None:
        raise ValueError("different metrics schema")

    run = SimpleNamespace(prepare_metrics_csv=_raise_schema_error)

    with pytest.raises(RuntimeError, match="rank 0: ValueError: different metrics schema"):
        _prepare_metrics_csv_rank_consistent(
            run,
            _context(distributed=True, primary=True),
        )


def test_metrics_csv_preflight_peer_uses_broadcast_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _broadcast(payload, *, src, device) -> None:
        del src, device
        payload[0] = "ValueError: different metrics schema"

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", _broadcast)
    prepared = False

    def _prepare() -> None:
        nonlocal prepared
        prepared = True

    with pytest.raises(RuntimeError, match="rank 0: ValueError: different metrics schema"):
        _prepare_metrics_csv_rank_consistent(
            SimpleNamespace(prepare_metrics_csv=_prepare),
            _context(distributed=True, primary=False),
        )

    assert prepared is False


@pytest.mark.parametrize("fail", [False, True])
def test_metrics_csv_preflight_is_rank_consistent_with_real_gloo(tmp_path, fail) -> None:
    context = mp.get_context("spawn")
    queue = context.Queue()
    marker = tmp_path / "prepared.txt"
    port = _free_port()
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
