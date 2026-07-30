"""Run supervisor tests — real subprocesses, real process groups, no fakes."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
import threading
import time
from dataclasses import fields
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from vrl.algorithms.types import InitialReplayStats, TrainStepMetrics
from vrl.scripts.supervise import (
    HEALTH_VERDICT_NAME,
    ContinuousHealthPolicy,
    HealthGateConfig,
    MetricsHealthGate,
    RunSupervisor,
    build_parser,
    build_train_launch,
)
from vrl.trainers.core.types import (
    ContinuousRolloutConfig,
    DebugConfig,
    RolloutOrchestrationConfig,
)
from vrl.trainers.metrics_io import (
    build_online_metric_row,
    format_online_metric_row,
    online_metric_columns,
)

_METRICS_HEADER = "epoch,loss,reward_mean,reward_std,grad_norm,pre_update_logprob_abs_diff_max\n"


def _metric_row(
    epoch: int,
    *,
    loss: str = "1.0",
    reward_mean: str = "2.0",
    reward_std: str = "0.2",
    grad_norm: str = "0.1",
    parity: str = "0.001",
) -> str:
    return f"{epoch},{loss},{reward_mean},{reward_std},{grad_norm},{parity}\n"


def _continuous_metric_row(
    epoch: int,
    *,
    parity: str = "0.02",
    stale_versions: str = "1",
    producer_errors: str = "0",
) -> dict[str, str]:
    return {
        "epoch": str(epoch),
        "loss": "1.0",
        "reward_mean": "2.0",
        "reward_std": "0.2",
        "grad_norm": "0.1",
        "pre_update_logprob_abs_diff_max": parity,
        "continuous_stale_versions": stale_versions,
        "continuous_producer_errors": producer_errors,
    }


def _continuous_metric_csv_row(
    epoch: int,
    *,
    producer_errors: str = "0",
) -> str:
    row = _continuous_metric_row(epoch, producer_errors=producer_errors)
    return ",".join(row.values()) + "\n"


def _write_continuous_metrics(
    out: Path,
    rows: list[dict[str, str]],
) -> None:
    header = ",".join(rows[0]) + "\n"
    body = "".join(",".join(row.values()) + "\n" for row in rows)
    (out / "metrics.csv").write_text(header + body)


def _child_script(tmp_path: Path, body: str) -> list[str]:
    script = tmp_path / "child.py"
    script.write_text(body)
    return [sys.executable, str(script)]


def _torchrun_command(script: Path, *, workers: int = 2) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={workers}",
        "--max-restarts=0",
        str(script),
    ]


def _write_verdict_snippet(out_dir: Path) -> str:
    return (
        "import json, pathlib\n"
        f"out = pathlib.Path({str(out_dir)!r})\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
    )


def test_success_first_attempt(tmp_path) -> None:
    out = tmp_path / "run"
    command = _child_script(
        tmp_path,
        _write_verdict_snippet(out)
        + "(out / 'run_verdict.json').write_text(json.dumps({'verdict': 'success'}))\n",
    )
    supervisor = RunSupervisor(command=command, output_dir=out, sleep=lambda _: None)
    assert supervisor.run() == 0


def test_same_cause_circuit_breaker_stops(tmp_path) -> None:
    out = tmp_path / "run"
    attempts_file = tmp_path / "attempts"
    command = _child_script(
        tmp_path,
        _write_verdict_snippet(out)
        + f"attempts = pathlib.Path({str(attempts_file)!r})\n"
        + "n = int(attempts.read_text()) + 1 if attempts.exists() else 1\n"
        + "attempts.write_text(str(n))\n"
        + "(out / 'run_verdict.json').write_text(json.dumps("
        "{'verdict': 'failed', 'error_class': 'ValueError'}))\n" + "raise SystemExit(1)\n",
    )
    supervisor = RunSupervisor(
        command=command,
        output_dir=out,
        same_cause_limit=2,
        sleep=lambda _: None,
    )
    assert supervisor.run() == 1
    assert attempts_file.read_text() == "2"  # stopped at the breaker, not later


def test_transient_failure_then_success_restarts(tmp_path) -> None:
    out = tmp_path / "run"
    marker = tmp_path / "failed_once"
    command = _child_script(
        tmp_path,
        _write_verdict_snippet(out)
        + f"marker = pathlib.Path({str(marker)!r})\n"
        + "if not marker.exists():\n"
        + "    marker.write_text('1')\n"
        + "    (out / 'run_verdict.json').write_text(json.dumps("
        "{'verdict': 'terminated', 'signal': 15, 'signal_name': 'SIGTERM'}))\n"
        + "    raise SystemExit(143)\n"
        + "(out / 'run_verdict.json').write_text(json.dumps({'verdict': 'success'}))\n",
    )
    supervisor = RunSupervisor(command=command, output_dir=out, sleep=lambda _: None)
    assert supervisor.run() == 0


def test_missing_verdict_is_a_distinct_failure_class(tmp_path) -> None:
    out = tmp_path / "run"
    # Child dies without unwinding: no verdict file is ever written.
    command = _child_script(tmp_path, "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n")
    supervisor = RunSupervisor(
        command=command,
        output_dir=out,
        same_cause_limit=2,
        sleep=lambda _: None,
    )
    exit_code = supervisor.run()
    assert exit_code != 0  # breaker tripped on consecutive no-verdict deaths


def test_restart_resumes_from_latest_complete_checkpoint(tmp_path) -> None:
    import torch.nn as nn

    from vrl.trainers.checkpointing import save_training_checkpoint

    out = tmp_path / "run"
    out.mkdir(parents=True)

    class _Trainer:
        def state_dict(self):
            return {"step": 4, "global_step": 4}

    class _Bundle:
        def __init__(self) -> None:
            self.trainable_modules = {"module": nn.Linear(1, 1, bias=False)}

    save_training_checkpoint(
        out / "checkpoint-4",
        trainer=_Trainer(),
        bundle=_Bundle(),
        family="unit",
        progress={"next_epoch": 4, "global_step": 4},
        rng_state={},
        model_identity={"schema": "test"},
    )

    attempts_file = tmp_path / "attempts"
    argv_log = tmp_path / "argv.log"
    command = _child_script(
        tmp_path,
        "import sys\n"
        + _write_verdict_snippet(out)
        + f"attempts = pathlib.Path({str(attempts_file)!r})\n"
        + "n = int(attempts.read_text()) + 1 if attempts.exists() else 1\n"
        + "attempts.write_text(str(n))\n"
        + "if n == 1:\n"
        + "    (out / 'run_verdict.json').write_text(json.dumps("
        + "{'verdict': 'failed', 'error_class': 'TransientRuntimeError'}))\n"
        + "    raise SystemExit(1)\n"
        + f"pathlib.Path({str(argv_log)!r}).write_text('\\n'.join(sys.argv[1:]))\n"
        + "(out / 'run_verdict.json').write_text(json.dumps({'verdict': 'success'}))\n",
    )
    supervisor = RunSupervisor(command=command, output_dir=out, sleep=lambda _: None)
    assert supervisor.run() == 0

    assert attempts_file.read_text() == "2"
    argv = argv_log.read_text().splitlines()
    assert f"trainer.resume_from={out / 'checkpoint-4'}" in argv
    assert "model.lora.path=" in argv  # warm-start adapter cleared on resume


def test_torchrun_failure_restarts_all_ranks_from_complete_checkpoint(tmp_path) -> None:
    out = tmp_path / "run"
    worker = tmp_path / "restart_worker.py"
    worker.write_text(
        "import json, os, pathlib, sys, time\n"
        "from vrl.scripts.train import write_run_verdict\n"
        f"out = pathlib.Path({str(out)!r})\n"
        "rank = int(os.environ['RANK'])\n"
        "resume = next((arg for arg in sys.argv[1:] "
        "if arg.startswith('trainer.resume_from=')), '')\n"
        "if not resume:\n"
        "    checkpoint = out / 'checkpoint-4'\n"
        "    if rank == 0:\n"
        "        checkpoint.mkdir(parents=True, exist_ok=True)\n"
        "        payload = b'complete-checkpoint'\n"
        "        (checkpoint / 'checkpoint.pt').write_bytes(payload)\n"
        "        meta = {'schema_version': 1, 'checkpoint_file_bytes': len(payload), "
        "'global_step': 4, 'next_epoch': 4}\n"
        "        (checkpoint / 'checkpoint_meta.json').write_text(json.dumps(meta))\n"
        "        write_run_verdict(str(out))\n"
        "        raise SystemExit(0)\n"
        "    for _ in range(100):\n"
        "        if (checkpoint / 'checkpoint_meta.json').exists():\n"
        "            break\n"
        "        time.sleep(0.02)\n"
        "    write_run_verdict(str(out), error=ValueError('transient rank failure'))\n"
        "    raise SystemExit(1)\n"
        "(out / f'resume-rank-{rank}.txt').write_text('\\n'.join(sys.argv[1:]))\n"
        "write_run_verdict(str(out))\n",
    )
    supervisor = RunSupervisor(
        command=[*_torchrun_command(worker), str(out)],
        output_dir=out,
        expected_world_size=2,
        max_attempts=2,
        sleep=lambda _: None,
    )

    assert supervisor.run() == 0
    checkpoint = out / "checkpoint-4"
    for rank in range(2):
        argv = (out / f"resume-rank-{rank}.txt").read_text().splitlines()
        assert f"trainer.resume_from={checkpoint}" in argv
        assert "model.lora.path=" in argv
    aggregate = json.loads((out / "run_verdict.json").read_text())
    assert aggregate["verdict"] == "success"
    assert len(aggregate["rank_verdicts"]) == 2


def test_stop_kills_whole_child_process_group(tmp_path) -> None:
    out = tmp_path / "run"
    grandchild_pid_file = tmp_path / "grandchild.pid"
    # Child spawns a grandchild in the SAME group, then sleeps forever.
    command = _child_script(
        tmp_path,
        "import pathlib, subprocess, sys, time\n"
        + "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        + f"pathlib.Path({str(grandchild_pid_file)!r}).write_text(str(p.pid))\n"
        + "time.sleep(600)\n",
    )
    supervisor = RunSupervisor(
        command=command,
        output_dir=out,
        term_grace_seconds=3.0,
        sleep=lambda _: None,
    )
    result: list[int] = []
    runner = threading.Thread(target=lambda: result.append(supervisor.run()))
    runner.start()
    deadline = time.monotonic() + 10
    while not grandchild_pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert grandchild_pid_file.exists(), "child never started its grandchild"
    grandchild_pid = int(grandchild_pid_file.read_text())

    supervisor.request_stop(signal.SIGTERM)
    runner.join(timeout=10)
    assert not runner.is_alive()
    assert result and result[0] != 0

    # The grandchild must be gone too (whole process group was signaled).
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(grandchild_pid, signal.SIGKILL)
        raise AssertionError("grandchild survived supervisor stop")


def test_stop_kills_torchrun_workers(tmp_path) -> None:
    out = tmp_path / "run"
    worker = tmp_path / "sleep_worker.py"
    worker.write_text(
        "import os, pathlib, time\n"
        f"out = pathlib.Path({str(out)!r})\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "rank = int(os.environ['RANK'])\n"
        "(out / f'worker-{rank}.pid').write_text(str(os.getpid()))\n"
        "time.sleep(600)\n",
    )
    supervisor = RunSupervisor(
        command=_torchrun_command(worker),
        output_dir=out,
        expected_world_size=2,
        term_grace_seconds=3.0,
        sleep=lambda _: None,
    )
    result: list[int] = []
    runner = threading.Thread(target=lambda: result.append(supervisor.run()))
    runner.start()
    deadline = time.monotonic() + 15
    pid_files = [out / f"worker-{rank}.pid" for rank in range(2)]
    while not all(path.exists() for path in pid_files) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert all(path.exists() for path in pid_files), "torchrun workers did not start"
    worker_pids = [int(path.read_text()) for path in pid_files]

    supervisor.request_stop(signal.SIGTERM)
    runner.join(timeout=15)
    assert not runner.is_alive()
    assert result and result[0] != 0
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if all(not Path(f"/proc/{pid}").exists() for pid in worker_pids):
            break
        time.sleep(0.1)
    else:
        for pid in worker_pids:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        raise AssertionError("torchrun worker survived supervisor stop")


def test_verdict_file_contract_matches_train_cli() -> None:
    """The supervisor reads the same file name vrl-train writes."""
    from vrl.scripts import supervise, train

    assert supervise.RUN_VERDICT_NAME == train.RUN_VERDICT_NAME


def test_build_train_launch_keeps_single_process_direct() -> None:
    cfg = OmegaConf.create(
        {"distributed": {"training": {"strategy": "single_process"}}},
    )

    launch = build_train_launch(
        cfg,
        config="experiment/unit",
        overrides=["trainer.seed=7"],
        python_executable="/venv/python",
    )

    assert launch.expected_world_size == 1
    assert launch.command == (
        "/venv/python",
        "-m",
        "vrl.scripts.train",
        "--config",
        "experiment/unit",
        "trainer.seed=7",
    )


@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_build_train_launch_uses_one_host_torchrun(strategy: str) -> None:
    cfg = OmegaConf.create(
        {
            "distributed": {
                "training": {
                    "strategy": strategy,
                    "num_nodes": 1,
                    "gpus_per_node": 4,
                },
            },
        },
    )

    launch = build_train_launch(
        cfg,
        config="experiment/cosmos",
        overrides=["trainer.total_epochs=2"],
        python_executable="/venv/python",
    )

    assert launch.expected_world_size == 4
    assert launch.command == (
        "/venv/python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=4",
        "--max-restarts=0",
        "--module",
        "vrl.scripts.train",
        "--config",
        "experiment/cosmos",
        "trainer.total_epochs=2",
    )


def test_build_train_launch_rejects_unowned_multi_node_rendezvous() -> None:
    cfg = OmegaConf.create(
        {
            "distributed": {
                "training": {
                    "strategy": "fsdp",
                    "num_nodes": 2,
                    "gpus_per_node": 4,
                },
            },
        },
    )

    with pytest.raises(ValueError, match="rendezvous"):
        build_train_launch(
            cfg,
            config="experiment/cosmos",
            overrides=[],
        )


def test_supervisor_rejects_nested_torchrun_owners() -> None:
    from vrl.scripts import supervise

    with pytest.raises(ValueError, match="exactly one owner"):
        supervise._require_single_supervisor_owner(
            {"RANK": "1", "WORLD_SIZE": "4"},
        )


def test_train_writes_atomic_rank_verdicts_and_failure_wins_aggregation(tmp_path) -> None:
    from vrl.scripts import train

    out = tmp_path / "run"
    train.write_run_verdict(
        str(out),
        environ={"RANK": "0", "WORLD_SIZE": "4"},
    )
    train.write_run_verdict(
        str(out),
        received_signal=signal.SIGTERM,
        environ={"RANK": "1", "WORLD_SIZE": "4"},
    )
    train.write_run_verdict(
        str(out),
        error=ValueError("rank 2 root cause"),
        environ={"RANK": "2", "WORLD_SIZE": "4"},
    )
    train.write_run_verdict(
        str(out),
        environ={"RANK": "3", "WORLD_SIZE": "4"},
    )
    assert not (out / "run_verdict.json").exists()

    supervisor = RunSupervisor(
        command=[],
        output_dir=out,
        expected_world_size=4,
    )
    aggregate = supervisor._collect_attempt_verdict()

    assert aggregate is not None
    assert aggregate["verdict"] == "failed"
    assert aggregate["error_class"] == "ValueError"
    assert aggregate["failed_ranks"] == [2]
    assert json.loads((out / "run_verdict.json").read_text()) == aggregate
    assert not list(out.glob(".*.tmp-*"))


def test_train_keeps_single_process_verdict_name(tmp_path) -> None:
    from vrl.scripts import train

    out = tmp_path / "run"
    train.write_run_verdict(str(out), environ={})

    assert json.loads((out / "run_verdict.json").read_text())["verdict"] == "success"
    assert not list(out.glob("run_verdict.rank-*.json"))


def test_train_verdict_uses_cleanup_wrapper_root_class(tmp_path) -> None:
    from vrl.ray.operation_deadline import RayOperationTimeout
    from vrl.rollouts.orchestration.prompt_collection import (
        PromptCollectionCleanupError,
    )
    from vrl.rollouts.orchestration.strict_on_policy import RolloutPhaseCleanupError
    from vrl.scripts import train

    root = RayOperationTimeout("rollout.generation.chunk", 1.0)
    collection = PromptCollectionCleanupError(
        root,
        [RuntimeError("reward cleanup failed")],
    )
    wrapped = RolloutPhaseCleanupError(
        collection,
        [RuntimeError("offload cleanup failed")],
    )
    out = tmp_path / "run"

    train.write_run_verdict(str(out), error=wrapped, environ={})

    verdict = json.loads((out / "run_verdict.json").read_text())
    assert verdict["error_class"] == "RayOperationTimeout"
    assert "offload cleanup failed" in verdict["error_message"]
    assert "RayOperationTimeout" in verdict["error_message"]


def test_train_verdict_keeps_terminal_identity_before_dependency_cause(tmp_path) -> None:
    from vrl.ray.operation_deadline import RayOperationTimeout
    from vrl.scripts import train

    timeout = RayOperationTimeout("rollout.startup.load_policy", 1.0)
    timeout.__cause__ = TimeoutError("dependency-owned timeout")
    out = tmp_path / "run"

    train.write_run_verdict(str(out), error=timeout, environ={})

    verdict = json.loads((out / "run_verdict.json").read_text())
    assert verdict["error_class"] == "RayOperationTimeout"


def test_distributed_success_requires_every_rank_verdict(tmp_path) -> None:
    from vrl.scripts import train

    out = tmp_path / "run"
    train.write_run_verdict(
        str(out),
        environ={"RANK": "0", "WORLD_SIZE": "2"},
    )
    supervisor = RunSupervisor(
        command=[],
        output_dir=out,
        expected_world_size=2,
    )

    incomplete = supervisor._collect_attempt_verdict()
    assert incomplete is not None
    assert incomplete["verdict"] == "failed"
    assert incomplete["error_class"] == "MissingRankVerdict"
    assert incomplete["missing_ranks"] == [1]

    train.write_run_verdict(
        str(out),
        environ={"RANK": "1", "WORLD_SIZE": "2"},
    )
    complete = supervisor._collect_attempt_verdict()
    assert complete is not None
    assert complete["verdict"] == "success"


def test_metrics_health_gate_is_disabled_by_default() -> None:
    args = build_parser().parse_args(["--config", "unit"])

    assert args.health_metrics is False


def test_supervisor_cli_defaults_are_derived_from_runtime_config() -> None:
    args = build_parser().parse_args(["--config", "unit"])
    defaults = {item.name: item.default for item in fields(RunSupervisor)}

    assert args.max_attempts == defaults["max_attempts"]
    assert args.same_cause_limit == defaults["same_cause_limit"]


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--max-attempts", "-1"),
        ("--same-cause-limit", "0"),
    ],
)
def test_supervisor_cli_rejects_invalid_retry_bounds(option: str, value: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--config", "unit", option, value])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_attempts": -1}, "max_attempts"),
        ({"same_cause_limit": 0}, "same_cause_limit"),
        ({"term_grace_seconds": -1.0}, "term_grace_seconds"),
        ({"backoff_seconds": -1.0}, "backoff_seconds"),
        ({"term_grace_seconds": float("inf")}, "term_grace_seconds"),
        ({"backoff_seconds": float("nan")}, "backoff_seconds"),
    ],
)
def test_supervisor_runtime_rejects_invalid_retry_config(
    kwargs: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RunSupervisor(command=[], output_dir=Path("unused"), **kwargs)


def _health_args(*extra: str):
    return build_parser().parse_args(["--config", "unit", "--health-metrics", *extra])


def test_health_config_derives_strict_parity_limit_from_training_config() -> None:
    health = HealthGateConfig.from_cli(
        _health_args(),
        schedule=RolloutOrchestrationConfig(schedule_mode="strict_on_policy"),
        debug=DebugConfig(max_abs_logprob_diff=0.004),
    )

    assert health.continuous is None
    assert health.max_pre_update_logprob_diff == pytest.approx(0.004)


def test_health_config_keeps_explicit_parity_override() -> None:
    health = HealthGateConfig.from_cli(
        _health_args("--health-max-pre-update-logprob-diff", "0.02"),
        schedule=RolloutOrchestrationConfig(schedule_mode="strict_on_policy"),
        debug=DebugConfig(max_abs_logprob_diff=0.004),
    )

    assert health.max_pre_update_logprob_diff == pytest.approx(0.02)


@pytest.mark.parametrize(
    ("override", "expected"),
    [(None, 2), ("0", 0)],
)
def test_health_config_derives_continuous_metrics_from_training_schedule(
    override: str | None,
    expected: int,
) -> None:
    extra = () if override is None else ("--health-max-stale-policy-versions", override)

    health = HealthGateConfig.from_cli(
        _health_args(*extra),
        schedule=RolloutOrchestrationConfig(
            schedule_mode="continuous",
            continuous=ContinuousRolloutConfig(max_stale_policy_versions=2),
        ),
        debug=DebugConfig(),
    )

    assert health.continuous is not None
    assert health.continuous.max_stale_policy_versions == expected


def test_health_config_accepts_a_continuous_schedule_on_its_own_defaults() -> None:
    """A recipe that omits the stale key inherits ContinuousRolloutConfig's default.

    ``base/rollout/orchestration/continuous.yaml`` documents exactly this: the
    queue/staleness defaults live on the typed config, not in YAML. Reading the
    key through a raw config path made the supervisor reject every such recipe.
    """

    health = HealthGateConfig.from_cli(
        _health_args(),
        schedule=RolloutOrchestrationConfig(schedule_mode="continuous"),
        debug=DebugConfig(),
    )

    assert health.continuous is not None
    assert health.continuous.max_stale_policy_versions == (
        RolloutOrchestrationConfig().continuous.max_stale_policy_versions
    )


def test_health_config_rejects_stale_override_wider_than_training() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        HealthGateConfig.from_cli(
            _health_args("--health-max-stale-policy-versions", "2"),
            schedule=RolloutOrchestrationConfig(
                schedule_mode="continuous",
                continuous=ContinuousRolloutConfig(max_stale_policy_versions=1),
            ),
            debug=DebugConfig(),
        )


def test_health_config_rejects_stale_override_for_strict_schedule() -> None:
    with pytest.raises(ValueError, match=r"requires.*schedule_mode=continuous"):
        HealthGateConfig.from_cli(
            _health_args("--health-max-stale-policy-versions", "0"),
            schedule=RolloutOrchestrationConfig(schedule_mode="strict_on_policy"),
            debug=DebugConfig(),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_stale_policy_versions": -1},
        {"max_stale_policy_versions": 1, "max_stale_logprob_diff": -0.1},
        {"max_stale_policy_versions": 1, "max_stale_logprob_diff": float("nan")},
    ],
)
def test_continuous_health_policy_rejects_invalid_thresholds(
    kwargs: dict[str, float | int],
) -> None:
    with pytest.raises(ValueError):
        ContinuousHealthPolicy(**kwargs)


def test_health_required_columns_must_exist_in_online_metric_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vrl.scripts.supervise as supervise

    monkeypatch.setattr(
        supervise,
        "_REQUIRED_HEALTH_METRICS",
        ("removed_metric",),
    )

    with pytest.raises(AssertionError, match="removed_metric"):
        HealthGateConfig()


def test_health_gate_reads_a_complete_online_metric_row(tmp_path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    row = build_online_metric_row(
        0,
        TrainStepMetrics(
            loss=1.0,
            reward_mean=2.0,
            reward_std=0.2,
            grad_norm=0.1,
            initial_replay=InitialReplayStats(logprob_abs_diff_max=0.001),
        ),
    )
    (out / "metrics.csv").write_text(
        ",".join(online_metric_columns()) + "\n" + format_online_metric_row(row),
    )

    gate = MetricsHealthGate(HealthGateConfig(failure_limit=1), out)

    assert gate.judge_new_rows() is False
    assert not (out / HEALTH_VERDICT_NAME).exists()


def test_metrics_health_gate_cli_thresholds_are_configurable() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "unit",
            "--health-metrics",
            "--health-poll-seconds",
            "2.5",
            "--health-failure-limit",
            "4",
            "--health-max-pre-update-logprob-diff",
            "0.02",
            "--health-max-stale-policy-versions",
            "1",
            "--health-max-stale-logprob-diff",
            "0.08",
            "--health-min-reward-std",
            "0.003",
            "--health-min-grad-norm",
            "0.0002",
        ],
    )

    assert args.health_metrics is True
    assert args.health_poll_seconds == 2.5
    assert args.health_failure_limit == 4
    assert args.health_max_pre_update_logprob_diff == 0.02
    assert args.health_max_stale_policy_versions == 1
    assert args.health_max_stale_logprob_diff == 0.08
    assert args.health_min_reward_std == 0.003
    assert args.health_min_grad_norm == 0.0002


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--health-poll-seconds", "0"),
        ("--health-failure-limit", "0"),
        ("--health-max-pre-update-logprob-diff", "nan"),
        ("--health-max-stale-policy-versions", "-1"),
        ("--health-max-stale-logprob-diff", "nan"),
        ("--health-min-reward-std", "-1"),
        ("--health-min-grad-norm", "inf"),
    ],
)
def test_metrics_health_gate_cli_rejects_invalid_thresholds(option, value) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--config", "unit", option, value])


def test_metrics_health_gate_allows_missing_and_header_only_files(tmp_path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    (out / "metrics.csv").write_text(_METRICS_HEADER)
    command = _child_script(
        tmp_path,
        _write_verdict_snippet(out)
        + "(out / 'run_verdict.json').write_text(json.dumps({'verdict': 'success'}))\n",
    )
    supervisor = RunSupervisor(
        command=command,
        output_dir=out,
        health=HealthGateConfig(poll_seconds=0.01),
        sleep=lambda _: None,
    )

    assert supervisor.run() == 0
    assert not (out / HEALTH_VERDICT_NAME).exists()


def test_continuous_health_gate_uses_stale_drift_bound() -> None:
    gate = MetricsHealthGate(
        HealthGateConfig(
            max_pre_update_logprob_diff=0.01,
            continuous=ContinuousHealthPolicy(
                max_stale_policy_versions=1,
                max_stale_logprob_diff=0.05,
            ),
        ),
        Path("unused"),
    )

    stale_reasons, _ = gate._metric_health_reasons(
        _continuous_metric_row(0, parity="0.02", stale_versions="1"),
    )
    on_policy_reasons, _ = gate._metric_health_reasons(
        _continuous_metric_row(1, parity="0.02", stale_versions="0"),
    )

    assert stale_reasons == []
    assert any("exceeds maximum 0.01" in reason for reason in on_policy_reasons)


def test_continuous_schedule_requires_metrics_even_with_zero_stale_limit() -> None:
    gate = MetricsHealthGate(
        HealthGateConfig(
            continuous=ContinuousHealthPolicy(max_stale_policy_versions=0),
        ),
        Path("unused"),
    )
    row = _continuous_metric_row(0)
    del row["continuous_stale_versions"]
    del row["continuous_producer_errors"]

    reasons, _ = gate._metric_health_reasons(row)

    assert "metric continuous_stale_versions is missing" in reasons
    assert "metric continuous_producer_errors is missing" in reasons


@pytest.mark.parametrize(
    ("stale_versions", "producer_errors", "expected"),
    [
        ("2", "0", "continuous_stale_versions 2 exceeds maximum 1"),
        ("0.5", "0", "must be a non-negative integer"),
        ("1", "0.5", "continuous_producer_errors must be a non-negative integer"),
        ("1", "-1", "continuous_producer_errors must be a non-negative integer"),
    ],
)
def test_continuous_health_gate_rejects_scheduler_failures(
    stale_versions,
    producer_errors,
    expected,
) -> None:
    gate = MetricsHealthGate(
        HealthGateConfig(continuous=ContinuousHealthPolicy(max_stale_policy_versions=1)),
        Path("unused"),
    )

    reasons, _ = gate._metric_health_reasons(
        _continuous_metric_row(
            0,
            stale_versions=stale_versions,
            producer_errors=producer_errors,
        ),
    )

    assert any(expected in reason for reason in reasons)


def test_continuous_health_gate_only_flags_new_producer_errors(tmp_path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    gate = MetricsHealthGate(
        HealthGateConfig(
            continuous=ContinuousHealthPolicy(max_stale_policy_versions=1),
            failure_limit=3,
        ),
        out,
    )

    rows = [_continuous_metric_row(0, producer_errors="0")]
    _write_continuous_metrics(out, rows)
    assert gate.judge_new_rows() is False

    rows.append(_continuous_metric_row(1, producer_errors="1"))
    _write_continuous_metrics(out, rows)
    assert gate.judge_new_rows() is False
    assert gate._unhealthy_rows == 1

    rows.append(_continuous_metric_row(2, producer_errors="1"))
    _write_continuous_metrics(out, rows)
    assert gate.judge_new_rows() is False
    assert gate._unhealthy_rows == 0


def test_continuous_health_gate_trips_on_consecutive_producer_error_increases(
    tmp_path,
) -> None:
    out = tmp_path / "run"
    out.mkdir()
    gate = MetricsHealthGate(
        HealthGateConfig(
            continuous=ContinuousHealthPolicy(max_stale_policy_versions=1),
            failure_limit=3,
        ),
        out,
    )

    for epoch, producer_errors in enumerate(("1", "2", "3")):
        rows = [_continuous_metric_row(i, producer_errors=str(i + 1)) for i in range(epoch + 1)]
        _write_continuous_metrics(out, rows)
        tripped = gate.judge_new_rows()
        assert tripped is (producer_errors == "3")

    verdict = json.loads((out / HEALTH_VERDICT_NAME).read_text())
    assert verdict["epoch"] == 2
    assert verdict["reasons"] == [
        "continuous_producer_errors increased from 2 to 3",
    ]


def test_continuous_health_gate_baselines_existing_producer_errors(tmp_path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    rows = [_continuous_metric_row(0, producer_errors="1")]
    _write_continuous_metrics(out, rows)
    command = _child_script(
        tmp_path,
        _write_verdict_snippet(out)
        + "with (out / 'metrics.csv').open('a') as handle:\n"
        + f"    handle.write({_continuous_metric_csv_row(1, producer_errors='0')!r})\n"
        + "(out / 'run_verdict.json').write_text(json.dumps({'verdict': 'success'}))\n",
    )
    supervisor = RunSupervisor(
        command=command,
        output_dir=out,
        health=HealthGateConfig(
            poll_seconds=0.01,
            continuous=ContinuousHealthPolicy(max_stale_policy_versions=1),
            failure_limit=1,
        ),
        sleep=lambda _: None,
    )

    assert supervisor.run() == 0
    assert not (out / HEALTH_VERDICT_NAME).exists()


def test_metrics_health_gate_resets_failure_streak_after_healthy_row(tmp_path) -> None:
    out = tmp_path / "run"
    metrics = _METRICS_HEADER + _metric_row(0, reward_std="0")
    metrics += _metric_row(1)
    metrics += _metric_row(2, grad_norm="0")
    command = _child_script(
        tmp_path,
        _write_verdict_snippet(out)
        + f"(out / 'metrics.csv').write_text({metrics!r})\n"
        + "(out / 'run_verdict.json').write_text(json.dumps({'verdict': 'success'}))\n",
    )
    supervisor = RunSupervisor(
        command=command,
        output_dir=out,
        health=HealthGateConfig(poll_seconds=0.01, failure_limit=2),
        sleep=lambda _: None,
    )

    assert supervisor.run() == 0
    assert not (out / HEALTH_VERDICT_NAME).exists()


def test_metrics_health_gate_ignores_old_rows_and_partial_append(tmp_path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    old_metrics = _METRICS_HEADER
    old_metrics += _metric_row(0, reward_std="0")
    old_metrics += _metric_row(1, reward_std="0")
    (out / "metrics.csv").write_text(old_metrics)
    command = _child_script(
        tmp_path,
        _write_verdict_snippet(out)
        + "with (out / 'metrics.csv').open('a') as handle:\n"
        + "    handle.write('2,1.0,2.0,0')\n"
        + "    handle.flush()\n"
        + "    import time; time.sleep(0.08)\n"
        + "    handle.write(',0.1,0.001\\n')\n"
        + "    handle.flush()\n"
        + "(out / 'run_verdict.json').write_text(json.dumps({'verdict': 'success'}))\n",
    )
    supervisor = RunSupervisor(
        command=command,
        output_dir=out,
        health=HealthGateConfig(poll_seconds=0.01, failure_limit=2),
        sleep=lambda _: None,
    )

    assert supervisor.run() == 0
    assert not (out / HEALTH_VERDICT_NAME).exists()


def test_metrics_health_gate_only_checks_replaced_suffix_after_truncation(tmp_path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    retained_row = _metric_row(0)
    (out / "metrics.csv").write_text(_METRICS_HEADER + retained_row + _metric_row(1))
    command = _child_script(
        tmp_path,
        "import pathlib, time\n"
        + f"out = pathlib.Path({str(out)!r})\n"
        + f"(out / 'metrics.csv').write_text({_METRICS_HEADER + retained_row!r})\n"
        + "time.sleep(0.04)\n"
        + "with (out / 'metrics.csv').open('a') as handle:\n"
        + f"    handle.write({_metric_row(1, reward_std='0')!r})\n"
        + "    handle.flush()\n"
        + "time.sleep(600)\n",
    )
    supervisor = RunSupervisor(
        command=command,
        output_dir=out,
        health=HealthGateConfig(poll_seconds=0.01, failure_limit=1),
        term_grace_seconds=0.2,
        sleep=lambda _: None,
    )

    assert supervisor.run() != 0
    verdict = json.loads((out / HEALTH_VERDICT_NAME).read_text())
    assert verdict["epoch"] == 1


def test_metrics_health_gate_stops_child_group_and_writes_verdict(tmp_path) -> None:
    out = tmp_path / "run"
    grandchild_pid_file = tmp_path / "health-grandchild.pid"
    unhealthy_rows = _metric_row(
        0,
        loss="nan",
        reward_mean="inf",
        reward_std="0",
        grad_norm="0",
        parity="0.02",
    ) + _metric_row(
        1,
        loss="nan",
        reward_mean="inf",
        reward_std="0",
        grad_norm="0",
        parity="0.02",
    )
    command = _child_script(
        tmp_path,
        "import pathlib, subprocess, sys, time\n"
        + f"out = pathlib.Path({str(out)!r})\n"
        + "out.mkdir(parents=True, exist_ok=True)\n"
        + "grandchild = subprocess.Popen([sys.executable, '-c', "
        + "'import time; time.sleep(600)'])\n"
        + f"pathlib.Path({str(grandchild_pid_file)!r}).write_text(str(grandchild.pid))\n"
        + f"rows = {unhealthy_rows!r}.splitlines(keepends=True)\n"
        + "with (out / 'metrics.csv').open('w') as handle:\n"
        + f"    handle.write({_METRICS_HEADER!r})\n"
        + "    handle.flush()\n"
        + "    for row in rows:\n"
        + "        handle.write(row)\n"
        + "        handle.flush()\n"
        + "        time.sleep(0.05)\n"
        + "time.sleep(600)\n",
    )
    supervisor = RunSupervisor(
        command=command,
        output_dir=out,
        health=HealthGateConfig(poll_seconds=0.01, failure_limit=2),
        term_grace_seconds=0.2,
        sleep=lambda _: None,
    )

    assert supervisor.run() != 0
    verdict = json.loads((out / HEALTH_VERDICT_NAME).read_text())
    assert verdict["verdict"] == "failed"
    assert verdict["source"] == "metrics_health_gate"
    assert verdict["epoch"] == 1
    assert verdict["consecutive_unhealthy_rows"] == 2
    assert any("loss is not finite" in reason for reason in verdict["reasons"])
    assert any("reward_mean is not finite" in reason for reason in verdict["reasons"])
    assert any("exceeds maximum" in reason for reason in verdict["reasons"])
    assert any("reward_std" in reason for reason in verdict["reasons"])
    assert any("grad_norm" in reason for reason in verdict["reasons"])

    grandchild_pid = int(grandchild_pid_file.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(grandchild_pid, signal.SIGKILL)
        raise AssertionError("grandchild survived metrics health-gate stop")
