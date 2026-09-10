"""Real actor transport and CLI lifecycle with explicitly tiny CPU model fixtures."""

from types import SimpleNamespace

import pytest
import torch

from vrl.scripts.perf import weight_delivery_probe as probe


class TinyProbeWorker(probe.WeightDeliveryProbeWorker):
    def __init__(self, worker_id, _inputs):
        from tests.generation.execution.test_worker_versioned_slots import _core, _ReadbackModel

        self.core = _core(_ReadbackModel(), versioned_weight_sync=False)
        self.core.worker_id = worker_id


class MissedInstallProbeWorker(TinyProbeWorker):
    def update_weights(self, state_ref, policy_version, *, verify_content=False):
        if policy_version == 1 and self.core.worker_id == "acceptance-1":
            self.core.executor.model.skip_install = True
        return super().update_weights(state_ref, policy_version, verify_content=verify_content)


@pytest.mark.slow_test
@pytest.mark.parametrize("failure", [None, "cleanup", "install"])
def test_probe_cli_exports_real_cpu_parameters_and_checks_two_receivers(
    tmp_path, monkeypatch, local_ray, failure
):
    import json

    source = torch.nn.Linear(2, 2, bias=False)
    original = source.weight.detach().clone()
    replay = SimpleNamespace(
        identity={"fixture": "tiny-cpu"},
        materialize=lambda **_: SimpleNamespace(trainable_modules={"transformer": source}),
    )
    resolved = SimpleNamespace(
        family=SimpleNamespace(family="test"),
        built=SimpleNamespace(root=None, precision=None),
        resources=SimpleNamespace(rollout_devices=(), rollout_gpus_per_engine=1),
        ray_launch_inputs=lambda _: SimpleNamespace(
            launch_contract=SimpleNamespace(versioned_weight_sync=False)
        ),
    )
    monkeypatch.setattr(probe, "resolve_online_run", lambda _: resolved)
    monkeypatch.setattr(probe, "resolve_model", lambda *_, **__: replay)
    monkeypatch.setattr(
        probe,
        "WeightDeliveryProbeWorker",
        MissedInstallProbeWorker if failure == "install" else TinyProbeWorker,
    )
    monkeypatch.setattr("vrl.config.loading.load_config", lambda *_, **__: None)
    calls = []

    def shutdown():
        calls.append("shutdown")
        if failure == "cleanup":
            raise RuntimeError("cleanup failure fixture")

    # The fixture owns the real cluster. Only the CLI's ownership calls are
    # replaced here; actor launch, serialization, receiver core and readback run.
    monkeypatch.setattr(
        probe,
        "require_ray",
        lambda: SimpleNamespace(
            is_initialized=lambda: False,
            init=lambda **kwargs: calls.append(kwargs),
            shutdown=shutdown,
            put=local_ray.put,
            get=local_ray.get,
        ),
    )
    report = tmp_path / "report.json"
    args = ["--config", "fixture", "--report", str(report), "--workers", "2"]
    if failure == "cleanup":
        with pytest.raises(RuntimeError, match="cleanup failure fixture"):
            probe.main(args)
        assert not report.exists()
    elif failure == "install":
        with pytest.raises(local_ray.exceptions.RayTaskError, match="installed weight content"):
            probe.main(args)
        assert not report.exists()
    else:
        probe.main(args)
        record = json.loads(report.read_text())
        assert {r["worker_id"] for r in record["receivers"]} == {"acceptance-0", "acceptance-1"}
        assert all(r["bytes"] == 16 and r["policy_version"] == 2 for r in record["receivers"])
        with pytest.raises(FileExistsError):
            probe.main(args)
    assert calls[0] == {"address": "local", "include_dashboard": False}
    assert calls[-1] == "shutdown"
    assert torch.equal(source.weight, original)


@pytest.mark.parametrize("timeout", ["0", "nan", "inf"])
def test_invalid_deadline_rejected_before_model_construction(tmp_path, timeout):
    with pytest.raises(ValueError, match="positive"):
        probe.main(["--config", "unused", "--report", str(tmp_path / "x"), "--timeout-s", timeout])
