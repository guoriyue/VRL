"""Real actor transport and CLI lifecycle with explicitly tiny CPU model fixtures."""

from types import SimpleNamespace

import pytest
import torch

from vrl.ray.actor_pool import RayActorCallError
from vrl.run import OnlineRunConfig
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

    def commit_weight_transfer(self, transfer_id, *, verify_content=False):
        if (
            self.core._weight_transfer.policy_version == 1
            and self.core.worker_id == "acceptance-1"
        ):
            self.core.executor.model.skip_install = True
        return super().commit_weight_transfer(transfer_id, verify_content=verify_content)


class MissingChunkProbeWorker(TinyProbeWorker):
    def receive_weight_bucket(self, bucket, transfer_id):
        transfer = self.core._weight_transfer
        if (
            self.core.worker_id == "acceptance-1"
            and transfer.policy_version == 1
            and bucket[0][1] > 0
        ):
            return transfer.policy_version
        return super().receive_weight_bucket(bucket, transfer_id)


@pytest.mark.slow_test
@pytest.mark.parametrize(
    ("bucket_bytes", "failure"),
    [(size, failure) for size in (None, 8) for failure in (None, "cleanup", "install")]
    + [(8, "chunk")],
)
def test_probe_cli_exports_real_cpu_parameters_and_checks_two_receivers(
    tmp_path, monkeypatch, local_ray, failure, bucket_bytes
):
    import json

    sources = []
    originals = []

    def materialize(**_):
        source = torch.nn.Linear(2, 2, bias=False)
        sources.append(source)
        originals.append(source.weight.detach().clone())
        return SimpleNamespace(trainable_modules={"transformer": source})

    replay = SimpleNamespace(
        identity={"fixture": "tiny-cpu"},
        materialize=materialize,
    )
    resolved = SimpleNamespace(
        run=OnlineRunConfig(total_epochs=1, seed=17),
        generation=SimpleNamespace(worker=SimpleNamespace(weight_sync_bucket_bytes=bucket_bytes)),
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
        {
            "install": MissedInstallProbeWorker,
            "chunk": MissingChunkProbeWorker,
        }.get(failure, TinyProbeWorker),
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
    elif failure in ("install", "chunk"):
        with pytest.raises(RayActorCallError) as error:
            probe.main(args)
        message = (
            "installed weight content" if failure == "install" else "incomplete weight transfer"
        )
        assert message in str(error.value.__cause__)
        assert not report.exists()
    else:
        probe.main(args)
        record = json.loads(report.read_text())
        assert record["schema"] == "vrl.weight-delivery-acceptance/v2"
        assert record["transport"]["bucket_bytes"] == bucket_bytes
        assert record["transport"]["kind"] == ("staged_buckets" if bucket_bytes else "snapshot")
        assert set(record["transport"]["sync_verify_wall_s"]) == {"first", "repeat"}
        assert record["source_initialization"] == {"seed": 17, "deterministic": False}
        assert {r["worker_id"] for r in record["receivers"]} == {"acceptance-0", "acceptance-1"}
        assert all(r["bytes"] == 16 and r["policy_version"] == 2 for r in record["receivers"])
        with pytest.raises(FileExistsError):
            probe.main(args)
        # Fresh initialization must use the configured seed, irrespective of
        # unrelated caller RNG use. These repeat runs still cross real Ray RPCs.
        torch.manual_seed(999)
        repeated_args = [*args]
        repeated_args[3] = str(tmp_path / "repeated.json")
        probe.main(repeated_args)
        assert torch.equal(originals[0], originals[1])
        resolved.run = OnlineRunConfig(total_epochs=1, seed=18)
        repeated_args[3] = str(tmp_path / "different-seed.json")
        probe.main(repeated_args)
        assert not torch.equal(originals[0], originals[2])
    assert calls[0] == {"address": "local", "include_dashboard": False}
    assert calls[-1] == "shutdown"
    for source, original in zip(sources, originals, strict=True):
        assert torch.equal(source.weight, original)


@pytest.mark.parametrize("timeout", ["0", "nan", "inf"])
def test_invalid_deadline_rejected_before_model_construction(tmp_path, timeout):
    with pytest.raises(ValueError, match="positive"):
        probe.main(["--config", "unused", "--report", str(tmp_path / "x"), "--timeout-s", timeout])
