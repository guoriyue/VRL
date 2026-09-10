"""Launch-record integrity; these CPU fixtures do not claim a verified recipe."""

from __future__ import annotations

import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from vrl.trainers import evidence


@pytest.fixture
def stable_environment(monkeypatch):
    monkeypatch.delenv("VRL_RUN_ATTEMPT_ID", raising=False)
    monkeypatch.setattr(evidence, "_runtime_identity", lambda: {"devices": [], "python": "test"})
    monkeypatch.setattr(
        evidence,
        "_code_identity",
        lambda _path: {"available": True, "commit": "abc", "dirty": False},
    )


def test_launch_and_resume_have_distinct_immutable_records(tmp_path, stable_environment):
    cfg = OmegaConf.create({"seed": 17, "sampling": {"seed": "${seed}"}})
    identity = {"schema": "test-identity", "checkpoint": "sha256:abc"}
    first = evidence.write_run_evidence(cfg, tmp_path, model_identity=identity, resumed=False)
    original = first.read_bytes()
    second = evidence.write_run_evidence(cfg, tmp_path, model_identity=identity, resumed=True)
    assert first != second
    assert first.read_bytes() == original
    record = json.loads(original)
    assert record["schema"] == evidence.RUN_EVIDENCE_SCHEMA
    assert record["config"]["sampling"]["seed"] == 17
    canonical = json.dumps(
        record["config"], sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    assert record["config_sha256"] == hashlib.sha256(canonical.encode()).hexdigest()
    assert record["model_identity"] == identity
    assert json.loads(second.read_text())["resumed"] is True
    assert "success" not in record and "verified" not in record


def test_launch_id_collision_never_replaces_prior_evidence(
    tmp_path, stable_environment, monkeypatch
):
    monkeypatch.setattr(evidence.uuid, "uuid4", lambda: SimpleNamespace(hex="fixed"))
    cfg = OmegaConf.create({"seed": 1})
    first = evidence.write_run_evidence(
        cfg, tmp_path, model_identity={"model": "a"}, resumed=False
    )
    original = first.read_bytes()
    with pytest.raises(FileExistsError):
        evidence.write_run_evidence(cfg, tmp_path, model_identity={"model": "b"}, resumed=True)
    assert first.read_bytes() == original
    assert list(first.parent.iterdir()) == [first]


def test_invalid_config_or_missing_model_identity_is_not_published(tmp_path, stable_environment):
    with pytest.raises(ValueError, match="resolved model identity"):
        evidence.write_run_evidence(
            OmegaConf.create({}), tmp_path, model_identity={}, resumed=False
        )
    with pytest.raises(ValueError):
        evidence.write_run_evidence(
            OmegaConf.create({"value": float("nan")}),
            tmp_path,
            model_identity={"model": "a"},
            resumed=False,
        )
    assert not (tmp_path / "run_evidence").exists()


def test_runtime_snapshot_records_cpu_scope_without_dumping_environment(monkeypatch):
    monkeypatch.setattr(evidence.torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("PRIVATE_API_TOKEN", "must-not-be-recorded")
    record = evidence._runtime_identity()
    assert record["device_scope"] == "trainer-process-visible-devices"
    assert record["devices"] == []
    assert record["environment"]["WORLD_SIZE"] == "1"
    assert "PRIVATE_API_TOKEN" not in record["environment"]
    assert "torch" in {name.lower() for name in record["packages"]}
    assert isinstance(record["deterministic_algorithms"], bool)


def test_git_identity_distinguishes_clean_dirty_and_unavailable(tmp_path):
    assert evidence._code_identity(tmp_path)["available"] is False

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(tmp_path), *args], stderr=subprocess.DEVNULL
        )

    git("init")
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    git("add", "source.py")
    git(
        "-c",
        "user.name=Evidence Test",
        "-c",
        "user.email=evidence@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    clean = evidence._code_identity(tmp_path)
    assert clean["available"] is True and clean["dirty"] is False
    source.write_text("value = 2\n")
    (tmp_path / "untracked.py").write_text("extra = True\n")
    dirty = evidence._code_identity(tmp_path)
    assert dirty["commit"] == clean["commit"]
    assert dirty["dirty"] is True
    assert dirty["tracked_diff_sha256"] != clean["tracked_diff_sha256"]
    assert any("untracked.py" in line for line in dirty["status"])


def test_manifest_content_changes_are_recorded_even_when_config_is_identical(
    tmp_path, stable_environment
):
    manifest = tmp_path / "prompts.jsonl"
    manifest.write_text('{"prompt": "first"}\n')
    cfg = OmegaConf.create({"data": {"manifest": {str(manifest): 2}}})
    first = evidence.write_run_evidence(
        cfg, tmp_path, model_identity={"model": "a"}, resumed=False
    )
    manifest.write_text('{"prompt": "changed"}\n')
    second = evidence.write_run_evidence(
        cfg, tmp_path, model_identity={"model": "a"}, resumed=True
    )
    before, after = [json.loads(path.read_text()) for path in (first, second)]
    assert before["config_sha256"] == after["config_sha256"]
    assert (
        before["configured_data_files"][str(manifest)]["sha256"]
        != after["configured_data_files"][str(manifest)]["sha256"]
    )
    assert after["configured_data_files"][str(manifest)]["changed_during_read"] is False


@pytest.fixture
def completed_loop(tmp_path, stable_environment):
    launch = evidence.write_run_evidence(
        OmegaConf.create({"seed": 17}),
        tmp_path,
        model_identity={"model": "fixture"},
        resumed=False,
    )
    (tmp_path / "metrics.csv").write_text("epoch,reward\n0,0.25\n")
    checkpoint = tmp_path / "checkpoint-final"
    checkpoint.mkdir()
    # Intentionally not a pickle: checking integrity must never deserialize it.
    (checkpoint / "checkpoint.pt").write_bytes(b"opaque checkpoint fixture")
    (checkpoint / "checkpoint_meta.json").write_text('{"global_step": 1}\n')
    return launch


def test_artifact_seal_binds_full_tree_and_is_relocatable(completed_loop, tmp_path):
    import shutil

    seal = evidence.seal_run_artifacts(completed_loop)
    record = evidence.verify_run_artifacts(seal)
    assert record["artifacts"]["final_checkpoint"]["content"]["files"] == 2
    assert record["phase"] == "after-training-loop-before-cleanup"
    assert "success" not in record and "verified" not in record
    original = seal.read_bytes()
    with pytest.raises(FileExistsError):
        evidence.seal_run_artifacts(completed_loop)
    assert seal.read_bytes() == original
    copied = tmp_path.parent / (tmp_path.name + "-archived")
    shutil.copytree(tmp_path, copied)
    assert evidence.verify_run_artifacts(copied / "run_evidence" / seal.name) == record


@pytest.mark.parametrize(
    "change",
    ["metrics", "checkpoint", "extra-file", "missing-file", "model-identity", "config"],
)
def test_artifact_drift_is_rejected(completed_loop, tmp_path, change):
    seal = evidence.seal_run_artifacts(completed_loop)
    if change == "metrics":
        (tmp_path / "metrics.csv").write_text("epoch,reward\n0,0.99\n")
    elif change == "checkpoint":
        (tmp_path / "checkpoint-final/checkpoint.pt").write_bytes(b"replacement checkpoint")
    elif change == "extra-file":
        (tmp_path / "checkpoint-final/extra.pt").write_bytes(b"extra shard")
    elif change == "missing-file":
        (tmp_path / "checkpoint-final/checkpoint_meta.json").unlink()
    else:
        launch = json.loads(completed_loop.read_text())
        if change == "model-identity":
            launch["model_identity"] = {"model": "replacement"}
        else:
            launch["config"]["seed"] = 99
        completed_loop.write_text(json.dumps(launch))
    with pytest.raises(ValueError, match="mismatch"):
        evidence.verify_run_artifacts(seal)


def test_missing_metrics_cannot_be_sealed(completed_loop, tmp_path):
    (tmp_path / "metrics.csv").unlink()
    with pytest.raises(RuntimeError, match="does not exist"):
        evidence.seal_run_artifacts(completed_loop)
    assert not completed_loop.with_suffix(".artifacts.json").exists()


@pytest.mark.parametrize("change", ["missing-role", "external-path", "launch-id", "schema"])
def test_incomplete_or_redirected_receipt_is_rejected(completed_loop, change):
    seal = evidence.seal_run_artifacts(completed_loop)
    record = json.loads(seal.read_text())
    if change == "missing-role":
        del record["artifacts"]["final_checkpoint"]
    elif change == "external-path":
        record["artifacts"]["metrics"]["path"] = "../../other/metrics.csv"
    elif change == "launch-id":
        record["launch_id"] = "another-launch"
    else:
        record["schema"] = "unknown/v2"
    seal.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        evidence.verify_run_artifacts(seal)


@pytest.mark.parametrize("world_size", [1, 2])
def test_completion_requires_matching_attempt_and_every_successful_rank(
    completed_loop, tmp_path, world_size
):
    from vrl.scripts.supervise import RunSupervisor
    from vrl.scripts.train import write_run_verdict

    launch = json.loads(completed_loop.read_text())
    launch["attempt_id"] = "current"
    launch["runtime"]["environment"] = {"WORLD_SIZE": str(world_size)}
    completed_loop.write_text(json.dumps(launch))
    seal = evidence.seal_run_artifacts(completed_loop)
    supervisor = RunSupervisor(command=[], output_dir=tmp_path, expected_world_size=world_size)
    supervisor._attempt_id = "current"
    for rank in range(world_size):
        write_run_verdict(
            str(tmp_path),
            environ={
                "RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "VRL_RUN_ATTEMPT_ID": "current",
            },
        )
    supervisor._collect_attempt_verdict(exit_code=0)
    verdict_path = tmp_path / "run_verdict.json"
    verdict = evidence.verify_run_completion(seal, verdict_path)
    assert verdict["verdict"] == "success"
    verdict["attempt_id"] = "previous"
    verdict_path.write_text(json.dumps(verdict))
    with pytest.raises(ValueError, match="different attempt"):
        evidence.verify_run_completion(seal, verdict_path)
    verdict["attempt_id"] = "current"
    verdict["verdict"] = "failed"
    verdict_path.write_text(json.dumps(verdict))
    with pytest.raises(ValueError, match="did not complete"):
        evidence.verify_run_completion(seal, verdict_path)
    if world_size > 1:
        verdict["verdict"] = "success"
        for mutate in (
            lambda ranks: ranks.pop(),
            lambda ranks: ranks[1].update(attempt_id="previous"),
            lambda ranks: ranks[1].update(verdict="failed"),
            lambda ranks: ranks[1].update(rank=0),
        ):
            candidate = json.loads(json.dumps(verdict))
            mutate(candidate["rank_verdicts"])
            verdict_path.write_text(json.dumps(candidate))
            with pytest.raises(ValueError, match="rank verdict"):
                evidence.verify_run_completion(seal, verdict_path)


def test_old_launch_without_attempt_identity_cannot_borrow_a_success(completed_loop, tmp_path):
    seal = evidence.seal_run_artifacts(completed_loop)
    verdict = tmp_path / "run_verdict.json"
    verdict.write_text('{"schema_version": 1, "verdict": "success"}')
    with pytest.raises(ValueError, match="no shared attempt identity"):
        evidence.verify_run_completion(seal, verdict)


def test_launch_captures_supervisor_attempt_identity(tmp_path, stable_environment, monkeypatch):
    monkeypatch.setenv("VRL_RUN_ATTEMPT_ID", "supervised-attempt")
    launch = evidence.write_run_evidence(
        OmegaConf.create({}), tmp_path, model_identity={"model": "fixture"}, resumed=False
    )
    assert json.loads(launch.read_text())["attempt_id"] == "supervised-attempt"


@pytest.mark.parametrize("exit_code", [None, 1, -9, False])
def test_success_payload_requires_observed_zero_process_exit(completed_loop, tmp_path, exit_code):
    launch = json.loads(completed_loop.read_text())
    launch["attempt_id"] = "current"
    completed_loop.write_text(json.dumps(launch))
    seal = evidence.seal_run_artifacts(completed_loop)
    verdict = tmp_path / "run_verdict.json"
    verdict.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "verdict": "success",
                "attempt_id": "current",
                "supervisor_exit_code": exit_code,
            }
        )
    )
    with pytest.raises(ValueError, match="successful process exit"):
        evidence.verify_run_completion(seal, verdict)


def test_full_precision_metrics_are_bound_when_present(completed_loop, tmp_path):
    path = tmp_path / "metrics.full_precision.csv"
    path.write_text("epoch,loss\n0,0.123456789\n")
    seal = evidence.seal_run_artifacts(completed_loop)
    record = evidence.verify_run_artifacts(seal)
    assert record["artifacts"]["full_precision_metrics"]["path"] == path.name
    path.write_text("epoch,loss\n0,0.123456788\n")
    with pytest.raises(ValueError, match="full_precision_metrics artifact content mismatch"):
        evidence.verify_run_artifacts(seal)
