"""CPU evidence fixtures: regression guards, not actual training determinism."""

import copy
import json

import pytest
from omegaconf import OmegaConf

from vrl.scripts.eval.compare_training_metrics import main
from vrl.trainers import evidence


@pytest.fixture
def make_run(tmp_path, monkeypatch):
    manifest = tmp_path / "prompts.txt"
    manifest.write_text("a test prompt\n")

    def make(
        name,
        *,
        rows=None,
        config_change=None,
        runtime_change=None,
        full_precision=True,
        model="fixture",
        data_text=None,
        code_change=None,
    ):
        directory = tmp_path / name
        if data_text is not None:
            manifest.write_text(data_text)
        code = {"available": True, "commit": "fixture", "dirty": False}
        code.update(code_change or {})
        monkeypatch.setattr(
            evidence.TrainingRunEvidence, "_code_identity", lambda _: copy.deepcopy(code)
        )
        runtime = {
            "python": "fixture",
            "platform": "fixture",
            "packages": {"torch": "fixture"},
            "torch_build": "fixture",
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "cudnn_benchmark": False,
            "matmul_reduced_precision_reduction": {"fp16": True, "bf16": True},
            "devices": [],
            "environment": {},
        }
        if runtime_change:
            runtime.update(runtime_change)
        monkeypatch.setattr(
            evidence.TrainingRunEvidence, "_runtime_identity", lambda: copy.deepcopy(runtime)
        )
        config = {
            "trainer": {"seed": 17, "total_epochs": 2, "output_dir": str(directory)},
            "data": {"manifest": str(manifest)},
        }
        if config_change:
            config_change(config)
        launch = evidence.TrainingRunEvidence.capture(
            OmegaConf.create(config), directory, model_identity={"model": model}, resumed=False
        ).launch_path
        (directory / "metrics.csv").write_text("epoch,loss\n0,0.123457\n1,0.234568\n")
        if full_precision:
            (directory / "metrics.full_precision.csv").write_text(
                "epoch,loss,reward_mean\n"
                + (rows if rows is not None else "0,0.123456789,0.5\n1,0.234567891,0.6\n")
            )
        checkpoint = directory / "checkpoint-final"
        checkpoint.mkdir()
        (checkpoint / "checkpoint.pt").write_bytes(b"opaque fixture, not model training")
        seal = evidence.TrainingRunEvidence.load(launch).seal_artifacts()
        verdict = directory / "run_verdict.json"
        verdict.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "verdict": "success",
                    "supervisor_exit_code": 0,
                }
            )
        )
        return seal, verdict

    return make


def compare(reference, candidate, **kwargs):
    return evidence.compare_run_metrics(
        *reference, *candidate, columns=("loss", "reward_mean"), expected_epochs=2, **kwargs
    )


def test_exact_independent_runs_match_with_different_output_directories(make_run):
    reference, candidate = make_run("reference"), make_run("candidate")
    result = compare(reference, candidate)
    assert result["matched"] is True
    assert result["reference"]["launch_id"] != result["candidate"]["launch_id"]
    assert result["protocol"]["expected_epochs"] == 2
    assert result["protocol"]["columns"] == ["loss", "reward_mean"]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("0,0.123456788,0.5\n1,0.234567891,0.6\n", "metric mismatch at epoch=0, loss"),
        ("0,nan,0.5\n1,0.234567891,0.6\n", "finite"),
        ("0,0.123456789,inf\n1,0.234567891,0.6\n", "finite"),
        ("0,0.123456789,0.5\n", "all expected epochs"),
        ("0,0.123456789,0.5\n2,0.234567891,0.6\n", "consecutive epochs"),
        ("0,0.123456789,0.5\n0,0.234567891,0.6\n", "consecutive epochs"),
        ("0,0.123456789,0.5,extra\n1,0.234567891,0.6\n", "consecutive epochs"),
    ],
)
def test_metric_drift_and_incomplete_curves_fail(make_run, rows, message):
    reference = make_run("reference")
    candidate = make_run("candidate", rows=rows)
    with pytest.raises(ValueError, match=message):
        compare(reference, candidate)


def test_signed_zero_is_not_collapsed(make_run):
    reference = make_run("reference", rows="0,0.0,0.5\n1,0.234567891,0.6\n")
    candidate = make_run("candidate", rows="0,-0.0,0.5\n1,0.234567891,0.6\n")
    with pytest.raises(ValueError, match="metric mismatch"):
        compare(reference, candidate)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"runtime_change": {"deterministic_algorithms": False}}, "strict deterministic"),
        ({"runtime_change": {"deterministic_warn_only": True}}, "strict deterministic"),
        ({"runtime_change": {"cudnn_benchmark": True}}, "benchmark must be disabled"),
        ({"runtime_change": {"packages": {"torch": "changed"}}}, "runtime differs"),
        (
            {
                "runtime_change": {
                    "matmul_reduced_precision_reduction": {"fp16": True, "bf16": False}
                }
            },
            "runtime differs",
        ),
        (
            {"runtime_change": {"environment": {"TORCHINDUCTOR_EMULATE_PRECISION_CASTS": "1"}}},
            "runtime differs",
        ),
        ({"model": "different"}, "model_identity differs"),
        ({"data_text": "changed prompts\n"}, "configured_data_files differs"),
        ({"code_change": {"dirty": True}}, "identified clean checkout"),
        ({"code_change": {"commit": "other"}}, "code differs"),
        ({"runtime_change": {"packages": {}}}, "complete runtime identity"),
        ({"full_precision": False}, "not bound"),
        ({"config_change": lambda c: c["trainer"].update(seed=18)}, "config differs"),
        (
            {"config_change": lambda c: c["trainer"].update(total_epochs=3)},
            "configured complete run",
        ),
    ],
)
def test_identity_and_protocol_drift_fail(make_run, kwargs, message):
    reference = make_run("reference")
    candidate = make_run("candidate", **kwargs)
    with pytest.raises(ValueError, match=message):
        compare(reference, candidate)


def test_same_run_cannot_be_its_own_reference(make_run):
    reference = make_run("reference")
    with pytest.raises(ValueError, match="independent"):
        compare(reference, reference)


def test_cli_preserves_reference_and_does_not_replace_report(make_run, tmp_path):
    reference, candidate = make_run("reference"), make_run("candidate")
    before = [path.read_bytes() for path in (*reference, *candidate)]
    report = tmp_path / "comparison.json"
    args = [
        "--reference-receipt",
        str(reference[0]),
        "--reference-verdict",
        str(reference[1]),
        "--candidate-receipt",
        str(candidate[0]),
        "--candidate-verdict",
        str(candidate[1]),
        "--columns",
        "loss",
        "reward_mean",
        "--expected-epochs",
        "2",
        "--report",
        str(report),
    ]
    main(args)
    assert json.loads(report.read_text())["matched"] is True
    assert [path.read_bytes() for path in (*reference, *candidate)] == before
    with pytest.raises(FileExistsError):
        main(args)
    report.unlink()
    candidate[1].write_text("{}")
    with pytest.raises(ValueError, match="unsupported run verdict"):
        main(args)
    assert not report.exists()
