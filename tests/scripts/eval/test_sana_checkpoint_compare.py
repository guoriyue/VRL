from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

import vrl.models.families.registry as model_families
from tests.scripts.eval.fixtures import (
    TinySanaPipeline,
    build_official_sana_scheduler,
    write_tiny_sana_snapshot,
)
from tests.trainers._checkpoint_helpers import _Trainer
from vrl.config.precision import PrecisionPolicy, RolePrecision
from vrl.models import checkpoint_identity
from vrl.models.families.sana.model import SanaModel
from vrl.scripts.eval import sana_checkpoint_compare as checkpoint_compare
from vrl.scripts.eval import sana_inference
from vrl.trainers.checkpointing import load_resolved_run_config, save_training_checkpoint

SANA_PRECISION = RolePrecision(
    dtype="fp16",
    float32_precision="ieee",
    outer_autocast=False,
)
SANA_IDENTITY = {"schema": "vrl.model-identity/v1", "sources": {}, "build": {}}


@pytest.fixture(autouse=True)
def _effective_ieee_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        checkpoint_compare,
        "float32_precision_state",
        lambda: {"matmul": "ieee", "cudnn": "ieee"},
    )


def _real_model(*, precision: RolePrecision = SANA_PRECISION) -> SanaModel:
    """A real ``SanaModel`` over tiny real modules at SANA's native dtype boundary."""

    pipeline = TinySanaPipeline()
    model = SanaModel(pipeline=pipeline, device=torch.device("cpu"))
    model.transformer.half()
    pipeline.text_encoder.to(torch.bfloat16)
    model.precision = precision
    return model


def _config(*, policy_dtype: str = "fp16", path: str = "test/sana") -> object:
    return OmegaConf.create(
        {
            "model": {
                "family": "sana",
                "path": path,
                "revision": None,
                "use_lora": False,
                "lora": None,
            },
            "precision": {
                "float32_precision": "ieee",
                "training": {"dtype": policy_dtype, "outer_autocast": False},
                "rollout": {
                    "dtype": policy_dtype,
                    "outer_autocast": False,
                    "prompt_encoders": {"dtype": "bf16"},
                },
            },
        },
    )


def _real_build(run_dir: Path):
    """The real registry entry's build and identity for the run's resolved config."""

    _, root = load_resolved_run_config(run_dir)
    entry = model_families.get_model_family_entry("sana")
    build = entry.resolve_model_build(
        root,
        torch.device("cpu"),
        precision=PrecisionPolicy.from_section(root.precision),
        for_rollout=True,
    )
    return entry, build, checkpoint_identity.resolve_checkpoint_model_identity(build)


def _checkpoint(tmp_path: Path, **meta_overrides) -> SimpleNamespace:
    checkpoint_dir = tmp_path / "checkpoint-final"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    checkpoint_path.write_bytes(b"full-parameter-state")
    meta = {
        "schema_version": 2,
        "family": "sana",
        "model_identity": SANA_IDENTITY,
        "uses_lora": False,
        "checkpoint_file_bytes": checkpoint_path.stat().st_size,
        **meta_overrides,
    }
    (checkpoint_dir / "checkpoint_meta.json").write_text(json.dumps(meta))
    return SimpleNamespace(
        checkpoint_path=checkpoint_path,
        payload={"family": "sana"},
        meta=meta,
    )


def test_parser_uses_official_sana_defaults() -> None:
    args = checkpoint_compare.build_parser().parse_args(["--run-dir", "run"])

    assert args.checkpoint == Path("checkpoint-final")
    assert args.prompt == "a red apple on a blue ceramic plate, studio photo"
    assert args.seed == 20260712
    assert (args.height, args.width) == (1024, 1024)
    assert args.steps == 20
    assert args.guidance_scale == 4.5


def test_run_rejects_structurally_invalid_family_before_checkpoint_lookup(
    tmp_path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = _config()
    cfg.model.family = "not-a-family"
    OmegaConf.save(cfg, run_dir / "resolved_config.yaml")

    with pytest.raises(ValueError, match="unsupported model family"):
        checkpoint_compare.run_comparison(
            checkpoint_compare.build_parser().parse_args(
                ["--run-dir", str(run_dir)],
            ),
        )


def test_run_generates_base_before_strict_restore_and_current(monkeypatch, tmp_path) -> None:
    """The real sana entry, loader, bundle, checkpoint restore and scheduler load.

    Only the HF pipeline load and diffusers' denoising call are doubles. The
    checkpoint is saved by ``save_training_checkpoint`` from the same tiny
    transformer with every weight set to 1.0, so the restore is observable in
    the weights: the first image must be painted while the transformer still
    holds its base weights, the second after ``restore_model_checkpoint``
    replaced them.
    """

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    snapshot = write_tiny_sana_snapshot(tmp_path / "sana-snapshot")
    OmegaConf.save(_config(path=str(snapshot)), run_dir / "resolved_config.yaml")
    pipeline = TinySanaPipeline()
    pipeline.install(monkeypatch, snapshot)

    entry, build, identity = _real_build(run_dir)
    bundle = entry.build_rollout(build)
    assert bundle.model.pipeline is pipeline
    base_state = {name: value.clone() for name, value in pipeline.transformer.state_dict().items()}
    with torch.no_grad():
        for parameter in pipeline.transformer.parameters():
            parameter.fill_(1.0)
    save_training_checkpoint(
        run_dir / "checkpoint-final",
        trainer=_Trainer(),
        bundle=bundle,
        family="sana",
        progress={"next_epoch": 1},
        rng_state={},
        model_identity=identity,
    )
    pipeline.transformer.load_state_dict(base_state)
    assert pipeline.holds() == "base"
    checkpoint_file = run_dir / "checkpoint-final" / "checkpoint.pt"

    result = checkpoint_compare.run_comparison(
        checkpoint_compare.build_parser().parse_args(
            [
                "--run-dir",
                str(run_dir),
                "--device",
                "cpu",
                "--height",
                "8",
                "--width",
                "10",
                "--steps",
                "2",
            ],
        ),
    )

    # Order witnessed by the weights: base image first, restored weights second,
    # and the restore was the strict full-parameter one (every weight is 1.0).
    assert [call["weights"] for call in pipeline.calls] == ["base", "restored"]
    assert pipeline.fingerprint() == float(
        sum(parameter.numel() for parameter in pipeline.transformer.parameters())
    )
    schedulers = [call["scheduler"] for call in pipeline.calls]
    assert len(schedulers) == 2
    assert schedulers[0] is not schedulers[1]
    for scheduler in schedulers:
        sana_inference.require_scheduler(scheduler)
    assert [call["generator_seed"] for call in pipeline.calls] == [20260712, 20260712]
    assert all(call["negative_prompt"] == "" for call in pipeline.calls)
    assert all(call["use_resolution_binning"] is True for call in pipeline.calls)
    assert all(call["max_sequence_length"] == 300 for call in pipeline.calls)

    base_path = Path(result["base"])
    current_path = Path(result["current"])
    side_by_side_path = Path(result["side_by_side"])
    assert Image.open(base_path).getpixel((0, 0)) == TinySanaPipeline.BASE_COLOR
    assert Image.open(current_path).getpixel((0, 0)) == TinySanaPipeline.RESTORED_COLOR
    assert Image.open(side_by_side_path).size == (20, 8)

    evaluation_record = json.loads(Path(result["evaluation_record"]).read_text(encoding="utf-8"))
    assert evaluation_record["schema"] == checkpoint_compare.REPORT_SCHEMA
    assert evaluation_record["execution"]["generation_order"] == ["base", "current"]
    assert evaluation_record["execution"]["checkpoint_loaded_between_images"] is True
    assert evaluation_record["execution"]["strict_trainable_state_restore"] is True
    assert evaluation_record["scheduler_protocol"] == checkpoint_compare.SANA_EVAL_SCHEDULER_CONFIG
    assert evaluation_record["checkpoint"]["meta"]["uses_lora"] is False
    assert (
        evaluation_record["checkpoint"]["sha256"]
        == hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
    )
    for name, path in (
        ("base", base_path),
        ("current", current_path),
        ("side_by_side", side_by_side_path),
    ):
        assert (
            evaluation_record["artifacts"][name]["sha256"]
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )


def test_run_refuses_to_mix_artifacts_with_an_existing_comparison(
    tmp_path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    OmegaConf.save(_config(), run_dir / "resolved_config.yaml")
    _checkpoint(run_dir)
    output_dir = run_dir / "sana_checkpoint_compare"
    output_dir.mkdir()
    (output_dir / "current.png").write_bytes(b"stale")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        checkpoint_compare.run_comparison(
            checkpoint_compare.build_parser().parse_args(
                ["--run-dir", str(run_dir), "--device", "cpu"],
            ),
        )

    assert (output_dir / "current.png").read_bytes() == b"stale"


def test_run_rejects_meta_identity_before_model_construction(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    snapshot = write_tiny_sana_snapshot(tmp_path / "sana-snapshot")
    OmegaConf.save(_config(path=str(snapshot)), run_dir / "resolved_config.yaml")
    _checkpoint(run_dir, model_identity={"schema": "wrong/v1"})
    pipeline = TinySanaPipeline()
    pipeline.install(monkeypatch, snapshot)

    with pytest.raises(ValueError, match="metadata model identity mismatch"):
        checkpoint_compare.run_comparison(
            checkpoint_compare.build_parser().parse_args(
                ["--run-dir", str(run_dir), "--device", "cpu"],
            ),
        )

    assert pipeline.loads == 0


def test_run_rejects_model_source_drift_before_generation(monkeypatch, tmp_path) -> None:
    """A model directory that changes under the loader is caught before any image."""

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    snapshot = write_tiny_sana_snapshot(tmp_path / "sana-snapshot")
    OmegaConf.save(_config(path=str(snapshot)), run_dir / "resolved_config.yaml")
    _, _, identity = _real_build(run_dir)
    _checkpoint(run_dir, model_identity=identity)
    pipeline = TinySanaPipeline()

    def mutate_snapshot() -> None:
        (snapshot / "extra-weights.bin").write_bytes(b"drift")

    pipeline.install(monkeypatch, snapshot, on_load=mutate_snapshot)

    with pytest.raises(
        RuntimeError,
        match="SANA model source changed during runtime construction",
    ):
        checkpoint_compare.run_comparison(
            checkpoint_compare.build_parser().parse_args(
                ["--run-dir", str(run_dir), "--device", "cpu"],
            ),
        )

    assert pipeline.loads == 1
    assert pipeline.calls == []
    assert not (run_dir / "sana_checkpoint_compare").exists()


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        pytest.param("model.family", "wan", "model.family", id="non-sana"),
        pytest.param("model.use_lora", True, "full-parameter", id="lora-enabled"),
        pytest.param("model.lora", {"rank": 16}, "full-parameter", id="lora-config"),
    ],
)
def test_config_protocol_rejects_out_of_scope_runs(path, value, message) -> None:
    cfg = _config()
    OmegaConf.update(cfg, path, value, merge=False)

    with pytest.raises(ValueError, match=message):
        checkpoint_compare._validate_resolved_config(cfg)


def test_config_protocol_accepts_full_parameter_sana_run() -> None:
    checkpoint_compare._validate_resolved_config(_config())


@pytest.mark.parametrize(
    ("meta_overrides", "message"),
    [
        pytest.param({"uses_lora": True}, "uses_lora=false", id="lora"),
        pytest.param({"uses_lora": None}, "uses_lora=false", id="missing-lora-proof"),
        pytest.param(
            {"checkpoint_file_bytes": 1},
            "byte count",
            id="byte-count",
        ),
    ],
)
def test_checkpoint_protocol_rejects_wrong_identity(
    tmp_path,
    meta_overrides,
    message,
) -> None:
    checkpoint = _checkpoint(tmp_path, **meta_overrides)

    with pytest.raises(ValueError, match=message):
        checkpoint_compare._validate_checkpoint(checkpoint)


def test_checkpoint_protocol_accepts_full_parameter_sana(tmp_path) -> None:
    checkpoint_compare._validate_checkpoint(_checkpoint(tmp_path))


@pytest.mark.parametrize(
    ("target", "value", "record_key", "expected"),
    [
        pytest.param(
            "transformer.dtype",
            torch.bfloat16,
            "transformer",
            "bfloat16",
            id="transformer-bf16",
        ),
        pytest.param(
            "pipeline.text_encoder.dtype",
            torch.float16,
            "prompt_encoder",
            "float16",
            id="prompt-fp16",
        ),
        pytest.param(
            "pipeline.vae.dtype",
            torch.float16,
            "vae",
            "float16",
            id="vae-fp16",
        ),
    ],
)
def test_model_precision_snapshot_records_materialized_dtypes(
    target,
    value,
    record_key,
    expected,
) -> None:
    model = _real_model()
    owner = model
    for part in target.split(".")[:-1]:
        owner = getattr(owner, part)
    owner.to(value)

    actual = checkpoint_compare._model_precision_snapshot(model)

    assert actual[record_key] == expected


def test_model_precision_snapshot_records_native_configured_boundary() -> None:
    actual = checkpoint_compare._model_precision_snapshot(_real_model())

    assert actual["transformer"] == "float16"
    assert actual["prompt_encoder"] == "bfloat16"
    assert actual["vae"] == "float32"
    assert actual["outer_autocast"] is False
    assert actual["effective_float32_precision"] == {
        "matmul": "ieee",
        "cudnn": "ieee",
    }


def test_model_precision_snapshot_records_configured_outer_autocast() -> None:
    model = _real_model(
        precision=RolePrecision(
            dtype="fp16",
            float32_precision="ieee",
            outer_autocast=True,
        ),
    )

    actual = checkpoint_compare._model_precision_snapshot(model)

    assert actual["outer_autocast"] is True


@pytest.mark.parametrize("count", [True, 1.5, "2", 0, -1])
def test_generation_rejects_invalid_image_count_before_pipeline_access(count) -> None:
    with pytest.raises(ValueError, match="num_images"):
        sana_inference.generate_prompt_images(
            None,
            scheduler=None,
            prompt="draw",
            seed=0,
            num_images=count,
            device=torch.device("cpu"),
        )


def test_generation_preserves_device_autocast_query_failure(monkeypatch) -> None:
    failure = TypeError("device autocast query failed")
    calls = []

    def query(*args):
        calls.append(args)
        if args:
            raise failure
        return False

    monkeypatch.setattr(torch, "is_autocast_enabled", query)
    with pytest.raises(TypeError) as caught:
        sana_inference.generate_prompt_images(
            None,
            scheduler=None,
            prompt="draw",
            seed=0,
            num_images=1,
            device=torch.device("cpu"),
        )
    assert caught.value is failure
    assert calls == [("cpu",)]


def test_generation_rejects_an_active_outer_autocast() -> None:
    model = _real_model()

    with (
        torch.autocast("cpu", dtype=torch.bfloat16),
        pytest.raises(RuntimeError, match="without an outer autocast"),
    ):
        checkpoint_compare._generate_one(
            model,
            scheduler=build_official_sana_scheduler(),
            prompt="test",
            seed=1,
            height=8,
            width=8,
            steps=1,
            guidance_scale=4.5,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    ("key", "wrong_value"),
    [
        pytest.param("algorithm_type", "sde-dpmsolver++", id="algorithm"),
        pytest.param("use_flow_sigmas", False, id="flow-sigmas"),
    ],
)
def test_scheduler_protocol_rejects_wrong_config(key, wrong_value) -> None:
    scheduler = build_official_sana_scheduler(**{key: wrong_value})

    with pytest.raises(ValueError, match="official DPM-Solver"):
        sana_inference.require_scheduler(scheduler)


def test_scheduler_protocol_rejects_wrong_class() -> None:
    class FlowMatchEulerDiscreteScheduler:
        config = build_official_sana_scheduler().config

    with pytest.raises(ValueError, match="official DPM-Solver"):
        sana_inference.require_scheduler(FlowMatchEulerDiscreteScheduler())


def test_scheduler_protocol_accepts_official_identity() -> None:
    assert (
        sana_inference.require_scheduler(build_official_sana_scheduler())
        == checkpoint_compare.SANA_EVAL_SCHEDULER_CONFIG
    )
