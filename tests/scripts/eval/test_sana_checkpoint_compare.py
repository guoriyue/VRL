from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

import vrl.families.registry as model_families
from vrl.config.precision import RolePrecision
from vrl.models import checkpoint_identity
from vrl.models.interfaces.runtime import ModelBuild
from vrl.scripts.eval import sana_checkpoint_compare as checkpoint_compare
from vrl.scripts.eval import sana_inference

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


class DPMSolverMultistepScheduler:
    def __init__(self, **overrides) -> None:
        config = dict(checkpoint_compare.SCHEDULER_PROTOCOL)
        config.pop("class_name")
        config.update(overrides)
        self.config = config


class _FakePipeline:
    def __init__(self, events: list[str]) -> None:
        self.text_encoder = SimpleNamespace(dtype=torch.bfloat16)
        self.vae = SimpleNamespace(dtype=torch.float32)
        self.scheduler = None
        self.events = events
        self.current = False
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        assert torch.is_inference_mode_enabled()
        assert not torch.is_autocast_enabled("cpu")
        assert "complex_human_instruction" not in kwargs
        self.events.append("generate:current" if self.current else "generate:base")
        self.calls.append(
            {
                **kwargs,
                "scheduler": self.scheduler,
                "generator_seed": kwargs["generator"].initial_seed(),
            },
        )
        color = (200, 10, 20) if self.current else (10, 20, 200)
        image = Image.new("RGB", (kwargs["width"], kwargs["height"]), color=color)
        return SimpleNamespace(images=[image])


class _FakeModel:
    def __init__(
        self,
        events: list[str],
        *,
        transformer_dtype: torch.dtype = torch.float16,
        precision: RolePrecision = SANA_PRECISION,
    ) -> None:
        self.transformer = SimpleNamespace(dtype=transformer_dtype)
        self.pipeline = _FakePipeline(events)
        self.precision = precision

    def eval(self):
        return self


def _config(*, policy_dtype: str = "fp16") -> object:
    return OmegaConf.create(
        {
            "model": {
                "family": "sana",
                "path": "test/sana",
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


def test_run_generates_base_before_strict_restore_and_current(
    monkeypatch,
    tmp_path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    OmegaConf.save(_config(), run_dir / "resolved_config.yaml")
    checkpoint = _checkpoint(run_dir)
    events: list[str] = []
    model = _FakeModel(events)
    bundle = SimpleNamespace(
        model=model,
        trainable_modules={"transformer": model.transformer},
        precision=SANA_PRECISION,
    )
    expected_bundle = bundle
    build = ModelBuild(
        model_name_or_path="test/sana",
        revision=None,
        device="cpu",
        parameter_dtype=torch.float16,
        family="sana",
        precision=SANA_PRECISION,
        model_config={},
    )
    schedulers: list[DPMSolverMultistepScheduler] = []

    def fake_load_checkpoint(path):
        assert Path(path) == run_dir / "checkpoint-final"
        events.append("read:checkpoint")
        return checkpoint

    def fake_restore(
        actual_checkpoint,
        *,
        bundle: object,
        family: str,
        expected_model_identity: dict,
        strict: bool,
    ):
        assert actual_checkpoint is checkpoint
        assert bundle is expected_bundle
        assert family == "sana"
        assert expected_model_identity == SANA_IDENTITY
        assert strict is True
        events.append("load:strict")
        model.pipeline.current = True

    def fake_load_scheduler(actual_build):
        assert actual_build is build
        scheduler = DPMSolverMultistepScheduler()
        schedulers.append(scheduler)
        sana_inference.validate_scheduler(scheduler)
        return scheduler

    entry = SimpleNamespace(
        resolve_model_build=lambda *args, **kwargs: build,
        build_rollout=lambda value: bundle,
    )
    monkeypatch.setattr(checkpoint_compare, "load_training_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(checkpoint_compare, "restore_model_checkpoint", fake_restore)
    monkeypatch.setattr(
        checkpoint_identity,
        "resolve_checkpoint_model_identity",
        lambda actual_build: SANA_IDENTITY,
    )
    monkeypatch.setattr(model_families, "get_model_family_entry", lambda family: entry)
    monkeypatch.setattr(checkpoint_compare, "_load_official_scheduler", fake_load_scheduler)
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

    assert events == [
        "generate:base",
        "read:checkpoint",
        "load:strict",
        "generate:current",
    ]
    assert len(schedulers) == 2
    assert schedulers[0] is not schedulers[1]
    assert [call["scheduler"] for call in model.pipeline.calls] == schedulers
    assert [call["generator_seed"] for call in model.pipeline.calls] == [20260712, 20260712]
    assert all(call["negative_prompt"] == "" for call in model.pipeline.calls)
    assert all(call["use_resolution_binning"] is True for call in model.pipeline.calls)
    assert all(call["max_sequence_length"] == 300 for call in model.pipeline.calls)

    base_path = Path(result["base"])
    current_path = Path(result["current"])
    side_by_side_path = Path(result["side_by_side"])
    assert Image.open(base_path).getpixel((0, 0)) == (10, 20, 200)
    assert Image.open(current_path).getpixel((0, 0)) == (200, 10, 20)
    assert Image.open(side_by_side_path).size == (20, 8)

    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["schema"] == checkpoint_compare.REPORT_SCHEMA
    assert manifest["execution"]["generation_order"] == ["base", "current"]
    assert manifest["execution"]["checkpoint_loaded_between_images"] is True
    assert manifest["execution"]["strict_trainable_state_restore"] is True
    assert manifest["scheduler_protocol"] == checkpoint_compare.SCHEDULER_PROTOCOL
    assert manifest["checkpoint"]["meta"]["uses_lora"] is False
    assert manifest["checkpoint"]["sha256"] == checkpoint_compare._sha256(
        checkpoint.checkpoint_path,
    )
    for name, path in (
        ("base", base_path),
        ("current", current_path),
        ("side_by_side", side_by_side_path),
    ):
        assert manifest["artifacts"][name]["sha256"] == checkpoint_compare._sha256(path)


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
    OmegaConf.save(_config(), run_dir / "resolved_config.yaml")
    _checkpoint(run_dir, model_identity={"schema": "wrong/v1"})
    build = SimpleNamespace()
    built = False

    def fail_if_built(_build):
        nonlocal built
        built = True
        raise AssertionError("model construction must not run")

    monkeypatch.setattr(
        model_families,
        "get_model_family_entry",
        lambda family: SimpleNamespace(
            resolve_model_build=lambda *args, **kwargs: build,
            build_rollout=fail_if_built,
        ),
    )
    monkeypatch.setattr(
        checkpoint_identity,
        "resolve_checkpoint_model_identity",
        lambda actual_build: SANA_IDENTITY,
    )

    with pytest.raises(ValueError, match="metadata model identity mismatch"):
        checkpoint_compare.run_comparison(
            checkpoint_compare.build_parser().parse_args(
                ["--run-dir", str(run_dir), "--device", "cpu"],
            ),
        )

    assert built is False


def test_run_rejects_model_source_drift_before_generation(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    OmegaConf.save(_config(), run_dir / "resolved_config.yaml")
    _checkpoint(run_dir)
    build = object()
    built = False
    identities = iter(
        (
            SANA_IDENTITY,
            {"schema": "vrl.model-identity/v1", "sources": {"main": "changed"}},
        ),
    )

    def build_bundle(actual_build):
        nonlocal built
        assert actual_build is build
        built = True
        return SimpleNamespace(model=object())

    monkeypatch.setattr(
        model_families,
        "get_model_family_entry",
        lambda family: SimpleNamespace(
            resolve_model_build=lambda *args, **kwargs: build,
            build_rollout=build_bundle,
        ),
    )
    monkeypatch.setattr(
        checkpoint_identity,
        "resolve_checkpoint_model_identity",
        lambda actual_build: next(identities) if actual_build is build else None,
    )

    with pytest.raises(
        RuntimeError,
        match="SANA model source changed during runtime construction",
    ):
        checkpoint_compare.run_comparison(
            checkpoint_compare.build_parser().parse_args(
                ["--run-dir", str(run_dir), "--device", "cpu"],
            ),
        )

    assert built is True
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


@pytest.mark.parametrize("policy_dtype", ["bf16", "fp32"])
def test_config_protocol_leaves_precision_to_resolved_yaml(policy_dtype: str) -> None:
    checkpoint_compare._validate_resolved_config(_config(policy_dtype=policy_dtype))


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
    model = _FakeModel([])
    owner = model
    parts = target.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    setattr(owner, parts[-1], value)

    actual = checkpoint_compare._model_precision_snapshot(model)

    assert actual[record_key] == expected


def test_model_precision_snapshot_records_native_configured_boundary() -> None:
    actual = checkpoint_compare._model_precision_snapshot(_FakeModel([]))

    assert actual["transformer"] == "float16"
    assert actual["prompt_encoder"] == "bfloat16"
    assert actual["vae"] == "float32"
    assert actual["outer_autocast"] is False
    assert actual["effective_float32_precision"] == {
        "matmul": "ieee",
        "cudnn": "ieee",
    }


def test_model_precision_snapshot_records_configured_outer_autocast() -> None:
    model = _FakeModel(
        [],
        precision=RolePrecision(
            dtype="fp16",
            float32_precision="ieee",
            outer_autocast=True,
        ),
    )

    actual = checkpoint_compare._model_precision_snapshot(model)

    assert actual["outer_autocast"] is True


def test_model_precision_snapshot_records_effective_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        checkpoint_compare,
        "float32_precision_state",
        lambda: {"matmul": "tf32", "cudnn": "ieee"},
    )

    actual = checkpoint_compare._model_precision_snapshot(_FakeModel([]))

    assert actual["effective_float32_precision"] == {
        "matmul": "tf32",
        "cudnn": "ieee",
    }


def test_generation_rejects_an_active_outer_autocast() -> None:
    model = _FakeModel([])

    with (
        torch.autocast("cpu", dtype=torch.bfloat16),
        pytest.raises(RuntimeError, match="without an outer autocast"),
    ):
        checkpoint_compare._generate_one(
            model,
            scheduler=DPMSolverMultistepScheduler(),
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
        pytest.param("solver_order", 3, id="order"),
        pytest.param("solver_type", "heun", id="solver"),
        pytest.param("use_flow_sigmas", False, id="flow-sigmas"),
        pytest.param("flow_shift", 1.0, id="flow-shift"),
        pytest.param("prediction_type", "epsilon", id="prediction"),
    ],
)
def test_scheduler_protocol_rejects_wrong_config(key, wrong_value) -> None:
    scheduler = DPMSolverMultistepScheduler(**{key: wrong_value})

    with pytest.raises(ValueError, match="official DPM-Solver"):
        sana_inference.validate_scheduler(scheduler)


def test_scheduler_protocol_rejects_wrong_class() -> None:
    class FlowMatchEulerDiscreteScheduler:
        config = DPMSolverMultistepScheduler().config

    with pytest.raises(ValueError, match="official DPM-Solver"):
        sana_inference.validate_scheduler(FlowMatchEulerDiscreteScheduler())


def test_scheduler_protocol_accepts_official_identity() -> None:
    assert (
        sana_inference.validate_scheduler(DPMSolverMultistepScheduler())
        == checkpoint_compare.SCHEDULER_PROTOCOL
    )
