from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from vrl.models.diffusion import build as diffusion_build
from vrl.models.interfaces.runtime import ForwardPrecision
from vrl.scripts.eval import sana_checkpoint_compare as checkpoint_compare
from vrl.scripts.eval import sana_inference

SANA_FORWARD_PRECISION = ForwardPrecision(
    autocast="off",
    float32_precision="ieee",
)


@pytest.fixture(autouse=True)
def _effective_ieee_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sana_inference,
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
    ) -> None:
        self.transformer = SimpleNamespace(dtype=transformer_dtype)
        self.pipeline = _FakePipeline(events)

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
                "training": {"dtype": policy_dtype},
                "rollout": {
                    "dtype": policy_dtype,
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
        "family": "sana",
        "uses_lora": False,
        "checkpoint_file_bytes": checkpoint_path.stat().st_size,
        **meta_overrides,
    }
    return SimpleNamespace(
        checkpoint_path=checkpoint_path,
        payload={"family": "sana"},
        meta=meta,
        trainable_state={"transformer": {"weight": torch.ones(1)}},
    )


def test_parser_uses_official_sana_defaults() -> None:
    args = checkpoint_compare.build_parser().parse_args(["--run-dir", "run"])

    assert args.checkpoint == Path("checkpoint-final")
    assert args.prompt == "a red apple on a blue ceramic plate, studio photo"
    assert args.seed == 20260712
    assert (args.height, args.width) == (1024, 1024)
    assert args.steps == 20
    assert args.guidance_scale == 4.5


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
        forward_precision=SANA_FORWARD_PRECISION,
    )
    build = SimpleNamespace(
        model_name_or_path="test/sana",
        model_config={"revision": None},
        parameter_dtype=torch.float16,
    )
    schedulers: list[DPMSolverMultistepScheduler] = []

    def fake_load_checkpoint(path):
        assert Path(path) == run_dir / "checkpoint-final"
        events.append("read:checkpoint")
        return checkpoint

    def fake_load_state(actual_bundle, state, *, strict):
        assert actual_bundle is bundle
        assert state is checkpoint.trainable_state
        assert strict is True
        events.append("load:strict")
        model.pipeline.current = True

    def fake_load_scheduler(actual_build):
        assert actual_build is build
        scheduler = DPMSolverMultistepScheduler()
        schedulers.append(scheduler)
        checkpoint_compare._validate_scheduler(scheduler)
        return scheduler

    monkeypatch.setattr(checkpoint_compare, "load_training_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(checkpoint_compare, "load_trainable_state", fake_load_state)
    monkeypatch.setattr(
        diffusion_build, "resolve_family_model_build", lambda *args, **kwargs: build
    )
    monkeypatch.setattr(diffusion_build, "build_family_runtime_bundle", lambda value: bundle)
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


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        pytest.param("model.family", "wan", "model.family", id="non-sana"),
        pytest.param("model.use_lora", True, "full-parameter", id="lora-enabled"),
        pytest.param("model.lora", {"rank": 16}, "full-parameter", id="lora-config"),
        pytest.param("precision.training.dtype", "bf16", "matching", id="training-bf16"),
        pytest.param("precision.rollout.dtype", "bf16", "matching", id="rollout-bf16"),
        pytest.param(
            "precision.rollout.prompt_encoders.dtype",
            "fp16",
            "BF16 prompt encoder",
            id="prompt-fp16",
        ),
        pytest.param(
            "precision.rollout.quantization",
            {"format": "fp8", "recipe": "rowwise"},
            "quantized",
            id="rollout-quantized",
        ),
    ],
)
def test_config_protocol_rejects_non_native_runs(path, value, message) -> None:
    cfg = _config()
    OmegaConf.update(cfg, path, value, merge=False)

    with pytest.raises(ValueError, match=message):
        checkpoint_compare._validate_resolved_config(cfg)


def test_config_protocol_accepts_aligned_native_fp16_full_parameter_run() -> None:
    checkpoint_compare._validate_resolved_config(_config())


@pytest.mark.parametrize("policy_dtype", ["bf16", "fp32"])
def test_config_protocol_rejects_aligned_non_native_run(policy_dtype: str) -> None:
    with pytest.raises(ValueError, match="native-FP16"):
        checkpoint_compare._validate_resolved_config(_config(policy_dtype=policy_dtype))


@pytest.mark.parametrize(
    ("payload_family", "meta_overrides", "message"),
    [
        pytest.param("wan", {}, "payload family", id="payload-family"),
        pytest.param("sana", {"family": "wan"}, "metadata family", id="metadata-family"),
        pytest.param("sana", {"uses_lora": True}, "uses_lora=false", id="lora"),
        pytest.param("sana", {"uses_lora": None}, "uses_lora=false", id="missing-lora-proof"),
        pytest.param(
            "sana",
            {"checkpoint_file_bytes": 1},
            "byte count",
            id="byte-count",
        ),
    ],
)
def test_checkpoint_protocol_rejects_wrong_identity(
    tmp_path,
    payload_family,
    meta_overrides,
    message,
) -> None:
    checkpoint = _checkpoint(tmp_path, **meta_overrides)
    checkpoint.payload["family"] = payload_family

    with pytest.raises(ValueError, match=message):
        checkpoint_compare._validate_checkpoint(checkpoint)


def test_checkpoint_protocol_accepts_full_parameter_sana(tmp_path) -> None:
    checkpoint_compare._validate_checkpoint(_checkpoint(tmp_path))


@pytest.mark.parametrize(
    ("target", "value"),
    [
        pytest.param("transformer.dtype", torch.bfloat16, id="transformer-bf16"),
        pytest.param("pipeline.text_encoder.dtype", torch.float16, id="prompt-fp16"),
        pytest.param("pipeline.vae.dtype", torch.float16, id="vae-fp16"),
    ],
)
def test_model_precision_rejects_non_native_boundary(target, value) -> None:
    model = _FakeModel([])
    owner = model
    parts = target.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    setattr(owner, parts[-1], value)

    with pytest.raises(ValueError, match="precision boundary mismatch"):
        checkpoint_compare._validate_model_precision(model, SANA_FORWARD_PRECISION)


def test_model_precision_accepts_native_fp16_no_autocast_boundary() -> None:
    actual = checkpoint_compare._validate_model_precision(
        _FakeModel([]),
        SANA_FORWARD_PRECISION,
    )

    assert actual["transformer"] == "float16"
    assert actual["prompt_encoder"] == "bfloat16"
    assert actual["vae"] == "float32"
    assert actual["forward_precision"] == {
        "autocast": "off",
        "float32_precision": "ieee",
    }
    assert actual["effective_float32_precision"] == {
        "matmul": "ieee",
        "cudnn": "ieee",
    }


def test_model_precision_rejects_fp32_transformer() -> None:
    with pytest.raises(ValueError, match="precision boundary mismatch"):
        checkpoint_compare._validate_model_precision(
            _FakeModel([], transformer_dtype=torch.float32),
            SANA_FORWARD_PRECISION,
        )


@pytest.mark.parametrize(
    "forward_precision",
    [
        ForwardPrecision(autocast="bf16", float32_precision="ieee"),
        ForwardPrecision(autocast="off", float32_precision="tf32"),
    ],
)
def test_model_precision_rejects_wrong_resolved_contract(
    forward_precision: ForwardPrecision,
) -> None:
    with pytest.raises(ValueError, match="precision boundary mismatch"):
        checkpoint_compare._validate_model_precision(_FakeModel([]), forward_precision)


def test_model_precision_rejects_effective_backend_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sana_inference,
        "float32_precision_state",
        lambda: {"matmul": "tf32", "cudnn": "ieee"},
    )

    with pytest.raises(ValueError, match="precision boundary mismatch"):
        checkpoint_compare._validate_model_precision(
            _FakeModel([]),
            SANA_FORWARD_PRECISION,
        )


def test_generation_rejects_an_active_outer_autocast() -> None:
    model = _FakeModel([])

    with (
        torch.autocast("cpu", dtype=torch.bfloat16),
        pytest.raises(RuntimeError, match="without an outer autocast"),
    ):
        checkpoint_compare._generate_one(
            model,
            forward_precision=SANA_FORWARD_PRECISION,
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
        checkpoint_compare._validate_scheduler(scheduler)


def test_scheduler_protocol_rejects_wrong_class() -> None:
    class FlowMatchEulerDiscreteScheduler:
        config = DPMSolverMultistepScheduler().config

    with pytest.raises(ValueError, match="official DPM-Solver"):
        checkpoint_compare._validate_scheduler(FlowMatchEulerDiscreteScheduler())


def test_scheduler_protocol_accepts_official_identity() -> None:
    assert (
        checkpoint_compare._validate_scheduler(DPMSolverMultistepScheduler())
        == checkpoint_compare.SCHEDULER_PROTOCOL
    )
