from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from vrl.config.loading import load_config
from vrl.models.interfaces.runtime import ForwardPrecision
from vrl.scripts.eval import sana_aesthetic_checkpoint_eval as checkpoint_eval
from vrl.scripts.eval import sana_inference

SANA_FORWARD_PRECISION = ForwardPrecision(
    autocast="off",
    float32_precision="ieee",
)


def _write_run(tmp_path: Path, *, empty_manifest: bool = False) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = tmp_path / "eval.txt"
    manifest.write_text("" if empty_manifest else "a red fox\n", encoding="utf-8")
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "sana",
                "path": "test/sana",
                "revision": "model-revision",
                "use_lora": False,
                "lora": None,
            },
            "data": {"eval_manifest": str(manifest)},
            "precision": "bf16",
            "sampling": {
                "width": 8,
                "height": 8,
                "num_steps": 1,
                "guidance_scale": 4.5,
                "max_sequence_length": 8,
            },
            "rollout": {
                "denoise_mode": "sde",
                "noise_level": 0.7,
                "sde": {"type": "flow_grpo", "window_size": 0},
            },
            "reward": {
                "components": {"aesthetic": 1.0, "pickscore": 0.0},
                "kwargs": {
                    "aesthetic": {
                        "model_name": "test/aesthetic",
                        "model_revision": "aesthetic-revision",
                    },
                    "pickscore": {
                        "device": "cpu",
                        "model_name": "test/pickscore",
                        "model_revision": "pickscore-revision",
                        "processor_name": "test/processor",
                        "processor_revision": "processor-revision",
                    },
                },
            },
            "trainer": {"save_freq": 25, "total_epochs": 25},
        },
    )
    OmegaConf.save(cfg, run_dir / "resolved_config.yaml")
    (run_dir / "metrics.csv").write_text(
        "epoch,loss\n" + "".join(f"{epoch},1.0\n" for epoch in range(25)),
        encoding="utf-8",
    )
    checkpoint = run_dir / "checkpoint-25"
    checkpoint.mkdir()
    payload = b"checkpoint-state"
    (checkpoint / "checkpoint.pt").write_bytes(payload)
    (checkpoint / "checkpoint_meta.json").write_text(
        json.dumps(
            {
                "family": "sana",
                "completed_epoch": 25,
                "checkpoint_file_bytes": len(payload),
                "uses_lora": False,
            },
        ),
        encoding="utf-8",
    )
    return run_dir


def _allow_minimal_protocol(monkeypatch) -> None:
    monkeypatch.setattr(checkpoint_eval, "_normalize_run_config", lambda cfg: cfg)
    monkeypatch.setattr(checkpoint_eval, "_materialize_model_snapshot", lambda cfg: cfg)
    monkeypatch.setattr(
        checkpoint_eval,
        "_materialize_reward_model_snapshots",
        lambda reward_models: reward_models,
    )
    monkeypatch.setattr(
        checkpoint_eval,
        "_validate_training_log_provenance",
        lambda run_dir, cfg: {
            "path": "supervisor.log",
            "sha256": "test-log",
            "resolved_model_revisions": {},
        },
    )

    def resolve_manifests(cfg):
        path = Path(str(cfg.data.eval_manifest)).resolve()
        prompts = [
            line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        return path, path, prompts

    monkeypatch.setattr(checkpoint_eval, "_resolve_protocol_manifests", resolve_manifests)


def test_main_writes_provenance_bound_report(monkeypatch, tmp_path, capsys) -> None:
    run_dir = _write_run(tmp_path)
    _allow_minimal_protocol(monkeypatch)

    def fake_generate(cfg, targets, prompts, *, output_dir, sampling, device):
        del cfg, sampling, device
        generated = []
        for target in targets:
            for sample_index in range(checkpoint_eval.EVAL_SAMPLES_PER_PROMPT):
                path = output_dir / "images" / target.label / f"sample{sample_index}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{target.label}-{sample_index}".encode())
                generated.append(
                    checkpoint_eval.GeneratedImage(
                        checkpoint_label=target.label,
                        epoch=target.epoch,
                        prompt_index=0,
                        sample_index=sample_index,
                        group_seed=checkpoint_eval._group_seed(0),
                        prompt=prompts[0],
                        path=path.resolve(),
                        image_sha256=checkpoint_eval._sha256(path),
                    ),
                )
        return generated

    def fake_score(generated, rewards):
        assert [reward.name for reward in rewards] == ["aesthetic", "pickscore"]
        return [
            {
                "checkpoint_label": image.checkpoint_label,
                "epoch": image.epoch,
                "prompt_index": image.prompt_index,
                "sample_index": image.sample_index,
                "group_seed": image.group_seed,
                "prompt": image.prompt,
                "image_path": str(image.path),
                "image_sha256": image.image_sha256,
                "r_aesthetic": 5.0 + image.epoch / 1000,
                "r_pickscore": 0.8,
            }
            for image in generated
        ]

    monkeypatch.setattr(checkpoint_eval, "_generate_images", fake_generate)
    monkeypatch.setattr(checkpoint_eval, "_score_images", fake_score)
    checkpoint_eval.main(["--run-dir", str(run_dir), "--device", "cpu"])

    rows = checkpoint_eval.load_report_metrics(run_dir)
    assert [row["epoch"] for row in rows] == [-1.0, 25.0]
    assert all(row["sample_count"] == 2.0 for row in rows)
    payload = json.loads((run_dir / checkpoint_eval.REPORT_RELATIVE_PATH).read_text())
    assert payload["schema"] == checkpoint_eval.REPORT_SCHEMA
    assert payload["schema_version"] == checkpoint_eval.REPORT_SCHEMA_VERSION
    assert payload["provenance"]["seed_grid"]["base_seed"] == 20260710
    assert payload["provenance"]["evaluation_curve"] == {
        "checkpoint_interval": 25,
    }
    rewards = payload["provenance"]["rewards"]
    assert rewards[0]["identity"]["model"]["repo"] == "test/aesthetic"
    assert rewards[1]["identity"]["processor"]["repo"] == "test/processor"
    assert rewards[1]["identity"]["model"]["repo"] == "test/pickscore"
    checkpoints = payload["provenance"]["checkpoints"]
    assert checkpoints[0] == {
        "label": "baseline",
        "epoch": -1,
        "source": "pinned_base_model_snapshot",
        "checkpoint_loaded": False,
    }
    assert checkpoints[1]["path"] == "checkpoint-25"
    assert "curve_points" in capsys.readouterr().out

    checkpoint_path = run_dir / "checkpoint-25/checkpoint.pt"
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_path.write_bytes(b"X" + checkpoint_bytes[1:])
    with pytest.raises(ValueError, match="checkpoint provenance"):
        checkpoint_eval.load_report_metrics(run_dir)
    checkpoint_path.write_bytes(checkpoint_bytes)

    metrics_path = run_dir / "metrics.csv"
    metrics_text = metrics_path.read_text(encoding="utf-8")
    metrics_path.write_text(metrics_text.replace("1.0", "2.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="training metrics provenance hash changed"):
        checkpoint_eval.load_report_metrics(run_dir)
    metrics_path.write_text(metrics_text, encoding="utf-8")

    payload["metrics"] = []
    (run_dir / checkpoint_eval.REPORT_RELATIVE_PATH).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="report has no metric rows"):
        checkpoint_eval.load_report_metrics(run_dir)


def test_report_reader_rejects_empty_metrics(monkeypatch, tmp_path) -> None:
    run_dir = _write_run(tmp_path)
    _allow_minimal_protocol(monkeypatch)
    monkeypatch.setattr(checkpoint_eval, "_generate_images", lambda *args, **kwargs: [])
    monkeypatch.setattr(checkpoint_eval, "_score_images", lambda *args, **kwargs: [])

    with pytest.raises(ValueError, match=r"empty.*sample manifest"):
        checkpoint_eval.main(["--run-dir", str(run_dir), "--device", "cpu"])


def test_report_reader_rejects_changed_config_provenance(monkeypatch, tmp_path) -> None:
    run_dir = _write_run(tmp_path)
    _allow_minimal_protocol(monkeypatch)

    def fake_generate(cfg, targets, prompts, *, output_dir, sampling, device):
        del cfg, sampling, device
        images = []
        for target in targets:
            for sample_index in range(2):
                path = output_dir / f"{target.label}-{sample_index}.png"
                path.write_bytes(b"image")
                images.append(
                    checkpoint_eval.GeneratedImage(
                        target.label,
                        target.epoch,
                        0,
                        sample_index,
                        checkpoint_eval._group_seed(0),
                        prompts[0],
                        path,
                        checkpoint_eval._sha256(path),
                    ),
                )
        return images

    monkeypatch.setattr(checkpoint_eval, "_generate_images", fake_generate)

    def fake_score(images, rewards):
        del rewards
        return [
            {
                "checkpoint_label": image.checkpoint_label,
                "epoch": image.epoch,
                "prompt_index": image.prompt_index,
                "sample_index": image.sample_index,
                "group_seed": image.group_seed,
                "prompt": image.prompt,
                "image_path": str(image.path),
                "image_sha256": image.image_sha256,
                "r_aesthetic": 5.0,
                "r_pickscore": 0.8,
            }
            for image in images
        ]

    monkeypatch.setattr(checkpoint_eval, "_score_images", fake_score)
    checkpoint_eval.main(["--run-dir", str(run_dir), "--device", "cpu"])
    with (run_dir / "resolved_config.yaml").open("a", encoding="utf-8") as handle:
        handle.write("\n# changed\n")

    with pytest.raises(ValueError, match="resolved config provenance hash changed"):
        checkpoint_eval.load_report_metrics(run_dir)


def test_main_rejects_empty_manifest(monkeypatch, tmp_path) -> None:
    run_dir = _write_run(tmp_path, empty_manifest=True)
    _allow_minimal_protocol(monkeypatch)
    with pytest.raises(ValueError, match="manifest has no prompts"):
        checkpoint_eval.main(["--run-dir", str(run_dir), "--device", "cpu"])


def test_checkpoint_discovery_rejects_curve_gap(tmp_path) -> None:
    run_dir = _write_run(tmp_path)
    (run_dir / "checkpoint-25").rename(run_dir / "checkpoint-50")
    meta_path = run_dir / "checkpoint-50" / "checkpoint_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["completed_epoch"] = 50
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    cfg.trainer.total_epochs = 50

    with pytest.raises(ValueError, match="incomplete or has gaps"):
        checkpoint_eval._discover_checkpoint_targets(run_dir, cfg)


def test_checkpoint_discovery_rejects_incomplete_curve_before_model_load(tmp_path) -> None:
    run_dir = _write_run(tmp_path)
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    cfg.trainer.total_epochs = 50

    with pytest.raises(ValueError, match="incomplete or has gaps"):
        checkpoint_eval._discover_checkpoint_targets(run_dir, cfg)


def test_checkpoint_discovery_keeps_recovery_saves_out_of_eval_curve(tmp_path) -> None:
    run_dir = _write_run(tmp_path)
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    cfg.trainer.save_freq = 5

    for epoch in (5, 10, 15, 20):
        checkpoint = run_dir / f"checkpoint-{epoch}"
        checkpoint.mkdir()
        payload = f"checkpoint-{epoch}".encode()
        (checkpoint / "checkpoint.pt").write_bytes(payload)
        (checkpoint / "checkpoint_meta.json").write_text(
            json.dumps(
                {
                    "family": "sana",
                    "completed_epoch": epoch,
                    "checkpoint_file_bytes": len(payload),
                    "uses_lora": False,
                },
            ),
            encoding="utf-8",
        )

    targets = checkpoint_eval._discover_checkpoint_targets(run_dir, cfg)

    assert [target.epoch for target in targets] == [-1, 25]


def test_training_metrics_preflight_requires_every_registered_update(tmp_path) -> None:
    run_dir = _write_run(tmp_path)
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    metrics = run_dir / "metrics.csv"
    checkpoint_eval._validate_training_metrics(metrics, cfg)
    metrics.write_text("epoch,loss\n0,1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete or out of order"):
        checkpoint_eval._validate_training_metrics(metrics, cfg)


def test_fullparam_long_config_is_the_exact_registered_protocol() -> None:
    canonical = load_config(checkpoint_eval.CANONICAL_CONFIG_NAME)
    normalized = checkpoint_eval._normalize_run_config(canonical)

    assert normalized.model.use_lora is False
    assert normalized.actor.ppo_epochs == 1
    assert normalized.trainer.total_epochs == 300
    assert normalized.trainer.save_freq == 5


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("model.use_lora", True),
        ("actor.optim.lr", 1.0e-5),
        ("actor.ppo_epochs", 4),
        ("trainer.total_epochs", 301),
        ("sampling.num_steps", 11),
    ],
)
def test_fullparam_protocol_rejects_scientific_drift(path: str, value: object) -> None:
    changed = load_config(checkpoint_eval.CANONICAL_CONFIG_NAME)
    OmegaConf.update(changed, path, value, merge=False)

    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        checkpoint_eval._normalize_run_config(changed)


def test_quality_sampling_is_official_and_not_derived_from_training_sde() -> None:
    cfg = load_config(checkpoint_eval.CANONICAL_CONFIG_NAME)

    assert (cfg.sampling.width, cfg.sampling.height, cfg.sampling.num_steps) == (512, 512, 10)
    assert checkpoint_eval._resolve_sampling() == {
        "negative_prompt": "",
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 20,
        "guidance_scale": 4.5,
        "max_sequence_length": 300,
        "use_resolution_binning": True,
        "complex_human_instruction": "official_pipeline_default",
    }


def test_canonical_preset_change_requires_protocol_digest_update(monkeypatch) -> None:
    actual = load_config(checkpoint_eval.CANONICAL_CONFIG_NAME)
    real_load_config = checkpoint_eval.load_config

    def changed_canonical(name, *args, **kwargs):
        cfg = real_load_config(name, *args, **kwargs)
        if str(name) == checkpoint_eval.CANONICAL_CONFIG_NAME:
            cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            cfg.sampling.num_steps = 11
        return cfg

    monkeypatch.setattr(checkpoint_eval, "load_config", changed_canonical)
    with pytest.raises(ValueError, match="preset changed without a protocol schema update"):
        checkpoint_eval._normalize_run_config(actual)


def test_registered_manifest_assets_are_exact_and_disjoint() -> None:
    cfg = load_config(checkpoint_eval.CANONICAL_CONFIG_NAME)

    training_path, eval_path, eval_prompts = checkpoint_eval._resolve_protocol_manifests(cfg)

    assert checkpoint_eval._sha256(training_path) == checkpoint_eval.TRAIN_MANIFEST_SHA256
    assert checkpoint_eval._sha256(eval_path) == checkpoint_eval.EVAL_MANIFEST_SHA256
    assert len(eval_prompts) == checkpoint_eval.EVAL_PROMPT_COUNT


def test_manifest_replacement_and_overlap_are_rejected(monkeypatch, tmp_path) -> None:
    canonical = load_config(checkpoint_eval.CANONICAL_CONFIG_NAME)
    replaced = OmegaConf.create(OmegaConf.to_container(canonical, resolve=True))
    changed_eval = tmp_path / "changed_eval.txt"
    changed_eval.write_text("replacement prompt\n", encoding="utf-8")
    replaced.data.eval_manifest = str(changed_eval)
    with pytest.raises(ValueError, match="does not match the registered asset"):
        checkpoint_eval._resolve_protocol_manifests(replaced)

    training = tmp_path / "training.txt"
    evaluation = tmp_path / "evaluation.txt"
    training.write_text("shared prompt\ntraining only\n", encoding="utf-8")
    evaluation.write_text("shared prompt\neval only\n", encoding="utf-8")
    overlap_cfg = OmegaConf.create(
        {"data": {"manifest": str(training), "eval_manifest": str(evaluation)}},
    )
    monkeypatch.setattr(
        checkpoint_eval, "TRAIN_MANIFEST_SHA256", checkpoint_eval._sha256(training)
    )
    monkeypatch.setattr(
        checkpoint_eval, "EVAL_MANIFEST_SHA256", checkpoint_eval._sha256(evaluation)
    )
    monkeypatch.setattr(checkpoint_eval, "TRAIN_PROMPT_COUNT", 2)
    monkeypatch.setattr(checkpoint_eval, "EVAL_PROMPT_COUNT", 2)
    with pytest.raises(ValueError, match="overlap on 1 prompts"):
        checkpoint_eval._resolve_protocol_manifests(overlap_cfg)


def test_reward_model_definitions_resolve_device_and_require_explicit_identity(tmp_path) -> None:
    run_dir = _write_run(tmp_path)
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    cfg.reward.kwargs.aesthetic.device = None
    cfg.reward.kwargs.pickscore.device = None

    reward_models = checkpoint_eval._build_reward_model_definitions(
        cfg,
        generation_device="cuda:3",
    )

    assert [model.model_config["device"] for model in reward_models] == ["cuda:3", "cuda:3"]
    assert reward_models[0].model_config["model_name"] == "test/aesthetic"
    assert reward_models[1].model_config["processor_name"] == "test/processor"

    cfg.reward.kwargs.pickscore.model_name = None
    with pytest.raises(ValueError, match="explicit pickscore reward identity"):
        checkpoint_eval._build_reward_model_definitions(cfg, generation_device="cuda:3")


def test_reward_provenance_includes_pinned_revisions_and_asset_hash() -> None:
    cfg = load_config(checkpoint_eval.CANONICAL_CONFIG_NAME)

    reward_models = checkpoint_eval._build_reward_model_definitions(
        cfg,
        generation_device="cuda:0",
    )
    records = [checkpoint_eval._reward_model_record(model) for model in reward_models]

    assert (
        records[0]["identity"]["model"]["revision"] == cfg.reward.kwargs.aesthetic.model_revision
    )
    assert records[0]["identity"]["mlp_asset"] == {
        "package": "vrl.rewards.assets",
        "name": "sac+logos+ava1-l14-linearMSE.pth",
        "sha256": checkpoint_eval.AESTHETIC_ASSET_SHA256,
        "bytes": checkpoint_eval.AESTHETIC_ASSET_BYTES,
    }
    assert (
        records[1]["identity"]["processor"]["revision"]
        == cfg.reward.kwargs.pickscore.processor_revision
    )
    assert (
        records[1]["identity"]["model"]["revision"] == cfg.reward.kwargs.pickscore.model_revision
    )


def test_snapshot_materialization_uses_all_four_pinned_revisions(monkeypatch) -> None:
    import huggingface_hub

    calls: list[tuple[str, str]] = []

    def fake_snapshot_download(*, repo_id, revision):
        calls.append((repo_id, revision))
        return f"/immutable/{revision}"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    cfg = load_config(checkpoint_eval.CANONICAL_CONFIG_NAME)
    reward_models = checkpoint_eval._build_reward_model_definitions(
        cfg,
        generation_device="cuda:0",
    )

    build_cfg = checkpoint_eval._materialize_model_snapshot(cfg)
    materialized_reward_models = checkpoint_eval._materialize_reward_model_snapshots(
        reward_models,
    )

    assert str(cfg.model.path) == "Efficient-Large-Model/Sana_1600M_1024px_diffusers"
    assert str(build_cfg.model.path) == f"/immutable/{cfg.model.revision}"
    assert set(calls) == {
        (str(cfg.model.path), str(cfg.model.revision)),
        (
            "openai/clip-vit-large-patch14",
            str(cfg.reward.kwargs.aesthetic.model_revision),
        ),
        (
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            str(cfg.reward.kwargs.pickscore.processor_revision),
        ),
        ("yuvalkirstain/PickScore_v1", str(cfg.reward.kwargs.pickscore.model_revision)),
    }
    assert materialized_reward_models[0].model_config["model_name"].startswith("/immutable/")
    assert (
        materialized_reward_models[1]
        .model_config["processor_name"]
        .startswith(
            "/immutable/",
        )
    )
    assert materialized_reward_models[1].model_config["model_name"].startswith("/immutable/")
    assert "/immutable/" not in json.dumps(
        [checkpoint_eval._reward_model_record(model) for model in materialized_reward_models],
    )


def test_training_log_binds_configured_revisions_without_network_log_scraping(tmp_path) -> None:
    cfg = load_config(checkpoint_eval.CANONICAL_CONFIG_NAME)
    with pytest.raises(FileNotFoundError, match=r"no supervisor\.log launch evidence"):
        checkpoint_eval._validate_training_log_provenance(tmp_path, cfg)

    reward_kwargs = OmegaConf.to_container(cfg.reward.kwargs, resolve=True)
    expected = {
        str(cfg.model.path): str(cfg.model.revision),
        reward_kwargs["aesthetic"]["model_name"]: reward_kwargs["aesthetic"]["model_revision"],
        reward_kwargs["pickscore"]["processor_name"]: reward_kwargs["pickscore"][
            "processor_revision"
        ],
        reward_kwargs["pickscore"]["model_name"]: reward_kwargs["pickscore"]["model_revision"],
    }
    log = tmp_path / "supervisor.log"
    log.write_text("all artifacts were cache hits\n", encoding="utf-8")

    record = checkpoint_eval._validate_training_log_provenance(tmp_path, cfg)

    assert record["configured_model_revisions"] == expected
    assert record["sha256"] == checkpoint_eval._sha256(log)

    cfg.reward.kwargs.pickscore.model_revision = None
    with pytest.raises(ValueError, match="requires pinned"):
        checkpoint_eval._validate_training_log_provenance(tmp_path, cfg)


def test_official_generation_keeps_two_images_in_one_fixed_seed_stream() -> None:
    class DPMSolverMultistepScheduler:
        def __init__(self) -> None:
            self.config = {
                key: value
                for key, value in sana_inference.SCHEDULER_PROTOCOL.items()
                if key != "class_name"
            }

    class FakePipeline:
        scheduler = None

        def __call__(self, **kwargs):
            assert torch.is_inference_mode_enabled()
            assert kwargs["generator"].initial_seed() == checkpoint_eval.EVAL_BASE_SEED
            assert kwargs["num_images_per_prompt"] == checkpoint_eval.EVAL_SAMPLES_PER_PROMPT
            assert kwargs["height"] == kwargs["width"] == 1024
            assert kwargs["num_inference_steps"] == 20
            return SimpleNamespace(
                images=[Image.new("RGB", (8, 8), color=index) for index in range(2)],
            )

    model = SimpleNamespace(pipeline=FakePipeline())
    decoded = sana_inference.generate_prompt_images(
        model,
        forward_precision=SANA_FORWARD_PRECISION,
        scheduler=DPMSolverMultistepScheduler(),
        prompt="fox",
        seed=checkpoint_eval._group_seed(0),
        num_images=checkpoint_eval.EVAL_SAMPLES_PER_PROMPT,
        device=torch.device("cpu"),
        sampling=checkpoint_eval._resolve_sampling(),
    )

    assert len(decoded) == checkpoint_eval.EVAL_SAMPLES_PER_PROMPT
    assert checkpoint_eval._group_seed(1) - checkpoint_eval._group_seed(0) == 2


def test_official_generation_rejects_sampling_drift() -> None:
    class DPMSolverMultistepScheduler:
        def __init__(self) -> None:
            self.config = {
                key: value
                for key, value in sana_inference.SCHEDULER_PROTOCOL.items()
                if key != "class_name"
            }

    changed = checkpoint_eval._resolve_sampling()
    changed["height"] = 512

    with pytest.raises(ValueError, match="changed from the official protocol"):
        sana_inference.generate_prompt_images(
            SimpleNamespace(pipeline=SimpleNamespace()),
            forward_precision=SANA_FORWARD_PRECISION,
            scheduler=DPMSolverMultistepScheduler(),
            prompt="fox",
            seed=0,
            num_images=2,
            device=torch.device("cpu"),
            sampling=changed,
        )


def test_generation_uses_fresh_base_before_reading_fullparam_checkpoints(
    monkeypatch,
    tmp_path,
) -> None:
    import vrl.families.registry as model_families
    import vrl.utils.media as media

    events: list[str] = []

    class FakeModel:
        state = "base"

        def eval(self):
            return self

    model = FakeModel()
    bundle = SimpleNamespace(model=model, forward_precision=SANA_FORWARD_PRECISION)
    entry = SimpleNamespace(
        resolve_model_build=lambda *args, **kwargs: object(),
        build_rollout=lambda build: bundle,
    )
    monkeypatch.setattr(
        model_families,
        "get_model_family_entry",
        lambda _family: entry,
    )
    monkeypatch.setattr(
        checkpoint_eval,
        "load_training_checkpoint",
        lambda path: (
            events.append(f"read:{Path(path).name.split('-')[-1]}")
            or SimpleNamespace(
                payload={"family": "sana"},
                meta={"uses_lora": False},
                next_epoch=int(Path(path).name.split("-")[-1]),
                trainable_state={"epoch": int(Path(path).name.split("-")[-1])},
            )
        ),
    )

    def fake_load(bundle_arg, state, *, strict):
        assert bundle_arg is bundle
        assert strict is True
        model.state = f"checkpoint-{state['epoch']}"
        events.append(f"load:{state['epoch']}")

    def fake_generate(model_arg, **kwargs):
        del kwargs
        assert model_arg is model
        events.append(f"generate:{model.state}")
        return [object(), object()]

    def fake_write_png(image, path):
        del image
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")

    monkeypatch.setattr(checkpoint_eval, "load_trainable_state", fake_load)
    monkeypatch.setattr(checkpoint_eval, "generate_prompt_images", fake_generate)
    monkeypatch.setattr(checkpoint_eval, "load_official_scheduler", lambda build: object())
    monkeypatch.setattr(
        checkpoint_eval,
        "validate_model_precision",
        lambda value, forward_precision: {},
    )
    monkeypatch.setattr(media, "write_png", fake_write_png)
    first_checkpoint = tmp_path / "checkpoint-25"
    targets = [
        checkpoint_eval.CheckpointTarget("baseline", -1, None, None, None),
        checkpoint_eval.CheckpointTarget(
            "checkpoint-25",
            25,
            first_checkpoint,
            "hash-25",
            1,
        ),
        checkpoint_eval.CheckpointTarget(
            "checkpoint-50",
            50,
            tmp_path / "checkpoint-50",
            "hash-50",
            1,
        ),
    ]

    generated = checkpoint_eval._generate_images(
        OmegaConf.create({"model": {"family": "sana"}}),
        targets,
        ["fox"],
        output_dir=tmp_path / "eval",
        sampling={},
        device=torch.device("cpu"),
    )

    assert len(generated) == 6
    assert events == [
        "generate:base",
        "read:25",
        "load:25",
        "generate:checkpoint-25",
        "read:50",
        "load:50",
        "generate:checkpoint-50",
    ]
