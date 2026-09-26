from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from tests.scripts.eval.fixtures import (
    TinySanaPipeline,
    build_official_sana_scheduler,
    write_tiny_sana_snapshot,
)
from tests.trainers._checkpoint_helpers import _Trainer
from vrl.config.loading import load_config
from vrl.config.precision import RolePrecision
from vrl.config.schema import parse_config
from vrl.models import checkpoint_identity
from vrl.models.interfaces.runtime import ModelBuild
from vrl.scripts.eval import sana_aesthetic_checkpoint_eval as checkpoint_eval
from vrl.scripts.eval import sana_aesthetic_report as sana_report
from vrl.scripts.eval import sana_inference
from vrl.trainers.checkpointing import save_training_checkpoint
from vrl.utils.artifacts import sha256_file

SANA_PRECISION = RolePrecision(
    dtype="fp16",
    float32_precision="ieee",
    outer_autocast=False,
)
SANA_IDENTITY = {"schema": "vrl.model-identity/v1", "sources": {}, "build": {}}

# The five `main()` tests below run on a minimal synthetic run directory and stub
# out snapshot materialization plus image generation. Which pinned revisions get
# materialized is covered for real by
# `test_snapshot_materialization_uses_all_four_pinned_revisions`; what has no
# counterpart anywhere is the download and the generation itself.
_HUB_SNAPSHOTS_AND_REWARD_WEIGHTS_NEED_THE_NETWORK = pytest.mark.real_cover(
    None,
    why=(
        "the model and the two reward repos are pinned Hub snapshots (snapshot_download needs "
        "the network and multi-GB weights) and scoring needs the real aesthetic/PickScore "
        "weights, so materialization and _score_images stay doubles; generation itself runs "
        "for real on the tiny local snapshot (see test_generation_uses_fresh_base_...)"
    ),
    tracked_in="docs/sprints/done/SPRINT_zero-cost-real-object-swaps.md",
)


def _write_run(
    tmp_path: Path,
    monkeypatch,
    *,
    empty_manifest: bool = False,
) -> tuple[Path, TinySanaPipeline]:
    """A 25-epoch run on the tiny snapshot: resolved config, metrics, and a real
    `checkpoint-25` saved from the real bundle (weights filled to 1.0 and labelled
    so the pipeline reports which weights painted each image)."""

    snapshot = write_tiny_sana_snapshot(tmp_path / "sana-snapshot")
    pipeline = TinySanaPipeline()
    pipeline.install(monkeypatch, snapshot)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = tmp_path / "eval.txt"
    manifest.write_text("" if empty_manifest else "a red fox\n", encoding="utf-8")
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "sana",
                "path": str(snapshot),
                "revision": None,
                "use_lora": False,
                "lora": None,
            },
            "data": {
                "manifest": str(manifest),
                "eval_manifest": str(manifest),
                "preprocessing": {},
                "sampler": {"type": "sequential_window"},
            },
            "precision": {
                "float32_precision": "ieee",
                "training": {"dtype": "fp16"},
            },
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

    import vrl.models.families.registry as model_families

    root, precision = _sana_root(snapshot)
    entry = model_families.get_model_family_entry("sana")
    build = entry.resolve_model_build(
        root, torch.device("cpu"), precision=precision, for_rollout=True
    )
    bundle = entry.build_rollout(build)
    base_state = {name: value.clone() for name, value in pipeline.transformer.state_dict().items()}
    with torch.no_grad():
        for parameter in pipeline.transformer.parameters():
            parameter.fill_(1.0)
    pipeline.label_weights("checkpoint-25")
    save_training_checkpoint(
        run_dir / "checkpoint-25",
        trainer=_Trainer(),
        bundle=bundle,
        family="sana",
        progress={"completed_epoch": 25, "next_epoch": 25},
        rng_state={},
        model_identity=checkpoint_identity.resolve_checkpoint_model_identity(build),
    )
    pipeline.transformer.load_state_dict(base_state)
    pipeline.loads = 0
    return run_dir, pipeline


def _allow_minimal_protocol(monkeypatch) -> None:
    """Replace remote materialization and log/manifest fixtures for tiny runs."""

    monkeypatch.setattr(
        checkpoint_eval,
        "_materialize_model_snapshot",
        checkpoint_eval.parse_config,
    )
    monkeypatch.setattr(
        checkpoint_eval,
        "_materialize_reward_model_snapshots",
        lambda reward_models: reward_models,
    )
    monkeypatch.setattr(
        sana_report,
        "require_training_log_provenance",
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

    monkeypatch.setattr(sana_report, "resolve_protocol_manifests", resolve_manifests)


@_HUB_SNAPSHOTS_AND_REWARD_WEIGHTS_NEED_THE_NETWORK
@pytest.mark.parametrize(("seed", "samples"), [(0, 2), (17, 3)])
def test_main_writes_provenance_bound_report(monkeypatch, tmp_path, capsys, seed, samples) -> None:
    run_dir, pipeline = _write_run(tmp_path, monkeypatch)
    _allow_minimal_protocol(monkeypatch)

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

    monkeypatch.setattr(checkpoint_eval, "_score_images", fake_score)
    checkpoint_eval.main(
        [
            "--run-dir",
            str(run_dir),
            "--device",
            "cpu",
            "--seed",
            str(seed),
            "--samples-per-prompt",
            str(samples),
            "--checkpoint-interval",
            "25",
        ]
    )

    # Real generation: the base grid was painted before checkpoint-25 was
    # restored into the same bundle, one pipeline load for the whole run.
    assert [call["weights"] for call in pipeline.calls] == ["base", "checkpoint-25"]
    assert pipeline.loads == 1
    rows = sana_report.load_report_metrics(run_dir)
    assert [row["epoch"] for row in rows] == [-1.0, 25.0]
    assert all(row["sample_count"] == samples for row in rows)
    payload = json.loads((run_dir / sana_report.REPORT_RELATIVE_PATH).read_text())
    assert payload["schema"] == sana_report.REPORT_SCHEMA
    assert "schema_version" not in payload
    assert payload["provenance"]["seed_grid"]["base_seed"] == seed
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
        sana_report.load_report_metrics(run_dir)
    checkpoint_path.write_bytes(checkpoint_bytes)

    metrics_path = run_dir / "metrics.csv"
    metrics_text = metrics_path.read_text(encoding="utf-8")
    metrics_path.write_text(metrics_text.replace("1.0", "2.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="training metrics provenance hash changed"):
        sana_report.load_report_metrics(run_dir)
    metrics_path.write_text(metrics_text, encoding="utf-8")

    payload["provenance"]["seed_grid"]["base_seed"] += 1
    report_path = run_dir / sana_report.REPORT_RELATIVE_PATH
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="wrong fixed-grid seed"):
        sana_report.load_report_metrics(run_dir)
    payload["provenance"]["seed_grid"]["base_seed"] -= 1

    payload["metrics"] = []
    (run_dir / sana_report.REPORT_RELATIVE_PATH).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="report has no metric rows"):
        sana_report.load_report_metrics(run_dir)


@_HUB_SNAPSHOTS_AND_REWARD_WEIGHTS_NEED_THE_NETWORK
def test_report_reader_rejects_empty_metrics(monkeypatch, tmp_path) -> None:
    run_dir, _pipeline = _write_run(tmp_path, monkeypatch)
    _allow_minimal_protocol(monkeypatch)
    monkeypatch.setattr(checkpoint_eval, "_score_images", lambda *args, **kwargs: [])

    with pytest.raises(ValueError, match=r"empty.*sample manifest"):
        checkpoint_eval.main(["--run-dir", str(run_dir), "--device", "cpu"])


@_HUB_SNAPSHOTS_AND_REWARD_WEIGHTS_NEED_THE_NETWORK
def test_main_rejects_checkpoint_identity_before_model_snapshot(monkeypatch, tmp_path) -> None:
    run_dir, pipeline = _write_run(tmp_path, monkeypatch)
    _allow_minimal_protocol(monkeypatch)
    meta_path = run_dir / "checkpoint-25" / "checkpoint_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["model_identity"] = {"schema": "wrong/v1"}
    meta_path.write_text(json.dumps(meta))
    materialized = False

    def fail_if_materialized(_cfg):
        nonlocal materialized
        materialized = True
        raise AssertionError("model snapshot must not be materialized")

    monkeypatch.setattr(checkpoint_eval, "_materialize_model_snapshot", fail_if_materialized)

    with pytest.raises(ValueError, match="metadata model identity mismatch"):
        checkpoint_eval.main(["--run-dir", str(run_dir), "--device", "cpu"])

    assert materialized is False
    assert pipeline.loads == 0


@_HUB_SNAPSHOTS_AND_REWARD_WEIGHTS_NEED_THE_NETWORK
def test_report_reader_rejects_changed_config_provenance(monkeypatch, tmp_path) -> None:
    run_dir, _pipeline = _write_run(tmp_path, monkeypatch)
    _allow_minimal_protocol(monkeypatch)

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
        sana_report.load_report_metrics(run_dir)


@_HUB_SNAPSHOTS_AND_REWARD_WEIGHTS_NEED_THE_NETWORK
def test_main_rejects_empty_manifest(monkeypatch, tmp_path) -> None:
    run_dir, _pipeline = _write_run(tmp_path, monkeypatch, empty_manifest=True)
    _allow_minimal_protocol(monkeypatch)
    with pytest.raises(ValueError, match="manifest has no prompts"):
        checkpoint_eval.main(["--run-dir", str(run_dir), "--device", "cpu"])


def test_checkpoint_discovery_rejects_curve_gap(tmp_path, monkeypatch) -> None:
    run_dir, _pipeline = _write_run(tmp_path, monkeypatch)
    (run_dir / "checkpoint-25").rename(run_dir / "checkpoint-50")
    meta_path = run_dir / "checkpoint-50" / "checkpoint_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["completed_epoch"] = 50
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    cfg.trainer.total_epochs = 50

    with pytest.raises(ValueError, match="incomplete or has gaps"):
        checkpoint_eval._discover_checkpoint_targets(run_dir, parse_config(cfg))


def test_checkpoint_discovery_rejects_incomplete_curve_before_model_load(
    tmp_path, monkeypatch
) -> None:
    run_dir, _pipeline = _write_run(tmp_path, monkeypatch)
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    cfg.trainer.total_epochs = 50

    with pytest.raises(ValueError, match="incomplete or has gaps"):
        checkpoint_eval._discover_checkpoint_targets(run_dir, parse_config(cfg))


def test_checkpoint_discovery_keeps_recovery_saves_out_of_eval_curve(
    tmp_path, monkeypatch
) -> None:
    run_dir, _pipeline = _write_run(tmp_path, monkeypatch)
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

    targets = checkpoint_eval._discover_checkpoint_targets(run_dir, parse_config(cfg))

    assert [target.epoch for target in targets] == [-1, 25]


def test_training_metrics_preflight_requires_every_registered_update(
    tmp_path, monkeypatch
) -> None:
    run_dir, _pipeline = _write_run(tmp_path, monkeypatch)
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    metrics = run_dir / "metrics.csv"
    sana_report.validate_training_metrics(metrics, parse_config(cfg))
    metrics.write_text("epoch,loss\n0,1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete or out of order"):
        sana_report.validate_training_metrics(metrics, parse_config(cfg))


def test_custom_manifest_is_accepted_but_overlap_is_rejected(tmp_path) -> None:
    canonical = load_config("experiment/sana/online_grpo_aesthetic_fullparam_long")
    replaced = OmegaConf.create(OmegaConf.to_container(canonical, resolve=True))
    changed_eval = tmp_path / "changed_eval.txt"
    changed_eval.write_text("replacement prompt\n", encoding="utf-8")
    replaced.data.eval_manifest = str(changed_eval)
    _, path, prompts = sana_report.resolve_protocol_manifests(parse_config(replaced))
    assert path == changed_eval
    assert prompts == ["replacement prompt"]

    training = tmp_path / "training.txt"
    evaluation = tmp_path / "evaluation.txt"
    training.write_text("shared prompt\ntraining only\n", encoding="utf-8")
    evaluation.write_text("shared prompt\neval only\n", encoding="utf-8")
    overlap_cfg = OmegaConf.create(
        {
            "data": {
                "manifest": str(training),
                "eval_manifest": str(evaluation),
                "preprocessing": {},
                "sampler": {"type": "sequential_window"},
            },
        },
    )
    with pytest.raises(ValueError, match="overlap on 1 prompts"):
        sana_report.resolve_protocol_manifests(parse_config(overlap_cfg))


def test_reward_model_definitions_resolve_device_and_require_explicit_identity(
    tmp_path, monkeypatch
) -> None:
    run_dir, _pipeline = _write_run(tmp_path, monkeypatch)
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    cfg.reward.kwargs.aesthetic.device = None
    cfg.reward.kwargs.pickscore.device = None

    reward_models = sana_report.build_reward_model_definitions(
        parse_config(cfg),
        generation_device="cuda:3",
    )

    assert [model.model_config["device"] for model in reward_models] == ["cuda:3", "cuda:3"]
    assert reward_models[0].model_config["model_name"] == "test/aesthetic"
    assert reward_models[1].model_config["processor_name"] == "test/processor"

    cfg.reward.kwargs.pickscore.model_name = None
    with pytest.raises(ValueError, match="explicit pickscore reward identity"):
        sana_report.build_reward_model_definitions(parse_config(cfg), generation_device="cuda:3")


def test_reward_provenance_includes_pinned_revisions_and_asset_hash() -> None:
    cfg = load_config("experiment/sana/online_grpo_aesthetic_fullparam_long")

    reward_models = sana_report.build_reward_model_definitions(
        parse_config(cfg),
        generation_device="cuda:0",
    )
    records = [model.to_report_record() for model in reward_models]

    assert (
        records[0]["identity"]["model"]["revision"] == cfg.reward.kwargs.aesthetic.model_revision
    )
    assert records[0]["identity"]["mlp_asset"] == {
        "package": "vrl.rewards.assets",
        "name": "aesthetic_predictor_v2_5.pth",
        "sha256": sha256_file(Path("vrl/rewards/assets/aesthetic_predictor_v2_5.pth")),
        "bytes": Path("vrl/rewards/assets/aesthetic_predictor_v2_5.pth").stat().st_size,
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
    cfg = load_config("experiment/sana/online_grpo_aesthetic_fullparam_long")
    reward_models = sana_report.build_reward_model_definitions(
        parse_config(cfg),
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
            "google/siglip-so400m-patch14-384",
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
        [model.to_report_record() for model in materialized_reward_models],
    )


def test_training_log_binds_configured_revisions_without_network_log_scraping(tmp_path) -> None:
    cfg = load_config("experiment/sana/online_grpo_aesthetic_fullparam_long")
    with pytest.raises(FileNotFoundError, match=r"no supervisor\.log launch evidence"):
        sana_report.require_training_log_provenance(tmp_path, parse_config(cfg))

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

    record = sana_report.require_training_log_provenance(tmp_path, parse_config(cfg))

    assert record["configured_model_revisions"] == expected
    assert record["sha256"] == sha256_file(log)

    cfg.reward.kwargs.pickscore.model_revision = None
    with pytest.raises(ValueError, match="requires pinned"):
        sana_report.require_training_log_provenance(tmp_path, parse_config(cfg))


def test_official_generation_keeps_two_images_in_one_fixed_seed_stream() -> None:
    class FakePipeline:
        scheduler = None

        def __call__(self, **kwargs):
            assert torch.is_inference_mode_enabled()
            assert kwargs["generator"].initial_seed() == sana_report.EvaluationSettings().base_seed
            assert (
                kwargs["num_images_per_prompt"]
                == sana_report.EvaluationSettings().samples_per_prompt
            )
            assert kwargs["height"] == kwargs["width"] == 1024
            assert kwargs["num_inference_steps"] == 20
            return SimpleNamespace(
                images=[Image.new("RGB", (8, 8), color=index) for index in range(2)],
            )

    model = SimpleNamespace(
        pipeline=FakePipeline(),
        precision=SANA_PRECISION,
    )
    decoded = sana_inference.generate_prompt_images(
        model,
        scheduler=build_official_sana_scheduler(),
        prompt="fox",
        seed=sana_report.EvaluationSettings().group_seed(0),
        num_images=sana_report.EvaluationSettings().samples_per_prompt,
        device=torch.device("cpu"),
        sampling=dict(sana_inference.SANA_EVAL_SAMPLING_CONFIG),
    )

    assert len(decoded) == sana_report.EvaluationSettings().samples_per_prompt
    assert (
        sana_report.EvaluationSettings().group_seed(1)
        - sana_report.EvaluationSettings().group_seed(0)
        == 2
    )


@pytest.mark.parametrize(
    ("revision", "expected_revision"),
    [
        pytest.param("immutable-revision", "immutable-revision", id="pinned"),
        pytest.param(None, None, id="absent"),
    ],
)
def test_official_scheduler_uses_build_revision_projection(
    monkeypatch,
    revision,
    expected_revision,
) -> None:
    import diffusers

    calls: list[tuple[object, dict]] = []

    # Recorder on the REAL class: a local directory has no revision to observe,
    # so which revision/subfolder reached the hub loader is only visible here.
    def record_from_pretrained(_cls, path, **kwargs):
        calls.append((path, kwargs))
        return build_official_sana_scheduler()

    monkeypatch.setattr(
        diffusers.DPMSolverMultistepScheduler,
        "from_pretrained",
        classmethod(record_from_pretrained),
    )
    build = ModelBuild(
        model_name_or_path="test/sana",
        revision=revision,
        device="cpu",
        parameter_dtype=torch.float16,
        family="sana",
        precision=SANA_PRECISION,
    )

    scheduler = sana_inference.load_official_scheduler(build)

    expected_kwargs = {"subfolder": "scheduler"}
    if expected_revision is not None:
        expected_kwargs["revision"] = expected_revision
    assert isinstance(scheduler, diffusers.DPMSolverMultistepScheduler)
    assert calls == [("test/sana", expected_kwargs)]


def test_official_generation_rejects_sampling_drift() -> None:
    changed = dict(sana_inference.SANA_EVAL_SAMPLING_CONFIG)
    changed["height"] = 512

    with pytest.raises(ValueError, match="changed from the official protocol"):
        sana_inference.generate_prompt_images(
            SimpleNamespace(
                pipeline=SimpleNamespace(),
                precision=SANA_PRECISION,
            ),
            scheduler=build_official_sana_scheduler(),
            prompt="fox",
            seed=0,
            num_images=2,
            device=torch.device("cpu"),
            sampling=changed,
        )


def _sana_root(snapshot: Path):
    root = checkpoint_eval.parse_config(
        OmegaConf.create(
            {
                "model": {"family": "sana", "path": str(snapshot)},
                "precision": {
                    "float32_precision": "ieee",
                    "training": {"dtype": "fp32"},
                },
            },
        ),
    )
    return root, checkpoint_eval.PrecisionPolicy.from_section(root.precision)


def test_generation_uses_fresh_base_before_reading_fullparam_checkpoints(
    monkeypatch,
    tmp_path,
) -> None:
    """Real sana entry, rollout bundle and strict restores over a tiny model.

    Two real checkpoints are saved from the same transformer with distinct
    weight fills, so every image carries a witness of which weights painted it:
    the base images must come before any checkpoint is read, and each
    checkpoint's images after its own strict restore.
    """

    import vrl.models.families.registry as model_families

    snapshot = write_tiny_sana_snapshot(tmp_path / "sana-snapshot")
    pipeline = TinySanaPipeline()
    pipeline.install(monkeypatch, snapshot)
    root, precision = _sana_root(snapshot)
    entry = model_families.get_model_family_entry("sana")
    build = entry.resolve_model_build(
        root, torch.device("cpu"), precision=precision, for_rollout=True
    )
    identity = checkpoint_identity.resolve_checkpoint_model_identity(build)
    bundle = entry.build_rollout(build)
    base_state = {name: value.clone() for name, value in pipeline.transformer.state_dict().items()}

    targets = [checkpoint_eval.CheckpointTarget("baseline", -1, None, None, None)]
    for epoch, fill in ((25, 1.0), (50, 2.0)):
        with torch.no_grad():
            for parameter in pipeline.transformer.parameters():
                parameter.fill_(fill)
        pipeline.label_weights(f"checkpoint-{epoch}")
        path = tmp_path / f"checkpoint-{epoch}"
        save_training_checkpoint(
            path,
            trainer=_Trainer(),
            bundle=bundle,
            family="sana",
            progress={"next_epoch": epoch},
            rng_state={},
            model_identity=identity,
        )
        targets.append(
            checkpoint_eval.CheckpointTarget(f"checkpoint-{epoch}", epoch, path, None, None)
        )
    pipeline.transformer.load_state_dict(base_state)
    assert pipeline.holds() == "base"

    generated = checkpoint_eval._generate_images(
        root,
        precision,
        targets,
        ["fox"],
        output_dir=tmp_path / "eval",
        sampling=dict(sana_inference.SANA_EVAL_SAMPLING_CONFIG),
        device=torch.device("cpu"),
        expected_model_identity=identity,
    )

    assert [call["weights"] for call in pipeline.calls] == [
        "base",
        "checkpoint-25",
        "checkpoint-50",
    ]
    assert [image.checkpoint_label for image in generated] == [
        "baseline",
        "baseline",
        "checkpoint-25",
        "checkpoint-25",
        "checkpoint-50",
        "checkpoint-50",
    ]
    for image in generated:
        assert image.path.is_file()
        assert image.image_sha256 == hashlib.sha256(image.path.read_bytes()).hexdigest()
    for scheduler in (call["scheduler"] for call in pipeline.calls):
        sana_inference.require_scheduler(scheduler)


def test_generate_images_rejects_materialized_source_drift_before_generation(
    monkeypatch,
    tmp_path,
) -> None:
    """A model directory that changes under the loader is caught before any image."""

    snapshot = write_tiny_sana_snapshot(tmp_path / "sana-snapshot")
    pipeline = TinySanaPipeline()

    def mutate_snapshot() -> None:
        (snapshot / "extra-weights.bin").write_bytes(b"drift")

    pipeline.install(monkeypatch, snapshot, on_load=mutate_snapshot)
    root, precision = _sana_root(snapshot)

    with pytest.raises(
        RuntimeError,
        match="materialized SANA model source changed during runtime construction",
    ):
        checkpoint_eval._generate_images(
            root,
            precision,
            [checkpoint_eval.CheckpointTarget("baseline", -1, None, None, None)],
            ["fox"],
            output_dir=tmp_path / "eval",
            sampling=dict(sana_inference.SANA_EVAL_SAMPLING_CONFIG),
            device=torch.device("cpu"),
            expected_model_identity={"schema": "unused"},
        )

    assert pipeline.loads == 1
    assert pipeline.calls == []


def test_run_config_keeps_experiment_choices_instead_of_replacing_with_a_preset() -> None:
    cfg = load_config("experiment/sana/online_grpo_aesthetic_fullparam_long")
    cfg.trainer.total_epochs = 50
    cfg.trainer.save_freq = 10
    root = sana_report.validate_run_config(cfg)
    assert root.trainer.total_epochs == 50
    assert sana_report.checkpoint_curve_epochs(
        root,
        sana_report.EvaluationSettings(checkpoint_interval=10),
    ) == [10, 20, 30, 40, 50]


def test_run_config_rejects_other_model_family() -> None:
    cfg = load_config("experiment/sana/online_grpo_aesthetic_fullparam_long")
    cfg.model.family = "flux"
    with pytest.raises(ValueError, match=r"model\.family=sana"):
        sana_report.validate_run_config(cfg)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"samples_per_prompt": 0},
        {"checkpoint_interval": -1},
        {"base_seed": -1},
        {"samples_per_prompt": True},
    ],
)
def test_evaluation_settings_reject_invalid_grid(kwargs) -> None:
    with pytest.raises(ValueError):
        sana_report.EvaluationSettings(**kwargs)
