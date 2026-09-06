from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from vrl.config.loading import load_config
from vrl.config.precision import RolePrecision
from vrl.config.schema import parse_config
from vrl.models import checkpoint_identity
from vrl.models.interfaces.runtime import ModelBuild
from vrl.scripts.eval import sana_aesthetic_checkpoint_eval as checkpoint_eval
from vrl.scripts.eval import sana_aesthetic_report as sana_report
from vrl.scripts.eval import sana_inference
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
_SNAPSHOTS_AND_GENERATION_NEED_THE_REAL_WEIGHTS = pytest.mark.real_cover(
    None,
    why=(
        "materializing the four pinned Hub snapshots needs network access and multi-GB weights, "
        "and generating the image grid from them needs a CUDA device; there is no SANA case in "
        "tests/e2e, so this repo has no lane where either runs for real"
    ),
    tracked_in="docs/sprints/done/SPRINT_zero-cost-real-object-swaps.md",
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
    checkpoint = run_dir / "checkpoint-25"
    checkpoint.mkdir()
    payload = b"checkpoint-state"
    (checkpoint / "checkpoint.pt").write_bytes(payload)
    (checkpoint / "checkpoint_meta.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "family": "sana",
                "model_identity": SANA_IDENTITY,
                "completed_epoch": 25,
                "checkpoint_file_bytes": len(payload),
                "uses_lora": False,
            },
        ),
        encoding="utf-8",
    )
    return run_dir


def _allow_minimal_protocol(monkeypatch) -> None:
    """Let a minimal synthetic run dir through, so the report machinery can be tested.

    The identity patch on ``_normalize_run_config`` is what makes the tiny config in
    ``_write_run`` acceptable — the real gate only accepts the registered
    300-epoch protocol. That means none of these tests says anything about main()
    calling the gate; the two tests driven from ``_write_protocol_run`` below own
    that, on a run dir the real gate does accept.
    """

    monkeypatch.setattr(sana_report, "normalize_run_config", lambda cfg: cfg)
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
    monkeypatch.setattr(
        checkpoint_identity,
        "resolve_checkpoint_model_identity",
        lambda _build: SANA_IDENTITY,
    )


@_SNAPSHOTS_AND_GENERATION_NEED_THE_REAL_WEIGHTS
def test_main_writes_provenance_bound_report(monkeypatch, tmp_path, capsys) -> None:
    run_dir = _write_run(tmp_path)
    _allow_minimal_protocol(monkeypatch)

    def fake_generate(
        root,
        precision,
        targets,
        prompts,
        *,
        output_dir,
        sampling,
        device,
        expected_model_identity,
    ):
        del root, precision, sampling, device
        assert expected_model_identity == SANA_IDENTITY
        generated = []
        for target in targets:
            for sample_index in range(sana_report.EVAL_SAMPLES_PER_PROMPT):
                path = output_dir / "images" / target.label / f"sample{sample_index}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{target.label}-{sample_index}".encode())
                generated.append(
                    checkpoint_eval.GeneratedImage(
                        checkpoint_label=target.label,
                        epoch=target.epoch,
                        prompt_index=0,
                        sample_index=sample_index,
                        group_seed=sana_report.group_seed(0),
                        prompt=prompts[0],
                        path=path.resolve(),
                        image_sha256=sha256_file(path),
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

    rows = sana_report.load_report_metrics(run_dir)
    assert [row["epoch"] for row in rows] == [-1.0, 25.0]
    assert all(row["sample_count"] == 2.0 for row in rows)
    payload = json.loads((run_dir / sana_report.REPORT_RELATIVE_PATH).read_text())
    assert payload["schema"] == sana_report.REPORT_SCHEMA
    assert payload["schema_version"] == sana_report.REPORT_SCHEMA_VERSION
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
        sana_report.load_report_metrics(run_dir)
    checkpoint_path.write_bytes(checkpoint_bytes)

    metrics_path = run_dir / "metrics.csv"
    metrics_text = metrics_path.read_text(encoding="utf-8")
    metrics_path.write_text(metrics_text.replace("1.0", "2.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="training metrics provenance hash changed"):
        sana_report.load_report_metrics(run_dir)
    metrics_path.write_text(metrics_text, encoding="utf-8")

    payload["metrics"] = []
    (run_dir / sana_report.REPORT_RELATIVE_PATH).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="report has no metric rows"):
        sana_report.load_report_metrics(run_dir)


@_SNAPSHOTS_AND_GENERATION_NEED_THE_REAL_WEIGHTS
def test_report_reader_rejects_empty_metrics(monkeypatch, tmp_path) -> None:
    run_dir = _write_run(tmp_path)
    _allow_minimal_protocol(monkeypatch)
    monkeypatch.setattr(checkpoint_eval, "_generate_images", lambda *args, **kwargs: [])
    monkeypatch.setattr(checkpoint_eval, "_score_images", lambda *args, **kwargs: [])

    with pytest.raises(ValueError, match=r"empty.*sample manifest"):
        checkpoint_eval.main(["--run-dir", str(run_dir), "--device", "cpu"])


@_SNAPSHOTS_AND_GENERATION_NEED_THE_REAL_WEIGHTS
def test_main_rejects_checkpoint_identity_before_model_snapshot(monkeypatch, tmp_path) -> None:
    run_dir = _write_run(tmp_path)
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


@_SNAPSHOTS_AND_GENERATION_NEED_THE_REAL_WEIGHTS
def test_report_reader_rejects_changed_config_provenance(monkeypatch, tmp_path) -> None:
    run_dir = _write_run(tmp_path)
    _allow_minimal_protocol(monkeypatch)

    def fake_generate(
        root,
        precision,
        targets,
        prompts,
        *,
        output_dir,
        sampling,
        device,
        expected_model_identity,
    ):
        del root, precision, sampling, device
        assert expected_model_identity == SANA_IDENTITY
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
                        sana_report.group_seed(0),
                        prompts[0],
                        path,
                        sha256_file(path),
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
        sana_report.load_report_metrics(run_dir)


@_SNAPSHOTS_AND_GENERATION_NEED_THE_REAL_WEIGHTS
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
        checkpoint_eval._discover_checkpoint_targets(run_dir, parse_config(cfg))


def test_checkpoint_discovery_rejects_incomplete_curve_before_model_load(tmp_path) -> None:
    run_dir = _write_run(tmp_path)
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    cfg.trainer.total_epochs = 50

    with pytest.raises(ValueError, match="incomplete or has gaps"):
        checkpoint_eval._discover_checkpoint_targets(run_dir, parse_config(cfg))


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

    targets = checkpoint_eval._discover_checkpoint_targets(run_dir, parse_config(cfg))

    assert [target.epoch for target in targets] == [-1, 25]


def test_training_metrics_preflight_requires_every_registered_update(tmp_path) -> None:
    run_dir = _write_run(tmp_path)
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    metrics = run_dir / "metrics.csv"
    sana_report.validate_training_metrics(metrics, parse_config(cfg))
    metrics.write_text("epoch,loss\n0,1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete or out of order"):
        sana_report.validate_training_metrics(metrics, parse_config(cfg))


def _historical_fullparam_config() -> DictConfig:
    raw = OmegaConf.to_container(
        load_config(sana_report.CANONICAL_CONFIG_NAME),
        resolve=True,
    )
    assert isinstance(raw, dict)

    raw["algorithm"]["kl_reward_coef"] = 0.0
    raw["data"]["preprocessing"]["target_text"] = "none"
    raw["distributed"]["rollout"].update(
        {
            "chunk_placement_strategy": "round_robin",
            "health_check_first_wait_s": 0.0,
            "health_check_interval_s": 30.0,
            "health_check_timeout_s": 30.0,
            "max_inflight_chunks_per_worker": 1,
            "pipelined": False,
            "sync_trainable_state": True,
        },
    )
    raw["rollout"]["trajectory_storage"] = {
        "device": "preserve",
        "dtype": "preserve",
    }
    raw["reward"]["kwargs"]["aesthetic"].pop("device")
    raw["precision"].pop("float32_precision")
    raw["precision"]["training"].pop("outer_autocast")
    raw["precision"]["rollout"].pop("outer_autocast")
    raw["trainer"]["entrypoint"] = "vrl.scripts.diffusion.train:train_diffusion_grpo"
    return OmegaConf.create(raw)


def test_historical_fullparam_shape_normalizes_to_the_live_protocol() -> None:
    normalized = sana_report.normalize_run_config(
        _historical_fullparam_config(),
    )
    canonical = load_config(sana_report.CANONICAL_CONFIG_NAME)

    assert OmegaConf.to_container(normalized, resolve=True) == OmegaConf.to_container(
        canonical,
        resolve=True,
    )


def test_historical_parity_threshold_normalizes_without_changing_frozen_identity() -> None:
    canonical = load_config(sana_report.CANONICAL_CONFIG_NAME)
    historical = OmegaConf.to_container(canonical, resolve=True)
    historical["trainer"]["debug"].update(historical["trainer"].pop("replay_parity"))

    assert sana_report._semantic_digest(historical) == sana_report.CANONICAL_PROTOCOL_SHA256
    normalized = sana_report.normalize_run_config(OmegaConf.create(historical))
    assert OmegaConf.to_container(normalized, resolve=True) == OmegaConf.to_container(
        canonical, resolve=True
    )


def test_parity_threshold_rejects_ambiguous_old_and_live_keys() -> None:
    changed = load_config(sana_report.CANONICAL_CONFIG_NAME)
    changed.trainer.debug.max_abs_logprob_diff = 1.0e-4

    with pytest.raises(ValueError, match="ambiguous SANA parity threshold"):
        sana_report.normalize_run_config(changed)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("algorithm.kl_reward_coef", 0.1),
        ("rollout.trajectory_storage.unexpected", True),
        ("precision.training.dtype", "bf16"),
    ],
)
def test_historical_shape_normalization_rejects_behavioral_drift(
    path: str,
    value: object,
) -> None:
    changed = _historical_fullparam_config()
    OmegaConf.update(changed, path, value, merge=False)

    with pytest.raises(
        ValueError,
        match="does not match the registered SANA full-parameter protocol",
    ):
        sana_report.normalize_run_config(changed)


def _write_protocol_run(tmp_path: Path, *, drift: tuple[str, object] | None = None) -> Path:
    """A run directory the real protocol gate accepts (or, with ``drift``, must reject).

    ``_write_run`` writes a tiny synthetic config that the gate rejects on sight,
    which is why its callers patch the gate out. This one writes a real historical
    full-parameter shape, plus the 300-row metrics CSV and the supervisor log that
    the two checks immediately after the gate demand — so ``main()`` can be driven
    through the gate for real.
    """

    run_dir = tmp_path / "protocol-run"
    run_dir.mkdir()
    cfg = _historical_fullparam_config()
    if drift is not None:
        OmegaConf.update(cfg, drift[0], drift[1], merge=False)
    OmegaConf.save(cfg, run_dir / "resolved_config.yaml")
    (run_dir / "metrics.csv").write_text(
        "epoch,loss\n"
        + "".join(f"{epoch},1.0\n" for epoch in range(int(cfg.trainer.total_epochs))),
        encoding="utf-8",
    )
    (run_dir / "supervisor.log").write_text("launched\n", encoding="utf-8")
    return run_dir


def test_main_runs_the_protocol_gate_before_touching_the_run(monkeypatch, tmp_path) -> None:
    """``main()`` must call the gate, not merely be able to.

    Every other ``main()`` test replaces ``_normalize_run_config`` with the identity
    function, so the wiring between the two was uncovered: deleting the call from
    ``main()`` left the whole file green. This run's config is the registered
    protocol with one behavioural field changed, a rejection only the real gate can
    produce — and nothing may be generated after it.
    """

    run_dir = _write_protocol_run(tmp_path, drift=("algorithm.kl_reward_coef", 0.1))
    monkeypatch.setattr(
        checkpoint_eval,
        "_generate_images",
        lambda *args, **kwargs: pytest.fail("generation started despite a rejected config"),
    )

    with pytest.raises(
        ValueError,
        match="does not match the registered SANA full-parameter protocol",
    ):
        checkpoint_eval.main(["--run-dir", str(run_dir), "--device", "cpu"])


def test_live_entrypoint_requires_explicit_precision_protocol() -> None:
    changed = load_config(sana_report.CANONICAL_CONFIG_NAME)
    del changed.precision["float32_precision"]

    with pytest.raises(
        ValueError,
        match="does not match the registered SANA full-parameter protocol",
    ):
        sana_report.normalize_run_config(changed)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("model.use_lora", True),
        ("trainer.replay_parity.max_abs_logprob_diff", 1.0e-4),
    ],
)
def test_fullparam_protocol_rejects_scientific_drift(path: str, value: object) -> None:
    changed = load_config(sana_report.CANONICAL_CONFIG_NAME)
    OmegaConf.update(changed, path, value, merge=False)

    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        sana_report.normalize_run_config(changed)


def test_canonical_preset_change_requires_protocol_digest_update(monkeypatch) -> None:
    path, value = "sampling.num_steps", 11
    actual = load_config(sana_report.CANONICAL_CONFIG_NAME)
    real_load_config = sana_report.load_config

    def changed_canonical(name, *args, **kwargs):
        cfg = real_load_config(name, *args, **kwargs)
        if str(name) == sana_report.CANONICAL_CONFIG_NAME:
            cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            OmegaConf.update(cfg, path, value, merge=False)
        return cfg

    monkeypatch.setattr(sana_report, "load_config", changed_canonical)
    with pytest.raises(ValueError, match="preset changed without a protocol schema update"):
        sana_report.normalize_run_config(actual)


def test_registered_manifest_assets_are_exact_and_disjoint() -> None:
    cfg = load_config(sana_report.CANONICAL_CONFIG_NAME)

    training_path, eval_path, eval_prompts = sana_report.resolve_protocol_manifests(
        parse_config(cfg)
    )

    assert sha256_file(training_path) == sana_report.TRAIN_MANIFEST_SHA256
    assert sha256_file(eval_path) == sana_report.EVAL_MANIFEST_SHA256
    assert len(eval_prompts) == sana_report.EVAL_PROMPT_COUNT


def test_manifest_replacement_and_overlap_are_rejected(monkeypatch, tmp_path) -> None:
    canonical = load_config(sana_report.CANONICAL_CONFIG_NAME)
    replaced = OmegaConf.create(OmegaConf.to_container(canonical, resolve=True))
    changed_eval = tmp_path / "changed_eval.txt"
    changed_eval.write_text("replacement prompt\n", encoding="utf-8")
    replaced.data.eval_manifest = str(changed_eval)
    with pytest.raises(ValueError, match="does not match the registered asset"):
        sana_report.resolve_protocol_manifests(parse_config(replaced))

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
    monkeypatch.setattr(sana_report, "TRAIN_MANIFEST_SHA256", sha256_file(training))
    monkeypatch.setattr(sana_report, "EVAL_MANIFEST_SHA256", sha256_file(evaluation))
    monkeypatch.setattr(sana_report, "TRAIN_PROMPT_COUNT", 2)
    monkeypatch.setattr(sana_report, "EVAL_PROMPT_COUNT", 2)
    with pytest.raises(ValueError, match="overlap on 1 prompts"):
        sana_report.resolve_protocol_manifests(parse_config(overlap_cfg))


def test_reward_model_definitions_resolve_device_and_require_explicit_identity(tmp_path) -> None:
    run_dir = _write_run(tmp_path)
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
    cfg = load_config(sana_report.CANONICAL_CONFIG_NAME)

    reward_models = sana_report.build_reward_model_definitions(
        parse_config(cfg),
        generation_device="cuda:0",
    )
    records = [sana_report.reward_model_record(model) for model in reward_models]

    assert (
        records[0]["identity"]["model"]["revision"] == cfg.reward.kwargs.aesthetic.model_revision
    )
    assert records[0]["identity"]["mlp_asset"] == {
        "package": "vrl.rewards.assets",
        "name": "sac+logos+ava1-l14-linearMSE.pth",
        "sha256": sana_report.AESTHETIC_ASSET_SHA256,
        "bytes": sana_report.AESTHETIC_ASSET_BYTES,
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
    cfg = load_config(sana_report.CANONICAL_CONFIG_NAME)
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
        [sana_report.reward_model_record(model) for model in materialized_reward_models],
    )


def test_training_log_binds_configured_revisions_without_network_log_scraping(tmp_path) -> None:
    cfg = load_config(sana_report.CANONICAL_CONFIG_NAME)
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
            assert kwargs["generator"].initial_seed() == sana_report.EVAL_BASE_SEED
            assert kwargs["num_images_per_prompt"] == sana_report.EVAL_SAMPLES_PER_PROMPT
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
        scheduler=DPMSolverMultistepScheduler(),
        prompt="fox",
        seed=sana_report.group_seed(0),
        num_images=sana_report.EVAL_SAMPLES_PER_PROMPT,
        device=torch.device("cpu"),
        sampling=sana_report.resolve_sampling(),
    )

    assert len(decoded) == sana_report.EVAL_SAMPLES_PER_PROMPT
    assert sana_report.group_seed(1) - sana_report.group_seed(0) == 2


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
    calls: list[tuple[object, dict]] = []

    class DPMSolverMultistepScheduler:
        def __init__(self) -> None:
            self.config = {
                key: value
                for key, value in sana_inference.SCHEDULER_PROTOCOL.items()
                if key != "class_name"
            }

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((path, kwargs))
            return cls()

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(DPMSolverMultistepScheduler=DPMSolverMultistepScheduler),
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
    assert isinstance(scheduler, DPMSolverMultistepScheduler)
    assert calls == [("test/sana", expected_kwargs)]


def test_official_generation_rejects_sampling_drift() -> None:
    class DPMSolverMultistepScheduler:
        def __init__(self) -> None:
            self.config = {
                key: value
                for key, value in sana_inference.SCHEDULER_PROTOCOL.items()
                if key != "class_name"
            }

    changed = sana_report.resolve_sampling()
    changed["height"] = 512

    with pytest.raises(ValueError, match="changed from the official protocol"):
        sana_inference.generate_prompt_images(
            SimpleNamespace(
                pipeline=SimpleNamespace(),
                precision=SANA_PRECISION,
            ),
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
    import vrl.models.families.registry as model_families
    import vrl.utils.media as media

    events: list[str] = []

    class FakeModel:
        state = "base"
        precision = SANA_PRECISION

        def eval(self):
            return self

    model = FakeModel()
    bundle = SimpleNamespace(
        model=model,
        precision=SANA_PRECISION,
    )
    expected_bundle = bundle
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
            )
        ),
    )

    def fake_restore(
        checkpoint,
        *,
        bundle: object,
        family: str,
        expected_model_identity: dict,
        strict: bool,
    ):
        assert bundle is expected_bundle
        assert family == "sana"
        assert expected_model_identity == SANA_IDENTITY
        assert strict is True
        model.state = f"checkpoint-{checkpoint.next_epoch}"
        events.append(f"load:{checkpoint.next_epoch}")

    def fake_generate(model_arg, **kwargs):
        del kwargs
        assert model_arg is model
        events.append(f"generate:{model.state}")
        return [object(), object()]

    def fake_write_png(image, path):
        del image
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")

    monkeypatch.setattr(checkpoint_eval, "restore_model_checkpoint", fake_restore)
    monkeypatch.setattr(checkpoint_eval, "generate_prompt_images", fake_generate)
    monkeypatch.setattr(checkpoint_eval, "load_official_scheduler", lambda build: object())
    monkeypatch.setattr(media, "write_png", fake_write_png)
    materialized_identity = {"schema": "local", "sources": {"main": "stable"}}
    identity_calls: list[object] = []

    def resolve_materialized_identity(build):
        identity_calls.append(build)
        return materialized_identity

    monkeypatch.setattr(
        checkpoint_identity,
        "resolve_checkpoint_model_identity",
        resolve_materialized_identity,
    )
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

    root = checkpoint_eval.parse_config(
        OmegaConf.create(
            {
                "model": {"family": "sana", "path": "unit-checkpoint"},
                "precision": {
                    "float32_precision": "ieee",
                    "training": {"dtype": "fp32"},
                },
            },
        ),
    )
    precision = checkpoint_eval.PrecisionPolicy.from_section(root.precision)
    generated = checkpoint_eval._generate_images(
        root,
        precision,
        targets,
        ["fox"],
        output_dir=tmp_path / "eval",
        sampling={},
        device=torch.device("cpu"),
        expected_model_identity=SANA_IDENTITY,
    )

    assert len(generated) == 6
    assert len(identity_calls) == 2
    assert identity_calls[0] is identity_calls[1]
    assert events == [
        "generate:base",
        "read:25",
        "load:25",
        "generate:checkpoint-25",
        "read:50",
        "load:50",
        "generate:checkpoint-50",
    ]


def test_generate_images_rejects_materialized_source_drift_before_generation(
    monkeypatch,
    tmp_path,
) -> None:
    import vrl.models.families.registry as model_families

    build = object()
    built = False
    generated = False

    class FakeModel:
        def eval(self):
            return self

    def build_rollout(actual_build):
        nonlocal built
        assert actual_build is build
        built = True
        return SimpleNamespace(model=FakeModel())

    entry = SimpleNamespace(
        resolve_model_build=lambda *args, **kwargs: build,
        build_rollout=build_rollout,
    )
    monkeypatch.setattr(
        model_families,
        "get_model_family_entry",
        lambda _family: entry,
    )
    identities = iter(
        (
            {"schema": "local", "sources": {"main": "before"}},
            {"schema": "local", "sources": {"main": "after"}},
        ),
    )
    monkeypatch.setattr(
        checkpoint_identity,
        "resolve_checkpoint_model_identity",
        lambda actual_build: next(identities) if actual_build is build else None,
    )

    def fail_if_generated(*args, **kwargs):
        del args, kwargs
        nonlocal generated
        generated = True
        raise AssertionError("generation must not run after local source drift")

    monkeypatch.setattr(checkpoint_eval, "generate_prompt_images", fail_if_generated)
    root = checkpoint_eval.parse_config(
        OmegaConf.create(
            {
                "model": {"family": "sana", "path": "unit-checkpoint"},
                "precision": {
                    "float32_precision": "ieee",
                    "training": {"dtype": "fp32"},
                },
            },
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="materialized SANA model source changed during runtime construction",
    ):
        checkpoint_eval._generate_images(
            root,
            checkpoint_eval.PrecisionPolicy.from_section(root.precision),
            [checkpoint_eval.CheckpointTarget("baseline", -1, None, None, None)],
            ["fox"],
            output_dir=tmp_path / "eval",
            sampling={},
            device=torch.device("cpu"),
            expected_model_identity=SANA_IDENTITY,
        )

    assert built is True
    assert generated is False
