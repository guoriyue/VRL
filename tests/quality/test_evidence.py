from __future__ import annotations

import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch.nn as nn
from omegaconf import OmegaConf

from tests.quality.contracts import InferencePhase
from tests.quality.io import sha256_file
from tests.quality.preflight import (
    evaluate_inference_preflight,
    load_quality_protocol,
    quality_provenance,
)
from vrl.utils.media import write_png


def _config(tmp_path: Path, *, family: str = "sd3_5", model_path: str | None = None):
    paths = {
        "sd3_5": "stabilityai/stable-diffusion-3.5-medium",
        "hunyuan_video": "hunyuanvideo-community/HunyuanVideo",
        "wan_2_1_i2v": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    }
    revisions = {
        "sd3_5": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
        "hunyuan_video": "e8c2aaa66fe3742a32c11a6766aecbf07c56e773",
        "wan_2_1_i2v": "b184e23a8a16b20f108f727c902e769e873ffc73",
    }
    return OmegaConf.create(
        {
            "model": {
                "family": family,
                "path": model_path or paths[family],
                "revision": revisions[family],
            },
            "algorithm": {"kind": "grpo"},
            "sampling": {"width": 32, "height": 32, "num_steps": 4},
            "rollout": {"sde": {"type": "flow_grpo"}},
            "trainer": {"output_dir": str(tmp_path / "run"), "total_epochs": 30},
        },
    )


def _texture(seed: int, *, frames: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    return np.stack([np.roll(base, shift=index, axis=1) for index in range(frames)])


def _sample(
    protocol,
    *,
    prompt: str,
    seed: int,
    native: str,
    production: str,
    condition_delta: float | None = None,
) -> dict:
    row = {
        "prompt": prompt,
        "seed": seed,
        "native": native,
        "production": production,
        "replay_max_abs_error": 0.0,
        "matched_alignment": 0.8,
        "shuffled_alignment": 0.2,
        "segments": list(protocol.required_segments),
        "clean_quality_score": 1.0,
        "corruption_scores": {name: 0.0 for name in protocol.required_corruptions},
    }
    if condition_delta is not None:
        row["condition_delta"] = condition_delta
    return row


def _write_image_evidence(tmp_path: Path, cfg, *, color_blocks: bool = False) -> Path:
    protocol = load_quality_protocol(cfg)
    samples = []
    for index in range(2):
        native = _texture(index)[0]
        production = native.copy()
        if color_blocks:
            production[:16, :16] = (255, 0, 0)
            production[:16, 16:] = (0, 255, 0)
            production[16:, :16] = (0, 0, 255)
            production[16:, 16:] = (255, 255, 0)
        native_path = tmp_path / f"native-{index}.png"
        production_path = tmp_path / f"production-{index}.png"
        write_png(native, native_path)
        write_png(production, production_path)
        samples.append(
            _sample(
                protocol,
                prompt=f"prompt {index}",
                seed=index,
                native=native_path.name,
                production=production_path.name,
            ),
        )
    manifest = {
        "schema_version": 1,
        "family": "sd3_5",
        "phase": "base",
        **quality_provenance(cfg),
        "scorer": {"name": "test-semantic", "revision": "1"},
        "samples": samples,
    }
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(manifest))
    return path


def test_image_evidence_passes_and_artifact_mutation_fails_recheck(tmp_path) -> None:
    cfg = _config(tmp_path)
    evidence = _write_image_evidence(tmp_path, cfg)

    report = evaluate_inference_preflight(cfg, evidence)

    assert report["verdict"] == "pass"

    write_png(np.zeros((32, 32, 3), dtype=np.uint8), tmp_path / "production-0.png")
    mutated = evaluate_inference_preflight(cfg, evidence)
    assert mutated["verdict"] == "fail"


def test_color_blocks_fail_even_when_scores_claim_they_are_good(tmp_path) -> None:
    cfg = _config(tmp_path)
    evidence = _write_image_evidence(tmp_path, cfg, color_blocks=True)

    report = evaluate_inference_preflight(cfg, evidence)

    assert report["verdict"] == "fail"
    assert any(
        not assertion["passed"] and "native_similarity" in assertion["name"]
        for assertion in report["assertions"]
    )


def test_reward_hacking_corruption_direction_fails(tmp_path) -> None:
    cfg = _config(tmp_path)
    evidence = _write_image_evidence(tmp_path, cfg)
    manifest = json.loads(evidence.read_text())
    manifest["samples"][0]["corruption_scores"]["color_blocks"] = 1.5
    evidence.write_text(json.dumps(manifest))

    report = evaluate_inference_preflight(cfg, evidence)

    assert report["verdict"] == "fail"
    failed = {item["name"] for item in report["assertions"] if not item["passed"]}
    assert "sample_0.corruption.color_blocks" in failed


def test_evidence_from_a_different_resolved_config_cannot_be_reblessed(tmp_path) -> None:
    cfg = _config(tmp_path)
    evidence = _write_image_evidence(tmp_path, cfg)
    cfg.sampling.num_steps = 5

    report = evaluate_inference_preflight(cfg, evidence)

    assert report["verdict"] == "fail"
    assert any(
        item["name"] == "config_sha256_identity" and not item["passed"]
        for item in report["assertions"]
    )


def test_evidence_from_a_different_model_identity_fails_closed(tmp_path) -> None:
    cfg = _config(tmp_path)
    evidence = _write_image_evidence(tmp_path, cfg)
    manifest = json.loads(evidence.read_text())
    manifest["model_identity"] = "someone/other-model@other-revision"
    evidence.write_text(json.dumps(manifest))

    report = evaluate_inference_preflight(cfg, evidence)

    assert report["verdict"] == "fail"
    assert any(
        item["name"] == "model_identity_identity" and not item["passed"]
        for item in report["assertions"]
    )


def test_source_fingerprint_changes_when_live_source_changes(monkeypatch, tmp_path) -> None:
    from tests.quality import io

    package_root = tmp_path / "vrl"
    quality_dir = package_root / "quality"
    quality_dir.mkdir(parents=True)
    fake_io = quality_dir / "io.py"
    fake_io.write_text("# identity anchor\n")
    source = package_root / "runtime.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.setattr(io, "__file__", str(fake_io))
    io._source_tree_snapshot_sha256.cache_clear()

    before = io.source_tree_sha256()
    source.write_text("VALUE = 2\n")
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    after = io.source_tree_sha256()

    assert after != before


def _write_video(path: Path, frames: np.ndarray) -> None:
    imageio.mimsave(path, list(frames), format="GIF", duration=0.1)


def _write_video_evidence(tmp_path: Path, cfg, *, static: bool, condition_delta=None) -> Path:
    protocol = load_quality_protocol(cfg)
    samples = []
    for index in range(2):
        native = _texture(index + 10, frames=4)
        production = np.repeat(native[:1], 4, axis=0) if static else native.copy()
        native_path = tmp_path / f"native-{index}.gif"
        production_path = tmp_path / f"production-{index}.gif"
        _write_video(native_path, native)
        _write_video(production_path, production)
        samples.append(
            _sample(
                protocol,
                prompt=f"video prompt {index}",
                seed=index,
                native=native_path.name,
                production=production_path.name,
                condition_delta=condition_delta,
            ),
        )
    manifest = {
        "schema_version": 1,
        "family": str(cfg.model.family),
        "phase": "base",
        **quality_provenance(cfg),
        "scorer": {"name": "test-video-semantic", "revision": "1"},
        "samples": samples,
    }
    path = tmp_path / "video-evidence.json"
    path.write_text(json.dumps(manifest))
    return path


def test_video_matching_native_passes_and_static_divergent_video_fails(tmp_path) -> None:
    cfg = _config(tmp_path, family="hunyuan_video")

    passing = evaluate_inference_preflight(
        cfg,
        _write_video_evidence(tmp_path, cfg, static=False),
    )
    failing = evaluate_inference_preflight(
        cfg,
        _write_video_evidence(tmp_path, cfg, static=True),
    )

    assert passing["verdict"] == "pass"
    assert failing["verdict"] == "fail"
    assert any(
        not item["passed"] and "native_similarity" in item["name"]
        for item in failing["assertions"]
    )


def test_conditioned_video_requires_a_real_condition_delta(tmp_path) -> None:
    cfg = _config(tmp_path, family="wan_2_1_i2v")

    report = evaluate_inference_preflight(
        cfg,
        _write_video_evidence(
            tmp_path,
            cfg,
            static=False,
            condition_delta=0.0,
        ),
    )

    assert report["verdict"] == "fail"
    assert any(
        not item["passed"] and "condition_delta" in item["name"]
        for item in report["assertions"]
    )


def test_checkpoint_report_binds_the_complete_checkpoint_hash(tmp_path) -> None:
    from vrl.trainers.checkpointing import TRAINING_CHECKPOINT_NAME, save_training_checkpoint

    class Trainer:
        def state_dict(self):
            return {"step": 1, "global_step": 1}

    class Bundle:
        def __init__(self) -> None:
            self.trainable_modules = {"module": nn.Linear(1, 1, bias=False)}
            self.metadata = {"family": "sd3_5"}

    checkpoint = tmp_path / "run" / "checkpoint-1"
    save_training_checkpoint(
        checkpoint,
        trainer=Trainer(),
        bundle=Bundle(),
        family="sd3_5",
        progress={"completed_epoch": 1, "next_epoch": 1, "global_step": 1},
        rng_state={},
    )
    cfg = _config(tmp_path)
    evidence = _write_image_evidence(tmp_path, cfg)
    manifest = json.loads(evidence.read_text())
    manifest["phase"] = "checkpoint"
    manifest["checkpoint_sha256"] = sha256_file(checkpoint / TRAINING_CHECKPOINT_NAME)
    evidence.write_text(json.dumps(manifest))

    report = evaluate_inference_preflight(
        cfg,
        evidence,
        phase=InferencePhase.CHECKPOINT,
        checkpoint_file=checkpoint / TRAINING_CHECKPOINT_NAME,
    )

    assert report["verdict"] == "pass"

    with (checkpoint / TRAINING_CHECKPOINT_NAME).open("ab") as handle:
        handle.write(b"corrupt")
    corrupted = evaluate_inference_preflight(
        cfg,
        evidence,
        phase=InferencePhase.CHECKPOINT,
        checkpoint_file=checkpoint / TRAINING_CHECKPOINT_NAME,
    )
    assert corrupted["verdict"] == "fail"
