"""Worker-side rollout checkpoint identity gates.

Each test hands ``GenerationWorkerCore`` the launch contract the real driver
builds (``ResolvedOnlineRun.ray_launch_inputs``) for the tiny SANA snapshot,
so the identity is the real local-directory hash and the executor is the real
family executor; the Hub load is the only double.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.scripts.eval.fixtures import (
    TinySanaPipeline,
    tiny_sana_online_config,
)
from vrl import run
from vrl.generation.bindings.full_sequence.executor import GenericDenoiseBatchExecutor
from vrl.generation.execution.worker import GenerationWorkerCore
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs


def _launch_inputs(
    tmp_path: Path,
    *,
    overrides: tuple[str, ...] = (),
    on_snapshot=None,
) -> RayGenerationLaunchInputs:
    cfg = tiny_sana_online_config(tmp_path, overrides=overrides)
    if on_snapshot is not None:
        on_snapshot(tmp_path / "sana-snapshot")
    resolved = run.resolve_online_run(cfg)
    replay = run.resolve_model(
        resolved.family,
        resolved.built.root,
        resolved.device,
        precision=resolved.built.precision,
        for_rollout=False,
    )
    return resolved.ray_launch_inputs(replay)


def _worker(inputs: RayGenerationLaunchInputs) -> GenerationWorkerCore:
    return GenerationWorkerCore("rollout-0", inputs.launch_contract, inputs.gatherer)


def test_worker_accepts_matching_identity_before_and_after_model_build(
    monkeypatch, tmp_path
) -> None:
    inputs = _launch_inputs(tmp_path)
    pipeline = TinySanaPipeline()
    pipeline.install(monkeypatch, tmp_path / "sana-snapshot")
    worker = _worker(inputs)

    worker.load_policy()

    assert isinstance(worker.executor, GenericDenoiseBatchExecutor)
    assert (worker.executor.family, worker.executor.task) == ("sana", "t2i")
    assert pipeline.loads == 1


def test_worker_rehydrates_generation_memory_before_family_build(monkeypatch, tmp_path) -> None:
    """The VAE memory policy crosses Ray as primitives and lands on the real VAE."""

    inputs = _launch_inputs(
        tmp_path,
        overrides=("model.memory.vae_decode.tiling=true", "model.memory.vae_decode.slicing=false"),
    )
    assert inputs.launch_contract.model_build["generation_memory"] == {
        "vae_decode": {"tiling": True, "slicing": False},
        "cpu_resident": (),
    }
    pipeline = TinySanaPipeline()
    pipeline.install(monkeypatch, tmp_path / "sana-snapshot")

    _worker(inputs).load_policy()

    assert pipeline.vae.use_tiling is True
    assert pipeline.vae.use_slicing is False


def test_worker_rejects_cross_node_identity_mismatch_before_model_build(
    monkeypatch, tmp_path
) -> None:
    """The driver resolved a different model directory than the worker sees."""

    inputs = _launch_inputs(tmp_path)
    pipeline = TinySanaPipeline()
    pipeline.install(monkeypatch, tmp_path / "sana-snapshot")
    driver_inputs = _launch_inputs(
        tmp_path / "driver",
        on_snapshot=lambda snapshot: (snapshot / "provenance.txt").write_text("other checkout"),
    )
    assert driver_inputs.launch_contract.expected_model_identity != (
        inputs.launch_contract.expected_model_identity
    )
    worker = _worker(
        replace(
            inputs,
            launch_contract=replace(
                inputs.launch_contract,
                expected_model_identity=driver_inputs.launch_contract.expected_model_identity,
            ),
        ),
    )

    with pytest.raises(ValueError, match="model identity mismatch before model construction"):
        worker.load_policy()

    assert pipeline.loads == 0
    assert worker.executor is None


def test_worker_rejects_source_change_during_model_build(monkeypatch, tmp_path) -> None:
    inputs = _launch_inputs(tmp_path)
    snapshot = tmp_path / "sana-snapshot"
    pipeline = TinySanaPipeline()
    pipeline.install(
        monkeypatch,
        snapshot,
        on_load=lambda: (snapshot / "extra-weights.bin").write_bytes(b"drift"),
    )
    worker = _worker(inputs)

    with pytest.raises(
        RuntimeError,
        match="model checkpoint source changed during bundle construction",
    ):
        worker.load_policy()

    assert pipeline.loads == 1
    assert worker.executor is None
