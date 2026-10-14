"""A Wan Diffusion-DPO run on the real tiny snapshot, end to end.

The entrypoint trains the registry-built model under the config it was given:
family aliases resolve, non-T2V families are refused before any side effect,
``actor.gradient_checkpointing`` reaches the real transformer, and the
checkpoint's model identity gates and follows the whole run (resume goes
through the real validate -> restore -> save chain).

Everything here is real code on a KB-scale snapshot; the only double is the
Pick-a-Pic Hub download, which is a network boundary: ``_install_local_pickapic``
hands ``from_hub`` an in-memory ``datasets.Dataset`` of genuine JPEG pairs, so
``PickAPicPreferenceDataset.__init__`` / ``__getitem__`` / ``collate`` still run.
"""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Any

import pytest

# The local Pick-a-Pic double is an HF ``Dataset``; that package ships with the
# ``data`` extra, which the model-only test environments do not install.
pytest.importorskip("datasets")

from tests.tiny_real_stack import write_tiny_wan_snapshot
from vrl.config.loading import load_config
from vrl.scripts.families.wan_2_1.train_dpo import train_wan_2_1_dpo
from vrl.trainers.activation_checkpointing import selective_checkpoint_func
from vrl.trainers.checkpointing import TrainingCheckpoint, read_checkpoint_meta
from vrl.trainers.data import preferences


def _tiny_dpo_run(tmp_path: Path, *, family: str = "wan_2_1", overrides: list[str] = ()) -> Any:
    """The Pick-a-Pic Wan DPO preset pointed at a tiny snapshot, one step, fp32 on CPU."""

    snapshot = tmp_path / "wan-snapshot"
    if not snapshot.exists():
        write_tiny_wan_snapshot(snapshot)
    return load_config(
        "experiment/wan_2_1/offline_dpo_pickapic",
        overrides=[
            f"model.family={family}",
            f"model.path={snapshot}",
            "model.revision=null",
            "model.use_lora=false",
            "precision.training.dtype=fp32",
            "sampling.height=32",
            "sampling.width=32",
            "trainer.max_train_steps=1",
            "trainer.checkpointing_steps=1",
            "trainer.log_interval=1",
            f"trainer.output_dir={tmp_path / 'run'}",
            "data.sampler.dataloader_num_workers=0",
            *overrides,
        ],
    )


def _install_local_pickapic(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the Hub download with one real red-vs-blue preference pair.

    Returns the list of ``from_hub`` keyword calls so a test can assert the run
    reached (or, for a guard, never reached) dataset loading.
    """

    def jpeg(color: tuple[int, int, int]) -> bytes:
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (40, 40), color).save(buffer, format="JPEG")
        return buffer.getvalue()

    rows = [
        {
            "jpg_0": jpeg((200, 10, 10)),
            "jpg_1": jpeg((10, 10, 200)),
            "label_0": 1,
            "caption": "a cat",
        }
    ]
    calls: list[dict[str, Any]] = []

    def from_hub(cls, **kwargs):
        from datasets import Dataset

        calls.append(kwargs)
        return cls(
            Dataset.from_list(rows),
            resolution=kwargs["resolution"],
            random_crop=kwargs["random_crop"],
            no_hflip=kwargs["no_hflip"],
        )

    monkeypatch.setattr(preferences.PickAPicPreferenceDataset, "from_hub", classmethod(from_hub))
    return calls


def _spy_wan_from_build(monkeypatch: pytest.MonkeyPatch, on_built=None) -> list[Any]:
    """Record every real ``WanT2VPipelineModel.from_build`` result (a spy, not a fake)."""

    from vrl.models.families.wan_2_1.model import WanT2VPipelineModel

    real = WanT2VPipelineModel.from_build.__func__
    built: list[Any] = []

    def from_build(cls, build):
        model = real(cls, build)
        built.append(model)
        if on_built is not None:
            on_built(model)
        return model

    monkeypatch.setattr(WanT2VPipelineModel, "from_build", classmethod(from_build))
    return built


@pytest.mark.parametrize("family", ["wan_2_1", "wan"])
def test_offline_dpo_trains_the_registry_model_end_to_end(monkeypatch, tmp_path, family) -> None:
    """One real DPO step on the tiny snapshot: the alias resolves to ``wan_2_1``,
    the loss is finite, and both checkpoints carry the canonical family."""

    _install_local_pickapic(monkeypatch)
    cfg = _tiny_dpo_run(tmp_path, family=family)

    train_wan_2_1_dpo(cfg)

    run = tmp_path / "run"
    with (run / "metrics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["step"] for row in rows] == ["0"]
    assert math.isfinite(float(rows[0]["loss"]))
    for name in ("checkpoint-1", "checkpoint-final"):
        meta = read_checkpoint_meta(run / name)
        assert meta["family"] == "wan_2_1"
        assert meta["uses_lora"] is False
        assert meta["next_step"] == 1


@pytest.mark.parametrize("family", ["wan_2_1_i2v", "sd3_5"])
def test_offline_dpo_rejects_non_t2v_wan_family_before_runtime_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    family: str,
) -> None:
    cfg = _tiny_dpo_run(tmp_path, family=family)
    if family == "sd3_5":
        del cfg.sampling.num_frames
    dataset_calls = _install_local_pickapic(monkeypatch)
    built = _spy_wan_from_build(monkeypatch)

    with pytest.raises(
        ValueError,
        match=rf"Wan Diffusion-DPO requires model\.family='wan_2_1'.*got '{family}'",
    ):
        train_wan_2_1_dpo(cfg)

    assert built == []
    assert dataset_calls == []


@pytest.mark.parametrize(
    ("checkpointing", "expected_mode"),
    [
        ('"off"', "off"),
        ("true", "full"),
        ('"selective"', "selective"),
    ],
)
def test_offline_dpo_uses_shared_gradient_checkpointing_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    checkpointing: str,
    expected_mode: str,
) -> None:
    """The policy lands on the real transformer: off leaves it untouched, full
    enables diffusers' default recompute, selective installs the SAC function."""

    _install_local_pickapic(monkeypatch)
    cfg = _tiny_dpo_run(tmp_path, overrides=[f"actor.gradient_checkpointing={checkpointing}"])
    built = _spy_wan_from_build(monkeypatch)

    train_wan_2_1_dpo(cfg)

    (model,) = built
    transformer = model.transformer
    assert transformer.is_gradient_checkpointing is (expected_mode != "off")
    funcs = {
        block._gradient_checkpointing_func
        for block in transformer.modules()
        if getattr(block, "gradient_checkpointing", False)
    }
    if expected_mode == "selective":
        assert funcs == {selective_checkpoint_func}
    elif expected_mode == "full":
        assert funcs and selective_checkpoint_func not in funcs


def test_matching_identity_restores_progress_and_keeps_saving_it(monkeypatch, tmp_path) -> None:
    _install_local_pickapic(monkeypatch)
    train_wan_2_1_dpo(_tiny_dpo_run(tmp_path))
    run = tmp_path / "run"
    first = read_checkpoint_meta(run / "checkpoint-1")
    assert first["model_identity"]["sources"]  # the real local snapshot hash

    train_wan_2_1_dpo(
        _tiny_dpo_run(
            tmp_path,
            overrides=[
                f"trainer.resume_from={run / 'checkpoint-1'}",
                "trainer.max_train_steps=2",
            ],
        ),
    )

    # The resumed run logged only step 1 and its checkpoints continue the
    # restored trainer: global_step 2 is impossible for a fresh trainer that ran
    # one step, so it proves the restore threaded through the optimizer state.
    with (run / "metrics.csv").open(encoding="utf-8") as handle:
        assert [row["step"] for row in csv.DictReader(handle)] == ["0", "1"]
    for name in ("checkpoint-2", "checkpoint-final"):
        checkpoint = TrainingCheckpoint.load(run / name)
        assert checkpoint.next_step == 2
        assert checkpoint.payload["trainer"]["global_step"] == 2
        assert checkpoint.meta["model_identity"] == first["model_identity"]


def test_identity_mismatch_stops_before_model_and_dataset(monkeypatch, tmp_path) -> None:
    _install_local_pickapic(monkeypatch)
    train_wan_2_1_dpo(_tiny_dpo_run(tmp_path))
    checkpoint = tmp_path / "run" / "checkpoint-1"
    other = write_tiny_wan_snapshot(tmp_path / "other-snapshot")
    (other / "provenance.txt").write_text("a different model directory", encoding="utf-8")
    dataset_calls = _install_local_pickapic(monkeypatch)
    built = _spy_wan_from_build(monkeypatch)

    with pytest.raises(ValueError, match="model identity mismatch"):
        train_wan_2_1_dpo(
            _tiny_dpo_run(
                tmp_path,
                overrides=[f"model.path={other}", f"trainer.resume_from={checkpoint}"],
            ),
        )

    assert built == []
    assert dataset_calls == []


def test_source_change_during_model_load_stops_before_dataset(monkeypatch, tmp_path) -> None:
    _install_local_pickapic(monkeypatch)
    train_wan_2_1_dpo(_tiny_dpo_run(tmp_path))
    checkpoint = tmp_path / "run" / "checkpoint-1"
    snapshot = tmp_path / "wan-snapshot"
    dataset_calls = _install_local_pickapic(monkeypatch)

    def drift(_model) -> None:
        (snapshot / "extra-weights.bin").write_bytes(b"drift")

    built = _spy_wan_from_build(monkeypatch, on_built=drift)

    with pytest.raises(
        RuntimeError,
        match="model checkpoint source changed during bundle construction",
    ):
        train_wan_2_1_dpo(_tiny_dpo_run(tmp_path, overrides=[f"trainer.resume_from={checkpoint}"]))

    assert len(built) == 1
    assert dataset_calls == []
