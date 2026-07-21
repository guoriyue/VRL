from __future__ import annotations

import json

import pytest
import torch

from vrl.trainers.checkpointing import (
    CHECKPOINT_META_NAME,
    LORA_WEIGHTS_NAME,
    TRAINING_CHECKPOINT_NAME,
    infer_next_epoch,
    load_trainable_state,
    load_training_checkpoint,
    restore_training_checkpoint,
    save_training_checkpoint,
)


def test_training_checkpoint_round_trips_trainer_and_trainable_modules(tmp_path) -> None:
    """Checks training checkpoint round trips trainer and trainable modules."""
    trainer = _Trainer()
    source = _Bundle()
    with torch.no_grad():
        source.module.weight.fill_(3.0)
    save_training_checkpoint(
        tmp_path / "checkpoint-2",
        trainer=trainer,
        bundle=source,
        family="unit",
        progress={"next_epoch": 2, "global_step": 5},
        rng_state={},
    )

    checkpoint = load_training_checkpoint(tmp_path / "checkpoint-2")
    restored = _Bundle()
    with torch.no_grad():
        restored.module.weight.fill_(0.0)
    restore_training_checkpoint(
        checkpoint,
        trainer=trainer,
        bundle=restored,
        family="unit",
        strict=True,
    )

    assert (tmp_path / "checkpoint-2" / TRAINING_CHECKPOINT_NAME).exists()
    assert checkpoint.next_epoch == 2
    assert trainer.loaded == {"step": 2, "global_step": 5}
    assert restored.module.weight.item() == pytest.approx(3.0)


def test_restore_training_checkpoint_rejects_family_mismatch(tmp_path) -> None:
    """The explicit runtime family rejects a checkpoint from another family."""
    trainer = _Trainer()
    source = _Bundle()
    save_training_checkpoint(
        tmp_path / "checkpoint-janus",
        trainer=trainer,
        bundle=source,
        family="janus_pro",
        progress={"next_epoch": 1, "global_step": 1},
        rng_state={},
    )

    checkpoint = load_training_checkpoint(tmp_path / "checkpoint-janus")
    with pytest.raises(ValueError, match="family mismatch"):
        restore_training_checkpoint(
            checkpoint,
            trainer=trainer,
            bundle=_Bundle(),
            family="wan_2_1",
            strict=True,
        )


def test_restore_training_checkpoint_strictly_checks_model_identity(tmp_path) -> None:
    identity = {
        "model_path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "revision": "pinned",
        "boundary_ratio": 0.9,
        "trainable_transformers": ["transformer", "transformer_2"],
    }
    path = tmp_path / "checkpoint-dual-expert"
    save_training_checkpoint(
        path,
        trainer=_Trainer(),
        bundle=_Bundle(),
        family="wan_2_1_i2v",
        progress={"next_epoch": 1},
        rng_state={},
        model_identity=identity,
    )
    checkpoint = load_training_checkpoint(path)

    restore_training_checkpoint(
        checkpoint,
        trainer=_Trainer(),
        bundle=_Bundle(),
        family="wan_2_1_i2v",
        expected_model_identity=identity,
        strict=True,
    )
    wrong = {**identity, "boundary_ratio": 0.8}
    with pytest.raises(ValueError, match="model identity mismatch"):
        restore_training_checkpoint(
            checkpoint,
            trainer=_Trainer(),
            bundle=_Bundle(),
            family="wan_2_1_i2v",
            expected_model_identity=wrong,
            strict=True,
        )


def test_restore_training_checkpoint_routes_model_load_through_strategy(tmp_path) -> None:
    """Distributed model restore must use the same strategy seam as export."""

    class _SpyStrategy:
        def __init__(self) -> None:
            self.calls = []

        def load_trainable_state(self, bundle, state, *, strict=True):
            self.calls.append((bundle, state, strict))

    source = _Bundle()
    save_training_checkpoint(
        tmp_path / "checkpoint-strategy-restore",
        trainer=_Trainer(),
        bundle=source,
        family="unit",
        progress={"next_epoch": 1},
        rng_state={},
    )
    checkpoint = load_training_checkpoint(tmp_path / "checkpoint-strategy-restore")
    trainer = _Trainer()
    trainer._strategy = _SpyStrategy()
    restored = _Bundle()

    restore_training_checkpoint(
        checkpoint,
        trainer=trainer,
        bundle=restored,
        family="unit",
        strict=True,
    )

    assert trainer._strategy.calls == [(restored, checkpoint.trainable_state, True)]


def test_save_training_checkpoint_routes_export_through_strategy(tmp_path) -> None:
    """When a strategy is wired, checkpoint trainable state comes from its export.

    Locks sprint P3 ownership: the checkpoint reads the strategy seam, not the
    bundle's raw modules directly, so the future FSDP full-state export controls
    what lands on disk. The spy returns weights the bundle does not hold, so a
    match proves the strategy -- not ``export_trainable_state(bundle)`` -- was used.
    """

    class _SpyStrategy:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def export_trainable_state(self, bundle):
            self.calls.append(bundle)
            return {"module": {"weight": torch.full((1, 1), 9.0)}}

    strategy = _SpyStrategy()
    bundle = _Bundle()  # module weight is the Linear default, never 9.0
    save_training_checkpoint(
        tmp_path / "checkpoint-strategy",
        trainer=_Trainer(),
        bundle=bundle,
        family="unit",
        progress={"next_epoch": 1},
        rng_state={},
        strategy=strategy,
    )

    checkpoint = load_training_checkpoint(tmp_path / "checkpoint-strategy")
    assert strategy.calls == [bundle]
    assert checkpoint.trainable_state["module"]["weight"].item() == pytest.approx(9.0)


def test_save_training_checkpoint_non_primary_gathers_but_writes_nothing(tmp_path) -> None:
    """Multi-rank contract: the trainable-state export (a COLLECTIVE under FSDP2)
    runs on every rank, but only the primary rank writes files.

    Gating the whole save to rank0 would deadlock FSDP — rank0 waits at the
    all-gather for peers that never call it. So save_training_checkpoint(is_primary
    =False) MUST still invoke the strategy export (joining the collective) yet
    create no checkpoint directory/files. The spy proves the export ran; the empty
    tmp_path proves nothing was written."""

    class _SpyStrategy:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def export_trainable_state(self, bundle):
            self.calls.append(bundle)
            return {"module": {"weight": torch.full((1, 1), 9.0)}}

    strategy = _SpyStrategy()
    bundle = _Bundle()
    ckpt_dir = tmp_path / "checkpoint-nonprimary"
    out = save_training_checkpoint(
        ckpt_dir,
        trainer=_Trainer(),
        bundle=bundle,
        family="unit",
        progress={"next_epoch": 1},
        rng_state={},
        strategy=strategy,
        is_primary=False,
    )

    assert strategy.calls == [bundle]  # joined the collective gather
    assert out == {}  # no meta returned
    assert not ckpt_dir.exists()  # rank0-only IO: nothing written on a non-primary rank


def test_training_checkpoint_writes_optional_lora_export(tmp_path) -> None:
    """Checks training checkpoint writes optional LoRA export."""

    class _ExportModule:
        def save_pretrained(self, path):
            path.mkdir(parents=True)
            (path / "adapter_model.safetensors").write_text("stub")

    save_training_checkpoint(
        tmp_path / "checkpoint-1",
        trainer=_Trainer(),
        bundle=_Bundle(),
        family="unit",
        progress={"next_epoch": 1},
        rng_state={},
        export_modules={LORA_WEIGHTS_NAME: _ExportModule()},
    )

    assert (tmp_path / "checkpoint-1" / TRAINING_CHECKPOINT_NAME).exists()
    assert (tmp_path / "checkpoint-1" / LORA_WEIGHTS_NAME / "adapter_model.safetensors").exists()
    assert (tmp_path / "checkpoint-1" / CHECKPOINT_META_NAME).exists()


def test_training_checkpoint_exports_lora_with_ema_without_mutating_resume_state(
    tmp_path,
) -> None:
    """Checks training checkpoint exports LoRA with EMA without mutating resume state."""

    class _ExportModule:
        def __init__(self, module) -> None:
            self.module = module
            self.saved_weight = None

        def save_pretrained(self, path):
            path.mkdir(parents=True)
            self.saved_weight = float(self.module.weight.item())
            (path / "adapter_model.safetensors").write_text(str(self.saved_weight))

    class _EMA:
        def copy_ema_to(self, parameters, *, store_temp=True):
            self.parameters = list(parameters)
            assert store_temp is True
            self.temp = [p.detach().clone() for p in self.parameters]
            with torch.no_grad():
                for p in self.parameters:
                    p.fill_(7.0)

        def copy_temp_to(self, parameters):
            with torch.no_grad():
                for p, temp in zip(parameters, self.temp, strict=True):
                    p.copy_(temp)

    bundle = _Bundle()
    with torch.no_grad():
        bundle.module.weight.fill_(3.0)
    export_module = _ExportModule(bundle.module)

    save_training_checkpoint(
        tmp_path / "checkpoint-ema",
        trainer=_Trainer(),
        bundle=bundle,
        family="unit",
        progress={"next_epoch": 1},
        rng_state={},
        export_modules={LORA_WEIGHTS_NAME: export_module},
        export_ema=_EMA(),
    )

    checkpoint = load_training_checkpoint(tmp_path / "checkpoint-ema")
    saved_trainable = checkpoint.trainable_state["module"]["weight"].item()

    assert saved_trainable == pytest.approx(3.0)
    assert export_module.saved_weight == pytest.approx(7.0)
    assert bundle.module.weight.item() == pytest.approx(3.0)


def test_training_checkpoint_skips_lora_ema_export_before_first_ema_update(
    tmp_path,
) -> None:
    """Checks training checkpoint skips LoRA EMA export before first EMA update."""

    class _ExportModule:
        def __init__(self, module) -> None:
            self.module = module
            self.saved_weight = None

        def save_pretrained(self, path):
            path.mkdir(parents=True)
            self.saved_weight = float(self.module.weight.item())
            (path / "adapter_model.safetensors").write_text(str(self.saved_weight))

    class _UnupdatedEMA:
        has_updates = False

        def __init__(self) -> None:
            self.copy_called = False

        def copy_ema_to(self, parameters, *, store_temp=True):
            del parameters, store_temp
            self.copy_called = True

        def copy_temp_to(self, parameters):
            del parameters
            raise AssertionError("raw export must not restore an unused EMA swap")

    bundle = _Bundle()
    with torch.no_grad():
        bundle.module.weight.fill_(3.0)
    export_module = _ExportModule(bundle.module)
    ema = _UnupdatedEMA()

    save_training_checkpoint(
        tmp_path / "checkpoint-raw-export",
        trainer=_Trainer(),
        bundle=bundle,
        family="unit",
        progress={"next_epoch": 1},
        rng_state={},
        export_modules={LORA_WEIGHTS_NAME: export_module},
        export_ema=ema,
    )

    assert export_module.saved_weight == pytest.approx(3.0)
    assert ema.copy_called is False
    assert bundle.module.weight.item() == pytest.approx(3.0)


def test_load_training_checkpoint_requires_checkpoint_pt(tmp_path) -> None:
    """Checks load training checkpoint requires checkpoint pt."""
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()

    with pytest.raises(FileNotFoundError, match=TRAINING_CHECKPOINT_NAME):
        load_training_checkpoint(ckpt)


def test_load_training_checkpoint_rejects_bad_schema(tmp_path) -> None:
    """Checks load training checkpoint rejects bad schema."""
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()
    torch.save({"schema_version": 999}, ckpt / TRAINING_CHECKPOINT_NAME)

    with pytest.raises(ValueError, match="schema_version"):
        load_training_checkpoint(ckpt)


def test_load_trainable_state_strict_rejects_key_mismatch() -> None:
    """Checks load trainable state strict rejects key mismatch."""
    with pytest.raises(ValueError, match="missing"):
        load_trainable_state(_Bundle(), {}, strict=True)


def test_infer_next_epoch_falls_back_to_trainer_step_for_checkpoint_final(tmp_path) -> None:
    """Checks infer next epoch falls back to trainer step for checkpoint final."""
    ckpt = tmp_path / "checkpoint-final"
    ckpt.mkdir()

    assert infer_next_epoch(ckpt, {"step": 12}, None) == 12


def test_infer_next_epoch_falls_back_to_numeric_checkpoint_suffix(tmp_path) -> None:
    """Checks infer next epoch falls back to numeric checkpoint suffix."""
    ckpt = tmp_path / "checkpoint-42"
    ckpt.mkdir()

    assert infer_next_epoch(ckpt, {}, {}) == 42


def test_load_training_checkpoint_rejects_non_object_meta(tmp_path) -> None:
    """Checks load training checkpoint rejects non object meta."""
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()
    torch.save(
        {
            "schema_version": 1,
            "trainer": {},
            "model": {"trainable_modules": {}},
            "progress": {},
            "rng": {},
        },
        ckpt / TRAINING_CHECKPOINT_NAME,
    )
    (ckpt / CHECKPOINT_META_NAME).write_text(json.dumps([{"next_epoch": 1}]))

    with pytest.raises(TypeError, match="JSON object"):
        load_training_checkpoint(ckpt)


class _Trainer:
    def __init__(self) -> None:
        self.loaded = None

    def state_dict(self):
        return {"step": 2, "global_step": 5}

    def load_state_dict(self, state, *, strict=True):
        del strict
        self.loaded = dict(state)


class _Bundle:
    def __init__(self) -> None:
        import torch.nn as nn

        self.module = nn.Linear(1, 1, bias=False)
        self.trainable_modules = {"module": self.module}


def test_checkpoint_publish_is_atomic(tmp_path) -> None:
    """A save that dies mid-write leaves no directory a resume would trust."""
    from vrl.trainers.checkpointing import (
        find_latest_complete_checkpoint,
        is_complete_checkpoint,
    )

    class _ExplodingExport:
        def save_pretrained(self, *_args, **_kwargs):
            raise RuntimeError("disk full")

    target = tmp_path / "checkpoint-3"
    with pytest.raises(RuntimeError, match="disk full"):
        save_training_checkpoint(
            target,
            trainer=_Trainer(),
            bundle=_Bundle(),
            family="unit",
            progress={"next_epoch": 3, "global_step": 3},
            rng_state={},
            export_modules={"lora_weights": _ExplodingExport()},
        )

    # Neither the final directory nor a trusted leftover exists.
    assert not target.exists()
    assert find_latest_complete_checkpoint(tmp_path) is None
    for leftover in tmp_path.iterdir():
        assert not is_complete_checkpoint(leftover)


def test_published_checkpoint_is_complete_and_size_stamped(tmp_path) -> None:
    from vrl.trainers.checkpointing import (
        TRAINING_CHECKPOINT_NAME,
        is_complete_checkpoint,
        read_checkpoint_meta,
    )

    target = tmp_path / "checkpoint-7"
    save_training_checkpoint(
        target,
        trainer=_Trainer(),
        bundle=_Bundle(),
        family="unit",
        progress={"next_epoch": 7, "global_step": 7},
        rng_state={},
    )

    assert is_complete_checkpoint(target)
    meta = read_checkpoint_meta(target)
    assert meta["checkpoint_file_bytes"] == (target / TRAINING_CHECKPOINT_NAME).stat().st_size
    # Truncation is detected without loading the payload.
    (target / TRAINING_CHECKPOINT_NAME).write_bytes(b"torn")
    assert not is_complete_checkpoint(target)


def test_find_latest_complete_checkpoint_skips_staging_and_orders_by_step(tmp_path) -> None:
    from vrl.trainers.checkpointing import find_latest_complete_checkpoint

    class _StepTrainer(_Trainer):
        def __init__(self, step: int) -> None:
            super().__init__()
            self._step = step

        def state_dict(self):
            return {"step": self._step, "global_step": self._step}

    for step in (2, 10):
        save_training_checkpoint(
            tmp_path / f"checkpoint-{step}",
            trainer=_StepTrainer(step),
            bundle=_Bundle(),
            family="unit",
            progress={"next_epoch": step, "global_step": step},
            rng_state={},
        )
    # A staging leftover (crash before publish) and a torn dir are both skipped.
    (tmp_path / "checkpoint-99.tmp-abc").mkdir()
    torn = tmp_path / "checkpoint-50"
    torn.mkdir()
    (torn / "checkpoint.pt").write_bytes(b"torn")

    latest = find_latest_complete_checkpoint(tmp_path)
    assert latest is not None and latest.name == "checkpoint-10"


def test_resave_to_existing_checkpoint_dir_replaces_it(tmp_path) -> None:
    """Crash-loop overwriting the same checkpoint-N republishes cleanly."""
    from vrl.trainers.checkpointing import is_complete_checkpoint

    target = tmp_path / "checkpoint-1"
    for _ in range(2):
        save_training_checkpoint(
            target,
            trainer=_Trainer(),
            bundle=_Bundle(),
            family="unit",
            progress={"next_epoch": 1, "global_step": 1},
            rng_state={},
        )
    assert is_complete_checkpoint(target)
    assert load_training_checkpoint(target).payload["family"] == "unit"


def test_prepare_metrics_csv_rejects_resume_across_schema_change(tmp_path) -> None:
    """Appending new-schema rows under an old header silently shifts columns."""
    from vrl.trainers.checkpointing import prepare_metrics_csv

    path = tmp_path / "metrics.csv"
    header = "epoch,loss,approx_kl\n"

    prepare_metrics_csv(path, header, resume_at=None)
    prepare_metrics_csv(path, header, resume_at=("epoch", 0))

    with pytest.raises(ValueError, match="different metrics schema"):
        prepare_metrics_csv(
            path,
            "epoch,loss,active_clip_fraction,approx_kl\n",
            resume_at=("epoch", 0),
        )


def test_prepare_metrics_csv_discards_rows_not_covered_by_checkpoint(tmp_path) -> None:
    from vrl.trainers.checkpointing import prepare_metrics_csv

    path = tmp_path / "metrics.csv"
    header = "epoch,loss\n"
    path.write_text(header + "38,1.0\n39,0.9\n40,0.8\n41,0.7\n42,0")

    prepare_metrics_csv(path, header, resume_at=("epoch", 40))

    assert path.read_text() == header + "38,1.0\n39,0.9\n"


@pytest.mark.parametrize(
    "rows",
    [
        "0,1.0\n0,0.9\n",
        "1,1.0\n0,0.9\n",
        "0.5,1.0\n",
        "-1,1.0\n",
    ],
)
def test_prepare_metrics_csv_rejects_invalid_resume_positions(tmp_path, rows) -> None:
    from vrl.trainers.checkpointing import prepare_metrics_csv

    path = tmp_path / "metrics.csv"
    header = "epoch,loss\n"
    path.write_text(header + rows)

    with pytest.raises(ValueError, match=r"integer|strictly increasing"):
        prepare_metrics_csv(path, header, resume_at=("epoch", 2))
