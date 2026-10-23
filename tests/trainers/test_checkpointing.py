from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from tests.trainers._checkpoint_helpers import (
    UNIT_IDENTITY,
    _Bundle,
    _ema_holding,
    _Trainer,
)
from vrl.trainers.checkpointing import (
    CHECKPOINT_META_NAME,
    CHECKPOINT_SCHEMA_VERSION,
    LORA_WEIGHTS_NAME,
    TRAINING_CHECKPOINT_NAME,
    AdapterExport,
    TrainingCheckpoint,
    export_checkpoint_state,
    load_checkpoint_state,
    save_training_checkpoint,
)
from vrl.trainers.distributed import DistributedTrainingContext


class _RecordingStrategy:
    """The checkpoint seams of a strategy on one rank of a ``world_size`` run.

    One process cannot produce a peer's rank-agreement votes, so this double
    records every vote and turns the 1-based votes in ``fail_agreements`` into a
    peer failure. It records every bundle it exports, raises on export call
    ``fail_export_at``, and returns ``exported_state`` instead of the real export
    when given. The real sequences run on gloo in
    ``tests/trainers/test_fsdp_gather_distributed.py``.
    """

    def __init__(
        self,
        *,
        rank: int = 0,
        world_size: int = 1,
        fail_agreements: tuple[int, ...] = (),
        fail_export_at: int | None = None,
        exported_state: dict[str, Any] | None = None,
    ) -> None:
        # A real context: ``is_primary`` is derived from ``rank``, so a double can
        # no longer claim "not primary, world size 1", and ``single_process`` is
        # world 1 exactly as the resolver enforces.
        self.context = DistributedTrainingContext(
            strategy="single_process" if world_size == 1 else "fsdp",
            rank=rank,
            world_size=world_size,
            device=torch.device("cpu"),
        )
        self.collectives = SimpleNamespace(
            succeeded=self._agree,
            gather_rng_states=lambda state: [state] * world_size,
        )
        self.agreements: list[bool] = []
        self.exported: list[Any] = []
        self._fail_agreements = fail_agreements
        self._fail_export_at = fail_export_at
        self._exported_state = exported_state

    def export_checkpoint_state(self, bundle):
        self.exported.append(bundle)
        if len(self.exported) == self._fail_export_at:
            raise RuntimeError("checkpoint gather failed")
        state = self._exported_state or export_checkpoint_state(bundle)
        return state

    def _agree(self, succeeded: bool) -> bool:
        self.agreements.append(succeeded)
        return succeeded and len(self.agreements) not in self._fail_agreements


@pytest.mark.parametrize("provider", ["python_random", "numpy"])
@pytest.mark.parametrize("operation", ["capture", "restore"])
def test_rng_state_errors_are_not_silently_ignored(monkeypatch, provider, operation):
    import random

    import numpy as np

    from vrl.trainers.checkpointing import capture_rng_state, restore_rng_state

    module = random if provider == "python_random" else np.random
    getter, setter = (
        ("getstate", "setstate")
        if provider == "python_random"
        else (
            "get_state",
            "set_state",
        )
    )
    saved = getattr(module, getter)()
    failure = RuntimeError("RNG state operation failed")

    def fail(*args):
        raise failure

    monkeypatch.setattr(module, getter if operation == "capture" else setter, fail)
    with pytest.raises(RuntimeError) as caught:
        if operation == "capture":
            capture_rng_state()
        else:
            restore_rng_state({provider: saved})
    assert caught.value is failure


@pytest.mark.parametrize("missing_module", ["numpy", "numpy._core"])
def test_rng_capture_only_tolerates_absent_numpy(monkeypatch, missing_module):
    import builtins

    from vrl.trainers.checkpointing import capture_rng_state, restore_rng_state

    original_import = builtins.__import__

    def import_without_numpy(name, *args, **kwargs):
        if name == "numpy":
            raise ModuleNotFoundError("unavailable dependency", name=missing_module)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_numpy)
    if missing_module == "numpy":
        assert "numpy" not in capture_rng_state()
    else:
        with pytest.raises(ModuleNotFoundError):
            capture_rng_state()
    # A saved NumPy state requires NumPy, even if capture can omit absent NumPy.
    with pytest.raises(ModuleNotFoundError):
        restore_rng_state({"numpy": ()})


def test_training_checkpoint_round_trips_trainer_and_owned_state(tmp_path) -> None:
    """Save then restore round-trips trainer progress (step, global_step) and checkpoint-owned
    module state; ``next_epoch`` comes from the saved progress.
    """
    trainer = _Trainer()
    source = _Bundle()
    with torch.no_grad():
        source.module.weight.fill_(3.0)
    save_training_checkpoint(
        tmp_path / "checkpoint-2",
        trainer=trainer,
        bundle=source,
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 2, "global_step": 5},
        rng_state={},
    )

    checkpoint = TrainingCheckpoint.load(tmp_path / "checkpoint-2")
    restored = _Bundle()
    with torch.no_grad():
        restored.module.weight.fill_(0.0)
    checkpoint.restore_training(
        trainer=trainer,
        bundle=restored,
        family="unit",
        expected_model_identity=UNIT_IDENTITY,
    )

    assert (tmp_path / "checkpoint-2" / TRAINING_CHECKPOINT_NAME).exists()
    assert checkpoint.next_epoch == 2
    assert trainer.loaded == {"step": 2, "global_step": 5}
    assert restored.module.weight.item() == pytest.approx(3.0)


def test_restore_model_checkpoint_restores_without_trainer_state(tmp_path) -> None:
    source = _Bundle()
    with torch.no_grad():
        source.module.weight.fill_(4.0)
    path = tmp_path / "checkpoint-model-only"
    save_training_checkpoint(
        path,
        trainer=_Trainer(),
        bundle=source,
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
    )
    restored = _Bundle()
    with torch.no_grad():
        restored.module.weight.zero_()

    TrainingCheckpoint.load(path).restore_model(
        bundle=restored,
        family="unit",
        expected_model_identity=UNIT_IDENTITY,
    )

    assert restored.module.weight.item() == pytest.approx(4.0)


def test_restore_model_checkpoint_rejects_identity_before_loading(tmp_path) -> None:
    source = _Bundle()
    path = tmp_path / "checkpoint-model-only"
    save_training_checkpoint(
        path,
        trainer=_Trainer(),
        bundle=source,
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
    )
    restored = _Bundle()
    before = restored.module.weight.detach().clone()

    with pytest.raises(ValueError, match="model identity mismatch"):
        TrainingCheckpoint.load(path).restore_model(
            bundle=restored,
            family="unit",
            expected_model_identity={"schema": "wrong/v1"},
        )

    assert torch.equal(restored.module.weight, before)


@pytest.mark.parametrize(
    ("family", "identity", "message"),
    [
        (
            "flux",
            {"schema": "vrl.model-identity/v1", "sources": {}, "build": {}},
            "family mismatch",
        ),
        (
            "sana",
            {"schema": "vrl.model-identity/v1", "sources": {"base": {}}, "build": {}},
            "model identity mismatch",
        ),
        (
            " sana",
            {"schema": "vrl.model-identity/v1", "sources": {}, "build": {}},
            "runtime family must be a non-empty trimmed string",
        ),
    ],
)
def test_checkpoint_compatibility_preflight_rejects_strict_mismatch(
    tmp_path,
    family,
    identity,
    message,
) -> None:
    checkpoint = TrainingCheckpoint(
        checkpoint_dir=tmp_path,
        checkpoint_path=tmp_path / TRAINING_CHECKPOINT_NAME,
        payload={
            "family": "sana",
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model": {
                "identity": {
                    "schema": "vrl.model-identity/v1",
                    "sources": {},
                    "build": {},
                },
                "owned_state": {},
            },
        },
        meta={},
    )

    with pytest.raises(ValueError, match=message):
        checkpoint.validate_compatibility(
            family=family,
            expected_model_identity=identity,
        )


def test_save_training_checkpoint_prefers_primary_only_snapshot_seams(tmp_path) -> None:
    """Checkpoint-owned state and trainer state come from their primary-only seams.

    The strategy returns weights the bundle does not hold, so a match proves the
    strategy export -- not ``export_checkpoint_state(bundle)`` -- was used.
    """

    class _CheckpointTrainer(_Trainer):
        def checkpoint_state_dict(self):
            return {"step": 3, "global_step": 8}

        def state_dict(self):
            raise AssertionError("generic trainer state used for checkpoint")

    strategy = _RecordingStrategy(exported_state={"module": {"weight": torch.full((1, 1), 7.0)}})
    bundle = _Bundle()
    save_training_checkpoint(
        tmp_path / "checkpoint-primary-only",
        trainer=_CheckpointTrainer(),
        bundle=bundle,
        family="unit",
        model_identity={"schema": "unit-model/v1"},
        progress={"next_epoch": 3},
        rng_state={},
        strategy=strategy,
    )

    checkpoint = TrainingCheckpoint.load(tmp_path / "checkpoint-primary-only")
    assert strategy.exported == [bundle]
    assert checkpoint.trainer_state == {"step": 3, "global_step": 8}
    assert checkpoint.checkpoint_state["module"]["weight"].item() == pytest.approx(7.0)


def test_save_training_checkpoint_non_primary_gathers_but_writes_nothing(tmp_path) -> None:
    """Multi-rank contract: the checkpoint-state export (a COLLECTIVE under FSDP2)
    runs on every rank, but only the primary rank writes files.

    Gating the whole save to rank0 would deadlock FSDP — rank0 waits at the
    all-gather for peers that never call it. A non-primary strategy context MUST
    still invoke the strategy export (joining the collective) yet create no
    checkpoint directory/files. The spy proves the export ran; the empty tmp_path
    proves nothing was written."""

    # rank1 of 2: "not primary" only exists in a multi-rank world, and a
    # multi-rank strategy owes the checkpoint a rank-agreement seam.
    strategy = _RecordingStrategy(rank=1, world_size=2)
    bundle = _Bundle()
    ckpt_dir = tmp_path / "checkpoint-nonprimary"
    out = save_training_checkpoint(
        ckpt_dir,
        trainer=_Trainer(),
        bundle=bundle,
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
        strategy=strategy,
    )

    assert strategy.exported == [bundle]  # joined the collective gather
    assert out == {}  # no meta returned
    assert not ckpt_dir.exists()  # rank0-only IO: nothing written on a non-primary rank


def test_training_checkpoint_restores_raw_weights_when_ema_gather_fails(tmp_path) -> None:
    class _ExportModule(nn.Linear):
        def save_pretrained(self, *_args, **_kwargs):
            raise AssertionError("failed gather must not reach artifact IO")

    module = _ExportModule(1, 1, bias=False)
    bundle = _Bundle(module)
    ema = _ema_holding(module, average=7.0, live=3.0)
    # The second gather is the EMA artifact's.
    strategy = _RecordingStrategy(fail_export_at=2)

    with pytest.raises(RuntimeError, match="checkpoint gather failed"):
        save_training_checkpoint(
            tmp_path / "checkpoint-failed-ema-gather",
            trainer=_Trainer(),
            bundle=bundle,
            family="unit",
            model_identity=UNIT_IDENTITY,
            progress={"next_epoch": 1},
            rng_state={},
            adapter_exports={LORA_WEIGHTS_NAME: AdapterExport(module)},
            export_ema=ema,
            strategy=strategy,
        )

    assert len(strategy.exported) == 2
    # The real wrapper's own copy_temp_to put the raw weights back, and dropped
    # its snapshot while doing so.
    assert module.weight.item() == pytest.approx(3.0)
    assert ema.temp_stored_parameters is None
    assert not (tmp_path / "checkpoint-failed-ema-gather").exists()


def test_training_checkpoint_preserves_swap_failure_without_unset_restore(
    monkeypatch,
    tmp_path,
) -> None:
    class _ExportModule(nn.Linear):
        def save_pretrained(self, *_args, **_kwargs):
            raise AssertionError("failed swap must not reach artifact IO")

    module = _ExportModule(1, 1, bias=False)
    # Real EMA, one injected fault: the real wrapper always snapshots before it
    # swaps, so "the swap failed before establishing a snapshot" only exists as an
    # injected fault. Only copy_ema_to is patched — leaving the real copy_temp_to
    # in place is what makes the assertion below bite: production calling it with
    # no snapshot would raise "snapshot is not available" instead of the swap error.
    ema = _ema_holding(module, average=7.0, live=3.0)

    def failing_copy_ema_to(_parameters, *, store_temp=True):
        raise RuntimeError("EMA swap failed")

    monkeypatch.setattr(ema, "copy_ema_to", failing_copy_ema_to)

    with pytest.raises(RuntimeError, match="EMA swap failed"):
        save_training_checkpoint(
            tmp_path / "checkpoint-failed-ema-swap",
            trainer=_Trainer(),
            bundle=_Bundle(module),
            family="unit",
            model_identity=UNIT_IDENTITY,
            progress={"next_epoch": 1},
            rng_state={},
            adapter_exports={LORA_WEIGHTS_NAME: AdapterExport(module)},
            export_ema=ema,
        )

    assert module.weight.item() == pytest.approx(3.0)
    assert ema.temp_stored_parameters is None
    assert not (tmp_path / "checkpoint-failed-ema-swap").exists()


def test_mixed_rank_ema_update_state_fails_before_swap_or_second_export(tmp_path) -> None:
    class _ExportModule(nn.Linear):
        def save_pretrained(self, *_args, **_kwargs):
            raise AssertionError("mixed EMA state must fail before artifact IO")

    module = _ExportModule(1, 1, bias=False)
    # Votes two and three are "every rank exports EMA" / "every rank skips it":
    # a peer that disagrees turns both False.
    strategy = _RecordingStrategy(world_size=2, fail_agreements=(2, 3))
    ema = _ema_holding(module, average=7.0, live=3.0)

    with pytest.raises(RuntimeError, match="disagree on EMA export decision"):
        save_training_checkpoint(
            tmp_path / "checkpoint-mixed-ema-state",
            trainer=_Trainer(),
            bundle=_Bundle(module),
            family="unit",
            model_identity=UNIT_IDENTITY,
            progress={"next_epoch": 1},
            rng_state={},
            adapter_exports={LORA_WEIGHTS_NAME: AdapterExport(module)},
            export_ema=ema,
            strategy=strategy,
        )

    assert strategy.exported == []
    # No swap was attempted: the sentinel assertions the previous stub raised are
    # now the real wrapper's own state — weights untouched, no snapshot taken.
    assert module.weight.item() == pytest.approx(3.0)
    assert ema.temp_stored_parameters is None
    assert not (tmp_path / "checkpoint-mixed-ema-state").exists()


def test_load_training_checkpoint_requires_checkpoint_pt(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()

    with pytest.raises(FileNotFoundError, match=TRAINING_CHECKPOINT_NAME):
        TrainingCheckpoint.load(ckpt)


def test_load_training_checkpoint_rejects_bad_schema(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()
    torch.save({"schema_version": 999}, ckpt / TRAINING_CHECKPOINT_NAME)

    with pytest.raises(ValueError, match="schema_version"):
        TrainingCheckpoint.load(ckpt)


@pytest.mark.parametrize(
    ("payload", "error", "message"),
    [
        pytest.param({}, ValueError, "missing schema_version", id="missing"),
        pytest.param(
            {"schema_version": "2"}, TypeError, "schema_version must be an integer", id="string"
        ),
    ],
)
def test_load_training_checkpoint_rejects_non_integer_schema_version(
    tmp_path,
    payload,
    error,
    message,
) -> None:
    ckpt = tmp_path / "checkpoint-invalid-schema"
    ckpt.mkdir()
    torch.save(payload, ckpt / TRAINING_CHECKPOINT_NAME)

    with pytest.raises(error, match=message):
        TrainingCheckpoint.load(ckpt)


def test_save_training_checkpoint_requires_non_empty_identity(tmp_path) -> None:
    with pytest.raises(ValueError, match="model_identity"):
        save_training_checkpoint(
            tmp_path / "checkpoint-no-identity",
            trainer=_Trainer(),
            bundle=_Bundle(),
            family="unit",
            progress={"next_epoch": 1},
            model_identity={},
        )


@pytest.mark.parametrize("family", ["", " unit "])
def test_save_training_checkpoint_requires_canonical_family(tmp_path, family) -> None:
    with pytest.raises(ValueError, match="family"):
        save_training_checkpoint(
            tmp_path / "checkpoint-no-family",
            trainer=_Trainer(),
            bundle=_Bundle(),
            family=family,
            progress={"next_epoch": 1},
            model_identity=UNIT_IDENTITY,
        )
    assert not (tmp_path / "checkpoint-no-family").exists()


def test_load_checkpoint_state_strict_rejects_key_mismatch() -> None:
    with pytest.raises(ValueError, match="missing"):
        load_checkpoint_state(_Bundle(), {})


@pytest.mark.parametrize("field", ["next_epoch", "next_step"])
def test_resume_position_requires_explicit_progress(tmp_path, field) -> None:
    # Neither the directory name nor the metadata sidecar may stand in for progress.
    checkpoint = TrainingCheckpoint(
        tmp_path / "checkpoint-42",
        tmp_path / "checkpoint-42" / TRAINING_CHECKPOINT_NAME,
        {"trainer": {"step": 12, "global_step": 24}},
        {"next_epoch": 42},
    )
    with pytest.raises(ValueError, match=f"required progress.{field}"):
        getattr(checkpoint, field)


def test_resume_positions_use_saved_progress(tmp_path) -> None:
    checkpoint = TrainingCheckpoint(
        tmp_path / "checkpoint-42",
        tmp_path / TRAINING_CHECKPOINT_NAME,
        {"progress": {"next_epoch": 3, "next_step": 17}, "trainer": {"step": 12}},
        {"next_epoch": 42},
    )
    assert checkpoint.next_epoch == 3
    assert checkpoint.next_step == 17


def test_load_training_checkpoint_rejects_non_object_meta(tmp_path) -> None:
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
        TrainingCheckpoint.load(ckpt)


def test_checkpoint_publish_is_atomic(tmp_path) -> None:
    """A save that dies mid-write leaves no directory a resume would trust."""
    from vrl.trainers.checkpointing import (
        find_latest_complete_checkpoint,
        is_complete_checkpoint,
    )

    class _ExplodingExport(nn.Linear):
        def save_pretrained(self, *_args, **_kwargs):
            raise RuntimeError("disk full")

    export_module = _ExplodingExport(1, 1, bias=False)
    target = tmp_path / "checkpoint-3"
    with pytest.raises(RuntimeError, match="disk full"):
        save_training_checkpoint(
            target,
            trainer=_Trainer(),
            bundle=_Bundle(export_module),
            family="unit",
            model_identity=UNIT_IDENTITY,
            progress={"next_epoch": 3, "global_step": 3},
            rng_state={},
            adapter_exports={"lora_weights": AdapterExport(export_module)},
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
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 7, "global_step": 7},
        rng_state={},
    )

    assert is_complete_checkpoint(target)
    meta = read_checkpoint_meta(target)
    assert meta["checkpoint_file_bytes"] == (target / TRAINING_CHECKPOINT_NAME).stat().st_size
    assert meta["model_identity"] == UNIT_IDENTITY
    # Truncation is detected without loading the payload.
    (target / TRAINING_CHECKPOINT_NAME).write_bytes(b"torn")
    assert not is_complete_checkpoint(target)


def test_checkpoint_load_rejects_meta_identity_drift(tmp_path) -> None:
    target = tmp_path / "checkpoint-identity-drift"
    save_training_checkpoint(
        target,
        trainer=_Trainer(),
        bundle=_Bundle(),
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
    )
    meta_path = target / CHECKPOINT_META_NAME
    meta = json.loads(meta_path.read_text())
    meta["model_identity"] = {"schema": "stale/v1"}
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="model_identity disagrees"):
        TrainingCheckpoint.load(target)


def test_checkpoint_load_rejects_meta_without_schema_version(tmp_path) -> None:
    target = tmp_path / "checkpoint-meta-no-schema"
    save_training_checkpoint(
        target,
        trainer=_Trainer(),
        bundle=_Bundle(),
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
    )
    meta_path = target / CHECKPOINT_META_NAME
    meta = json.loads(meta_path.read_text())
    meta.pop("schema_version")
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="missing schema_version"):
        TrainingCheckpoint.load(target)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda meta: meta.pop("family"), "family"),
        (lambda meta: meta.__setitem__("family", "other"), "family disagrees"),
    ],
)
def test_checkpoint_load_requires_meta_family_to_match_payload(
    tmp_path,
    mutate,
    message,
) -> None:
    target = tmp_path / "checkpoint-meta-family"
    save_training_checkpoint(
        target,
        trainer=_Trainer(),
        bundle=_Bundle(),
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
    )
    meta_path = target / CHECKPOINT_META_NAME
    meta = json.loads(meta_path.read_text())
    mutate(meta)
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(ValueError, match=message):
        TrainingCheckpoint.load(target)


@pytest.mark.parametrize("size", [True, -1])
def test_checkpoint_completeness_rejects_invalid_recorded_size(tmp_path, size) -> None:
    from vrl.trainers.checkpointing import is_complete_checkpoint

    (tmp_path / TRAINING_CHECKPOINT_NAME).write_bytes(b"x")
    (tmp_path / CHECKPOINT_META_NAME).write_text(
        json.dumps({"checkpoint_file_bytes": size}), encoding="utf-8"
    )
    assert not is_complete_checkpoint(tmp_path)


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
            model_identity=UNIT_IDENTITY,
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
    bundle = _Bundle()
    for value in (3.0, 7.0):
        with torch.no_grad():
            bundle.module.weight.fill_(value)
        save_training_checkpoint(
            target,
            trainer=_Trainer(),
            bundle=bundle,
            family="unit",
            model_identity=UNIT_IDENTITY,
            progress={"next_epoch": 1, "global_step": 1},
            rng_state={},
        )
    assert is_complete_checkpoint(target)
    saved = TrainingCheckpoint.load(target)
    assert saved.payload["family"] == "unit"
    assert saved.checkpoint_state["module"]["weight"].item() == 7.0


@pytest.mark.gpu
def test_require_equal_tensor_tree_bridges_devices() -> None:
    """Restored optimizer tensors move to CUDA while the checkpoint stays on CPU."""
    from vrl.trainers.checkpointing import _require_equal_tensor_tree

    if not torch.cuda.is_available():
        pytest.skip("cross-device case needs CUDA")
    cpu_tree = {"optimizer": {"exp_avg": torch.ones(3)}}
    cuda_tree = {"optimizer": {"exp_avg": torch.ones(3, device="cuda")}}
    _require_equal_tensor_tree(cpu_tree, cuda_tree, label="cross device")
    with pytest.raises(ValueError, match="tensor mismatch"):
        _require_equal_tensor_tree(
            {"optimizer": {"exp_avg": torch.zeros(3)}},
            cuda_tree,
            label="cross device mismatch",
        )


@pytest.mark.parametrize(("field", "value"), [("next_epoch", True), ("next_step", -1)])
def test_resume_position_rejects_invalid_values(tmp_path, field, value):
    checkpoint = TrainingCheckpoint(
        tmp_path,
        tmp_path / TRAINING_CHECKPOINT_NAME,
        {"progress": {field: value}},
        {},
    )
    with pytest.raises(ValueError, match="non-negative integer"):
        getattr(checkpoint, field)


def test_checkpoint_metadata_does_not_infer_progress_fields(tmp_path):
    from vrl.trainers.checkpointing import read_checkpoint_meta

    target = tmp_path / "checkpoint-999"
    save_training_checkpoint(
        target,
        trainer=_Trainer(),
        bundle=_Bundle(),
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 7},
        rng_state={},
    )
    meta = read_checkpoint_meta(target)
    assert meta["next_epoch"] == 7
    for field in ("completed_epoch", "completed_step", "next_step", "global_step"):
        assert field not in meta


def test_latest_checkpoint_does_not_guess_missing_step_from_name(tmp_path):
    from vrl.trainers.checkpointing import find_latest_complete_checkpoint

    save_training_checkpoint(
        tmp_path / "checkpoint-999",
        trainer=_Trainer(),
        bundle=_Bundle(),
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 7},
        rng_state={},
    )
    with pytest.raises(ValueError, match="requires a non-negative integer global_step"):
        find_latest_complete_checkpoint(tmp_path)
