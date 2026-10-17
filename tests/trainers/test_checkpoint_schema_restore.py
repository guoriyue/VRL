from __future__ import annotations

import pytest
import torch
from torch import nn

from tests.trainers._checkpoint_helpers import UNIT_IDENTITY, _Trainer
from vrl.trainers.checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    TRAINING_CHECKPOINT_NAME,
    TrainingCheckpoint,
    export_checkpoint_state,
)


class _OwnedModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.frozen_base = nn.Parameter(torch.tensor([2.0]), requires_grad=False)
        self.previous = nn.Parameter(torch.tensor([3.0]))


class _OwnedBundle:
    def __init__(self) -> None:
        self.module = _OwnedModule()
        self.trainable_modules = {"module": self.module}


def _training_checkpoint(tmp_path, payload) -> TrainingCheckpoint:
    return TrainingCheckpoint(
        checkpoint_dir=tmp_path,
        checkpoint_path=tmp_path / TRAINING_CHECKPOINT_NAME,
        payload=payload,
        meta={},
    )


def _v1_payload(
    state: dict[str, torch.Tensor],
    *,
    identity: dict | None = None,
) -> dict:
    model = {"trainable_modules": {"module": state}}
    if identity is not None:
        model["identity"] = identity
    return {
        "schema_version": 1,
        "family": "unit",
        "trainer": {"step": 2, "global_step": 5},
        "model": model,
        "progress": {"next_epoch": 2},
        "rng": {},
    }


@pytest.mark.parametrize("zip_format", [True, False])
def test_checkpoint_mapping_preserves_file_and_legacy_load(tmp_path, monkeypatch, zip_format):
    path = tmp_path / TRAINING_CHECKPOINT_NAME
    torch.save(
        _v1_payload({"weight": torch.arange(8.0)}),
        path,
        _use_new_zipfile_serialization=zip_format,
    )
    original = torch.load
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs.get("mmap"))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    checkpoint = TrainingCheckpoint.load(path)
    checkpoint.payload["model"]["trainable_modules"]["module"]["weight"].add_(10)
    again = TrainingCheckpoint.load(path)
    torch.testing.assert_close(
        again.payload["model"]["trainable_modules"]["module"]["weight"], torch.arange(8.0)
    )
    assert calls == [zip_format, zip_format]


@pytest.mark.parametrize("primary", [True, False])
@pytest.mark.parametrize("matches", [True, False])
def test_strict_resume_verifies_primary_optimizer_snapshot(
    tmp_path, monkeypatch, primary, matches
):
    from types import SimpleNamespace

    payload = _v1_payload({"weight": torch.ones(1)})
    payload["trainer"]["optimizer"] = {"moment": torch.ones(2)}
    checkpoint = _training_checkpoint(tmp_path, payload)
    monkeypatch.setattr(TrainingCheckpoint, "restore_model", lambda *a, **k: None)
    agreements = []

    def agree(success):
        agreements.append(success)
        return success and matches

    class Trainer(_Trainer):
        _strategy = SimpleNamespace(
            context=SimpleNamespace(is_primary=primary),
            collectives=SimpleNamespace(succeeded=agree),
        )

        def state_dict(self):
            raise AssertionError("must not replicate optimizer state on every rank")

        def checkpoint_state_dict(self):
            return {
                "optimizer": {"moment": torch.ones(2) * (1 if matches else 2)} if primary else {}
            }

    if matches:
        checkpoint.restore_training(trainer=Trainer(), bundle=None, family="unit")
    else:
        # The primary's own comparison fails; a peer learns it from the vote.
        error, message = (
            (ValueError, "restored optimizer state")
            if primary
            else (RuntimeError, "restored optimizer state verification failed on the primary")
        )
        with pytest.raises(error, match=message):
            checkpoint.restore_training(trainer=Trainer(), bundle=None, family="unit")
    assert agreements == [matches if primary else True]


def test_load_training_checkpoint_rejects_schema_v2_without_identity(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint-no-identity"
    ckpt.mkdir()
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "family": "unit",
            "trainer": {},
            "model": {"owned_state": {}},
            "progress": {},
            "rng": {},
        },
        ckpt / TRAINING_CHECKPOINT_NAME,
    )

    with pytest.raises(ValueError, match="model keys mismatch"):
        TrainingCheckpoint.load(ckpt)


def test_load_training_checkpoint_rejects_schema_v2_without_family(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint-no-family"
    ckpt.mkdir()
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "family": "",
            "trainer": {},
            "model": {"identity": UNIT_IDENTITY, "owned_state": {}},
            "progress": {},
            "rng": {},
        },
        ckpt / TRAINING_CHECKPOINT_NAME,
    )

    with pytest.raises(ValueError, match="family"):
        TrainingCheckpoint.load(ckpt)


def test_checkpoint_export_is_exact_owned_clone_and_unwraps_compile() -> None:
    bundle = _OwnedBundle()
    compiled = torch.compile(bundle.module)
    bundle.trainable_modules["module"] = compiled

    snapshot = export_checkpoint_state(bundle)["module"]

    assert set(snapshot) == {"weight", "previous"}
    assert snapshot["weight"].data_ptr() != bundle.module.weight.data_ptr()
    assert snapshot["previous"].data_ptr() != bundle.module.previous.data_ptr()
    assert all("_orig_mod" not in name for name in snapshot)
    before = {name: value.clone() for name, value in snapshot.items()}
    with torch.no_grad():
        bundle.module.weight.add_(10)
        bundle.module.previous.add_(10)
    assert all(torch.equal(snapshot[name], before[name]) for name in snapshot)


def test_checkpoint_export_rejects_module_without_owned_state() -> None:
    module = nn.Linear(1, 1)
    module.requires_grad_(False)
    bundle = type("_FrozenBundle", (), {"trainable_modules": {"module": module}})()

    with pytest.raises(ValueError, match="no checkpoint-owned state"):
        export_checkpoint_state(bundle)


def test_strict_schema_v2_restore_requires_exact_owned_keys(tmp_path) -> None:
    bundle = _OwnedBundle()
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "family": "unit",
        "trainer": {"step": 2, "global_step": 5},
        "model": {
            "identity": UNIT_IDENTITY,
            "owned_state": {
                "module": {
                    "weight": torch.tensor([7.0], dtype=torch.float64),
                    "previous": torch.tensor([8.0], dtype=torch.float64),
                },
            },
        },
        "progress": {"next_epoch": 2},
        "rng": {},
    }

    _training_checkpoint(tmp_path, payload).restore_training(
        trainer=_Trainer(),
        bundle=bundle,
        family="unit",
        expected_model_identity=UNIT_IDENTITY,
    )

    assert bundle.module.weight.item() == pytest.approx(7.0)
    assert bundle.module.previous.item() == pytest.approx(8.0)
    assert bundle.module.frozen_base.item() == pytest.approx(2.0)
    assert bundle.module.weight.dtype == torch.float32
    assert bundle.module.previous.dtype == torch.float32


def test_schema_v2_wrong_shape_rejects_all_roots_before_mutation(tmp_path) -> None:
    bundle = _OwnedBundle()
    bundle.second = _OwnedModule()
    bundle.trainable_modules["second"] = bundle.second
    before = {
        root_name: {name: value.clone() for name, value in module.state_dict().items()}
        for root_name, module in bundle.trainable_modules.items()
    }
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "family": "unit",
        "trainer": {},
        "model": {
            "identity": UNIT_IDENTITY,
            "owned_state": {
                "module": {
                    "weight": torch.tensor([7.0]),
                    "previous": torch.tensor([8.0]),
                },
                "second": {
                    "weight": torch.tensor([9.0]),
                    "previous": torch.tensor([10.0, 11.0]),
                },
            },
        },
        "progress": {},
        "rng": {},
    }

    with pytest.raises(ValueError, match="shape mismatch"):
        _training_checkpoint(tmp_path, payload).restore_training(
            trainer=_Trainer(),
            bundle=bundle,
            family="unit",
            expected_model_identity=UNIT_IDENTITY,
        )

    assert all(
        torch.equal(value, before[root_name][name])
        for root_name, module in bundle.trainable_modules.items()
        for name, value in module.state_dict().items()
    )


def test_schema_v2_non_tensor_owned_value_rejects_before_mutation(tmp_path) -> None:
    bundle = _OwnedBundle()
    before = {name: value.clone() for name, value in bundle.module.state_dict().items()}
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "family": "unit",
        "trainer": {},
        "model": {
            "identity": UNIT_IDENTITY,
            "owned_state": {
                "module": {
                    "weight": torch.tensor([7.0]),
                    "previous": 8.0,
                },
            },
        },
        "progress": {},
        "rng": {},
    }

    with pytest.raises(TypeError, match="must be a tensor"):
        _training_checkpoint(tmp_path, payload).restore_training(
            trainer=_Trainer(),
            bundle=bundle,
            family="unit",
            expected_model_identity=UNIT_IDENTITY,
        )

    assert all(
        torch.equal(value, before[name]) for name, value in bundle.module.state_dict().items()
    )


@pytest.mark.parametrize(
    ("owned_state", "message"),
    [
        (
            {"module": {"weight": torch.tensor([7.0])}},
            r"keys mismatch for 'module': missing=\['previous'\]",
        ),
        (
            {
                "module": {
                    "weight": torch.tensor([7.0]),
                    "previous": torch.tensor([8.0]),
                    "frozen_base": torch.tensor([9.0]),
                },
            },
            r"keys mismatch for 'module': missing=\[\], unexpected=\['frozen_base'\]",
        ),
        (
            {
                "module": {
                    "weight": torch.tensor([7.0]),
                    "previous": torch.tensor([8.0]),
                },
                "extra": {},
            },
            r"roots mismatch: missing=\[\], unexpected=\['extra'\]",
        ),
    ],
)
def test_strict_schema_v2_restore_rejects_missing_extra_keys_and_roots(
    tmp_path,
    owned_state,
    message,
) -> None:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "family": "unit",
        "trainer": {"step": 2, "global_step": 5},
        "model": {"identity": UNIT_IDENTITY, "owned_state": owned_state},
        "progress": {},
        "rng": {},
    }

    with pytest.raises(ValueError, match=message):
        _training_checkpoint(tmp_path, payload).restore_training(
            trainer=_Trainer(),
            bundle=_OwnedBundle(),
            family="unit",
            expected_model_identity=UNIT_IDENTITY,
        )


def test_strict_schema_v1_full_state_restores_without_identity(tmp_path) -> None:
    source = _OwnedBundle()
    with torch.no_grad():
        source.module.weight.fill_(7.0)
        source.module.frozen_base.fill_(9.0)
        source.module.previous.fill_(8.0)
    restored = _OwnedBundle()

    _training_checkpoint(tmp_path, _v1_payload(source.module.state_dict())).restore_training(
        trainer=_Trainer(),
        bundle=restored,
        family="unit",
    )

    assert restored.module.weight.item() == pytest.approx(7.0)
    assert restored.module.previous.item() == pytest.approx(8.0)
    assert restored.module.frozen_base.item() == pytest.approx(9.0)


def test_strict_schema_v1_compiled_full_state_normalizes_legacy_prefix(tmp_path) -> None:
    source = _OwnedBundle()
    with torch.no_grad():
        source.module.frozen_base.fill_(9.0)
    legacy_compiled = dict(torch.compile(source.module).state_dict())
    restored = _OwnedBundle()

    _training_checkpoint(tmp_path, _v1_payload(legacy_compiled)).restore_training(
        trainer=_Trainer(),
        bundle=restored,
        family="unit",
    )

    assert torch.equal(restored.module.weight, source.module.weight)
    assert torch.equal(restored.module.previous, source.module.previous)
    assert torch.equal(restored.module.frozen_base, source.module.frozen_base)


def test_strict_schema_v1_rejects_mixed_compile_prefixes(tmp_path) -> None:
    mixed = {
        "_orig_mod.weight": torch.tensor([7.0]),
        "frozen_base": torch.tensor([2.0]),
        "_orig_mod.previous": torch.tensor([8.0]),
    }

    with pytest.raises(ValueError, match="mixes compiled and uncompiled"):
        _training_checkpoint(tmp_path, _v1_payload(mixed)).restore_training(
            trainer=_Trainer(),
            bundle=_OwnedBundle(),
            family="unit",
        )


def test_strict_schema_v1_selective_state_requires_verified_identity(tmp_path) -> None:
    selective = {
        "weight": torch.tensor([7.0]),
        "previous": torch.tensor([8.0]),
    }

    with pytest.raises(ValueError, match="verified model identity"):
        _training_checkpoint(tmp_path, _v1_payload(selective)).restore_training(
            trainer=_Trainer(),
            bundle=_OwnedBundle(),
            family="unit",
        )


def test_strict_schema_v1_selective_state_rejects_missing_owned_state(tmp_path) -> None:
    payload = _v1_payload(
        {"weight": torch.tensor([7.0])},
        identity=UNIT_IDENTITY,
    )

    with pytest.raises(ValueError, match=r"missing=.*previous"):
        _training_checkpoint(tmp_path, payload).restore_training(
            trainer=_Trainer(),
            bundle=_OwnedBundle(),
            family="unit",
            expected_model_identity=UNIT_IDENTITY,
        )
