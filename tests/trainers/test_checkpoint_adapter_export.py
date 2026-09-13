from __future__ import annotations

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from tests.trainers._checkpoint_helpers import (
    UNIT_IDENTITY,
    _Bundle,
    _DenoisePolicy,
    _ema_holding,
    _export_bundle,
    _exported_adapter_weight,
    _PublishableModule,
    _TokenPolicy,
    _Trainer,
)
from vrl.trainers.checkpointing import (
    CHECKPOINT_META_NAME,
    LORA_WEIGHTS_NAME,
    TRAINING_CHECKPOINT_NAME,
    AdapterExport,
    TrainingCheckpoint,
    build_adapter_exports,
    save_training_checkpoint,
)


def test_training_checkpoint_writes_optional_lora_export(tmp_path) -> None:
    """The adapter artifact lands beside the resume checkpoint and is really loadable.

    ``save_pretrained`` stays a stand-in (real PEFT also writes an adapter config,
    which belongs to the PEFT-export tests), but the *file* it produces is a real
    safetensors container, so the assertion can read the tensor back instead of
    settling for "a file with that name exists".
    """

    class _ExportModule(nn.Linear):
        def save_pretrained(self, path, *, state_dict, selected_adapters):
            assert state_dict.keys() == {"weight"}
            assert selected_adapters == ["default"]
            path.mkdir(parents=True)
            save_file(dict(state_dict), path / "adapter_model.safetensors")

    export_module = _ExportModule(1, 1, bias=False)
    with torch.no_grad():
        export_module.weight.fill_(4.0)
    save_training_checkpoint(
        tmp_path / "checkpoint-1",
        trainer=_Trainer(),
        bundle=_Bundle(export_module),
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
        adapter_exports={LORA_WEIGHTS_NAME: AdapterExport(export_module)},
    )

    assert (tmp_path / "checkpoint-1" / TRAINING_CHECKPOINT_NAME).exists()
    assert (tmp_path / "checkpoint-1" / CHECKPOINT_META_NAME).exists()
    exported = _exported_adapter_weight(
        tmp_path / "checkpoint-1" / LORA_WEIGHTS_NAME / "adapter_model.safetensors",
    )
    assert torch.equal(exported, torch.full((1, 1), 4.0))


def test_adapter_export_accepts_safe_namespaced_path(tmp_path) -> None:
    class _ExportModule(nn.Linear):
        def save_pretrained(self, path, *, state_dict, selected_adapters):
            del selected_adapters
            path.mkdir(parents=True)
            save_file(dict(state_dict), path / "adapter_model.safetensors")

    module = _ExportModule(1, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(5.0)
    artifact_name = f"{LORA_WEIGHTS_NAME}/transformer"

    save_training_checkpoint(
        tmp_path / "checkpoint-namespaced",
        trainer=_Trainer(),
        bundle=_Bundle(module),
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
        adapter_exports={artifact_name: AdapterExport(module)},
    )

    exported = _exported_adapter_weight(
        tmp_path / "checkpoint-namespaced" / artifact_name / "adapter_model.safetensors",
    )
    assert torch.equal(exported, torch.full((1, 1), 5.0))


def test_adapter_export_accepts_safe_named_adapter(tmp_path) -> None:
    class _ExportModule(nn.Linear):
        def __init__(self) -> None:
            super().__init__(1, 1, bias=False)
            self.peft_config = {"publish": object()}

        def save_pretrained(self, path, *, state_dict, selected_adapters):
            assert state_dict.keys() == {"weight"}
            assert selected_adapters == ["publish"]
            path.mkdir(parents=True)
            save_file(dict(state_dict), path / "adapter_model.safetensors")

    module = _ExportModule()
    with torch.no_grad():
        module.weight.fill_(6.0)
    save_training_checkpoint(
        tmp_path / "checkpoint-named-adapter",
        trainer=_Trainer(),
        bundle=_Bundle(module),
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
        adapter_exports={
            LORA_WEIGHTS_NAME: AdapterExport(module, adapter_name="publish"),
        },
    )

    exported = _exported_adapter_weight(
        tmp_path / "checkpoint-named-adapter" / LORA_WEIGHTS_NAME / "adapter_model.safetensors",
    )
    assert torch.equal(exported, torch.full((1, 1), 6.0))


def test_build_adapter_exports_namespaces_multiple_denoise_roots() -> None:
    high = _PublishableModule()
    low = _PublishableModule()
    bundle = _export_bundle(_DenoisePolicy({"transformer": high, "transformer_2": low}))

    assert build_adapter_exports(bundle, use_lora=True) == {
        "lora_weights/transformer": AdapterExport(high),
        "lora_weights/transformer_2": AdapterExport(low),
    }


def test_build_adapter_exports_keeps_the_flat_path_for_a_single_root() -> None:
    transformer = _PublishableModule()
    bundle = _export_bundle(_DenoisePolicy({"transformer": transformer}))

    assert build_adapter_exports(bundle, use_lora=True) == {
        LORA_WEIGHTS_NAME: AdapterExport(transformer),
    }


def test_build_adapter_exports_drops_a_denoise_root_that_cannot_publish() -> None:
    """A full-finetune-shaped root is skipped, not turned into a failed export."""

    publishable = _PublishableModule()
    bundle = _export_bundle(
        _DenoisePolicy({"transformer": publishable, "plain": nn.Linear(1, 1)}),
    )

    assert build_adapter_exports(bundle, use_lora=True) == {
        LORA_WEIGHTS_NAME: AdapterExport(publishable),
    }


def test_build_adapter_exports_is_none_when_nothing_is_publishable() -> None:
    bundle = _export_bundle(_DenoisePolicy({"transformer": nn.Linear(1, 1)}))

    assert build_adapter_exports(bundle, use_lora=True) is None


def test_build_adapter_exports_is_none_for_a_full_finetune() -> None:
    bundle = _export_bundle(_DenoisePolicy({"transformer": _PublishableModule()}))

    assert build_adapter_exports(bundle, use_lora=False) is None


def test_build_adapter_exports_reaches_the_token_language_model() -> None:
    """The AR root is the wrapper, but the exported adapter is one hop inside it."""

    language_model = _PublishableModule()
    bundle = _export_bundle(_TokenPolicy(language_model))

    assert bundle.trainable_modules == {"model": bundle.model}
    assert build_adapter_exports(bundle, use_lora=True) == {
        LORA_WEIGHTS_NAME: AdapterExport(language_model),
    }


def test_build_adapter_exports_raises_for_an_unexportable_token_trunk() -> None:
    """Red line: the AR side must fail loudly, never publish silently nothing."""

    bundle = _export_bundle(_TokenPolicy(nn.Linear(1, 1)))

    with pytest.raises(TypeError, match="save_pretrained"):
        build_adapter_exports(bundle, use_lora=True)


@pytest.mark.parametrize(
    "adapter_name",
    ["", "a/b"],
)
def test_adapter_export_rejects_unsafe_adapter_name(adapter_name) -> None:
    class _ExportModule(nn.Linear):
        def __init__(self) -> None:
            super().__init__(1, 1, bias=False)
            self.peft_config = {adapter_name: object()}

        def save_pretrained(self, *_args, **_kwargs):
            raise AssertionError("unsafe adapter name must fail at construction")

    with pytest.raises(ValueError, match=r"trimmed|string|single path segment"):
        AdapterExport(_ExportModule(), adapter_name=adapter_name)


@pytest.mark.parametrize(
    "artifact_name",
    ["/absolute", "a/../b"],
)
def test_adapter_export_rejects_unsafe_output_path(tmp_path, artifact_name) -> None:
    class _ExportModule(nn.Linear):
        def save_pretrained(self, *_args, **_kwargs):
            raise AssertionError("unsafe artifact path must fail before IO")

    module = _ExportModule(1, 1, bias=False)

    with pytest.raises(ValueError, match="safe relative path"):
        save_training_checkpoint(
            tmp_path / "checkpoint-unsafe",
            trainer=_Trainer(),
            bundle=_Bundle(module),
            family="unit",
            model_identity=UNIT_IDENTITY,
            progress={"next_epoch": 1},
            rng_state={},
            adapter_exports={artifact_name: AdapterExport(module)},
        )

    assert not (tmp_path / "checkpoint-unsafe").exists()


def test_adapter_export_rejects_ancestor_output_paths(tmp_path) -> None:
    class _ExportModule(nn.Linear):
        def save_pretrained(self, *_args, **_kwargs):
            raise AssertionError("overlapping artifact paths must fail before IO")

    module = _ExportModule(1, 1, bias=False)

    with pytest.raises(ValueError, match=r"paths .* overlap"):
        save_training_checkpoint(
            tmp_path / "checkpoint-overlap",
            trainer=_Trainer(),
            bundle=_Bundle(module),
            family="unit",
            model_identity=UNIT_IDENTITY,
            progress={"next_epoch": 1},
            rng_state={},
            adapter_exports={
                "lora_weights": AdapterExport(module),
                "lora_weights/expert": AdapterExport(module),
            },
        )

    assert not (tmp_path / "checkpoint-overlap").exists()


def test_adapter_export_rejects_custom_adapter_effective_path_collision(tmp_path) -> None:
    class _ExportModule(nn.Linear):
        def __init__(self) -> None:
            super().__init__(1, 1, bias=False)
            self.peft_config = {"default": object(), "expert": object()}

        def save_pretrained(self, *_args, **_kwargs):
            raise AssertionError("colliding PEFT outputs must fail before IO")

    module = _ExportModule()

    with pytest.raises(ValueError, match="same PEFT output path"):
        save_training_checkpoint(
            tmp_path / "checkpoint-effective-collision",
            trainer=_Trainer(),
            bundle=_Bundle(module),
            family="unit",
            model_identity=UNIT_IDENTITY,
            progress={"next_epoch": 1},
            rng_state={},
            adapter_exports={
                "lora_weights": AdapterExport(module, adapter_name="expert"),
                "lora_weights/expert": AdapterExport(module),
            },
        )

    assert not (tmp_path / "checkpoint-effective-collision").exists()


def test_training_checkpoint_exports_lora_with_ema_without_mutating_resume_state(
    tmp_path,
) -> None:
    """The published adapter carries EMA weights; the resume checkpoint carries raw ones.

    Driven by a real ``EMAModuleWrapper``, so the 7.0 in the artifact is the
    wrapper's own running average copied in by its own ``copy_ema_to``, and the
    restore afterwards is its own ``copy_temp_to`` — the previous stand-in filled
    the number in itself, which made ``store_temp=True`` a claim about the stub
    rather than about the swap. Here that claim is implied: a swap taken without
    a snapshot could not restore 3.0 at all.
    """

    class _ExportModule(nn.Linear):
        def save_pretrained(self, path, *, state_dict, selected_adapters):
            assert selected_adapters == ["default"]
            path.mkdir(parents=True)
            save_file(dict(state_dict), path / "adapter_model.safetensors")

    export_module = _ExportModule(1, 1, bias=False)
    bundle = _Bundle(export_module)
    ema = _ema_holding(export_module, average=7.0, live=3.0)

    save_training_checkpoint(
        tmp_path / "checkpoint-ema",
        trainer=_Trainer(),
        bundle=bundle,
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
        adapter_exports={LORA_WEIGHTS_NAME: AdapterExport(export_module)},
        export_ema=ema,
    )

    checkpoint = TrainingCheckpoint.load(tmp_path / "checkpoint-ema")
    saved_trainable = checkpoint.checkpoint_state["module"]["weight"].item()
    published = _exported_adapter_weight(
        tmp_path / "checkpoint-ema" / LORA_WEIGHTS_NAME / "adapter_model.safetensors",
    )

    assert saved_trainable == pytest.approx(3.0)
    assert published.item() == pytest.approx(7.0)
    assert bundle.module.weight.item() == pytest.approx(3.0)
    assert ema.temp_stored_parameters is None


def test_training_checkpoint_skips_lora_ema_export_before_first_ema_update(
    tmp_path,
) -> None:
    """Before the first ``step()`` the EMA average is meaningless, so publish raw weights.

    The real wrapper derives ``has_updates`` from ``num_updates``, so "unupdated"
    is a state the object is genuinely in rather than a flag a stub sets. Its
    stored average is deliberately 7.0 while the live weight is 3.0: an export
    that swapped anyway would publish 7.0, and one that took a snapshot would
    leave ``temp_stored_parameters`` behind.
    """

    class _ExportModule(nn.Linear):
        def save_pretrained(self, path, *, state_dict, selected_adapters):
            assert selected_adapters == ["default"]
            path.mkdir(parents=True)
            save_file(dict(state_dict), path / "adapter_model.safetensors")

    export_module = _ExportModule(1, 1, bias=False)
    bundle = _Bundle(export_module)
    ema = _ema_holding(export_module, average=7.0, live=3.0, stepped=False)
    assert ema.has_updates is False

    save_training_checkpoint(
        tmp_path / "checkpoint-raw-export",
        trainer=_Trainer(),
        bundle=bundle,
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
        adapter_exports={LORA_WEIGHTS_NAME: AdapterExport(export_module)},
        export_ema=ema,
    )

    published = _exported_adapter_weight(
        tmp_path / "checkpoint-raw-export" / LORA_WEIGHTS_NAME / "adapter_model.safetensors",
    )
    assert published.item() == pytest.approx(3.0)
    assert ema.temp_stored_parameters is None  # no swap, so no snapshot was taken
    assert bundle.module.weight.item() == pytest.approx(3.0)


def test_adapter_export_derives_nested_module_state_prefix(tmp_path) -> None:
    class _ExportModule(nn.Linear):
        def __init__(self) -> None:
            super().__init__(1, 1, bias=False)
            self.saved_keys = None

        def save_pretrained(self, path, *, state_dict, selected_adapters):
            self.saved_keys = set(state_dict)
            assert selected_adapters == ["default"]
            path.mkdir(parents=True)
            save_file(dict(state_dict), path / "adapter_model.safetensors")

    class _Root(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = _ExportModule()

    root = _Root()
    bundle = _Bundle(root)

    save_training_checkpoint(
        tmp_path / "checkpoint-nested-adapter",
        trainer=_Trainer(),
        bundle=bundle,
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
        adapter_exports={
            LORA_WEIGHTS_NAME: AdapterExport(root.language_model),
        },
    )

    assert root.language_model.saved_keys == {"weight"}


def test_adapter_export_selects_default_and_excludes_frozen_previous(tmp_path) -> None:
    # peft is an optional extra, safetensors is a core dependency — only the
    # former can legitimately be missing, so only the former guards the skip.
    peft = pytest.importorskip("peft")

    from vrl.models.steps.denoise.common.lora import (
        freeze_checkpoint_owned_adapter_params,
    )

    class _Base(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(2, 2, bias=False)

        def forward(self, inputs):
            return self.lin(inputs)

    config = peft.LoraConfig(r=2, lora_alpha=4, target_modules=["lin"])
    module = peft.get_peft_model(_Base(), config)
    module.add_adapter("previous", config)
    freeze_checkpoint_owned_adapter_params(module, "previous")
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if ".default." in name:
                parameter.fill_(3.0)
            elif ".previous." in name:
                parameter.fill_(9.0)
    bundle = _Bundle(module)

    save_training_checkpoint(
        tmp_path / "checkpoint-default-only",
        trainer=_Trainer(),
        bundle=bundle,
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"next_epoch": 1},
        rng_state={},
        adapter_exports={LORA_WEIGHTS_NAME: AdapterExport(module)},
    )

    checkpoint = TrainingCheckpoint.load(tmp_path / "checkpoint-default-only")
    root_state = checkpoint.checkpoint_state["module"]
    assert any(".previous." in name for name in root_state)
    artifact_dir = tmp_path / "checkpoint-default-only" / LORA_WEIGHTS_NAME
    artifact = load_file(artifact_dir / "adapter_model.safetensors")
    assert artifact
    assert all(torch.equal(tensor, torch.full_like(tensor, 3.0)) for tensor in artifact.values())
    assert not (artifact_dir / "previous").exists()
