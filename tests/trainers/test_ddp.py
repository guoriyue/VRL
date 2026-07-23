"""DDP training strategy (SPRINT_symmetric_colocated_ddp.md).

We cannot run real multi-GPU here, but ``DistributedDataParallel`` runs on a
single CPU rank (``world_size=1`` + gloo): the module wraps, forward/backward
works, and because DDP keeps FULL (replicated, not sharded) params the rollout/
checkpoint export is the single-process full-state path on the unwrapped module.
That is enough to exercise the strategy for real — wrapping, clip, and the
invariant that the rollout-facing key space is identical to single-process.

The build_strategy dispatch + the model-handle guard need no process group and
run unconditionally.
"""

from __future__ import annotations

import socket

import pytest
import torch
from torch import nn

from vrl.config.schema import DDPConfig, RootConfig
from vrl.models.interfaces.runtime import register_checkpoint_owned_state
from vrl.trainers.distributed import DistributedTrainingContext
from vrl.trainers.strategy import (
    DDPStrategy,
    SingleProcessStrategy,
    build_strategy,
)

# ── fixtures / fakes ────────────────────────────────────────────────────────


def _cpu_ddp_context() -> DistributedTrainingContext:
    return DistributedTrainingContext(
        strategy="ddp",
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )


def _ddp_strategy(
    context: DistributedTrainingContext,
    **overrides: bool,
) -> DDPStrategy:
    config = DDPConfig.model_validate(overrides)
    return DDPStrategy(
        context,
        find_unused_parameters=config.find_unused_parameters,
    )


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


class _ToyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(2)])
        self.head = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class _FakePolicy:
    """Diffusion-policy shape the wrapper needs: transformer + _set_transformer."""

    def __init__(self, transformer: nn.Module) -> None:
        self.transformer = transformer
        self.set_calls = 0

    def _set_transformer(self, transformer: nn.Module) -> None:
        self.transformer = transformer
        self.set_calls += 1

    @property
    def trainable_modules(self) -> dict[str, nn.Module]:
        return {"transformer": self.transformer}


class _DualStagePolicy(_FakePolicy):
    """Wan-style policy with two independently writable trainable roots."""

    def __init__(self, transformer: nn.Module) -> None:
        super().__init__(transformer)
        self.transformer_2 = _ToyTransformer()
        self.set_2_calls = 0

    def _set_transformer_2(self, transformer: nn.Module) -> None:
        self.transformer_2 = transformer
        self.set_2_calls += 1

    @property
    def trainable_modules(self) -> dict[str, nn.Module]:
        return {"transformer": self.transformer, "transformer_2": self.transformer_2}


class _ARLikePolicy:
    """No transformer handle / _set_transformer (AR family shape)."""


class _Bundle:
    def __init__(self, module: nn.Module) -> None:
        self.trainable_modules = {"transformer": module}


@pytest.fixture(scope="module")
def cpu_process_group():
    """One gloo world_size=1 group for the collective tests in this module."""

    import torch.distributed as dist

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    mp = pytest.MonkeyPatch()
    mp.setenv("MASTER_ADDR", "127.0.0.1")
    mp.setenv("MASTER_PORT", str(port))
    mp.setenv("RANK", "0")
    mp.setenv("WORLD_SIZE", "1")
    mp.setenv("LOCAL_RANK", "0")
    created = False
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        created = True
    yield
    if created and dist.is_initialized():
        dist.destroy_process_group()
    mp.undo()


def _ddp_wrap(module: nn.Module) -> nn.Module:
    from torch.nn.parallel import DistributedDataParallel

    return DistributedDataParallel(module, device_ids=None)  # CPU rank


# ── build_strategy dispatch + model-handle guard (no process group) ──────────


@pytest.mark.parametrize("find_unused_parameters", [False, True])
def test_build_strategy_reads_ddp_public_default_and_override(
    find_unused_parameters: bool,
) -> None:
    ddp = {} if not find_unused_parameters else {"find_unused_parameters": True}
    strategy = build_strategy(
        RootConfig.model_validate(
            {
                "distributed": {
                    "training": {
                        "strategy": "ddp",
                        "ddp": ddp,
                    },
                },
            },
        ),
        _cpu_ddp_context(),
    )

    assert isinstance(strategy, DDPStrategy)
    assert strategy._find_unused_parameters is find_unused_parameters


def test_ddp_strategy_constructor_requires_resolved_config() -> None:
    with pytest.raises(TypeError, match="find_unused_parameters"):
        DDPStrategy(_cpu_ddp_context())  # type: ignore[call-arg]


def test_ddp_rejects_shared_gpu_training_state_parking_preflight() -> None:
    with pytest.raises(NotImplementedError, match="Use disjoint rollout GPUs"):
        _ddp_strategy(_cpu_ddp_context()).validate_training_state_parking()


def test_ddp_shutdown_releases_training_process_group(monkeypatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(
        "vrl.trainers.fsdp.shutdown_training_process_group",
        lambda: calls.append(True),
    )

    _ddp_strategy(_cpu_ddp_context()).shutdown()

    assert calls == [True]


def test_ddp_prepare_model_rejects_model_without_transformer_handle() -> None:
    with pytest.raises(NotImplementedError, match="trainable roots"):
        _ddp_strategy(_cpu_ddp_context()).prepare_model(_ARLikePolicy())


def test_ddp_prepare_model_wraps_multi_transformer_model(cpu_process_group) -> None:
    """Dual-stage Wan wraps both named roots and writes both aliases back."""
    from torch.nn.parallel import DistributedDataParallel

    policy = _DualStagePolicy(_ToyTransformer())
    out = _ddp_strategy(_cpu_ddp_context()).prepare_model(policy)

    assert out is policy
    assert policy.set_calls == 1
    assert policy.set_2_calls == 1
    assert isinstance(policy.transformer, DistributedDataParallel)
    assert isinstance(policy.transformer_2, DistributedDataParallel)


# ── wrapping + export (need a process group) ─────────────────────────────────


def test_ddp_prepare_model_wraps_transformer(cpu_process_group) -> None:
    from torch.nn.parallel import DistributedDataParallel

    policy = _FakePolicy(_ToyTransformer())
    out = _ddp_strategy(_cpu_ddp_context()).prepare_model(policy)

    assert out is policy
    assert policy.set_calls == 1
    assert isinstance(policy.transformer, DistributedDataParallel)


def test_ddp_rollout_export_matches_single_process_key_space(cpu_process_group) -> None:
    """The invariant: DDP-wrapped rollout state == single-process rollout state.

    Same keys (clean ``transformer.*``, no ``.module.`` leak), same values — a
    rollout worker is oblivious to whether the trainer ran DDP.
    """
    ref = _ToyTransformer()
    snapshot = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    replicated = _ToyTransformer()
    replicated.load_state_dict(snapshot)
    wrapped = _ddp_wrap(replicated)

    got = _ddp_strategy(_cpu_ddp_context()).export_rollout_state(_Bundle(wrapped))
    expected = SingleProcessStrategy().export_rollout_state(_Bundle(ref))

    assert got.keys() == expected.keys()
    assert all(key.startswith("transformer.") for key in got)
    assert all("module" not in key for key in got)
    for key in got:
        assert torch.allclose(got[key], expected[key])


def test_ddp_rollout_export_filters_frozen_params(cpu_process_group) -> None:
    """Rollout excludes frozen state while checkpoint includes registered state."""
    net = _ToyTransformer()
    net.head.requires_grad_(False)
    register_checkpoint_owned_state(
        net,
        [name for name, _ in net.named_parameters() if name.startswith("head.")],
    )
    wrapped = _ddp_wrap(net)
    strategy = _ddp_strategy(_cpu_ddp_context())

    rollout = strategy.export_rollout_state(_Bundle(wrapped))
    assert rollout
    assert not any("head" in key for key in rollout)

    checkpoint = strategy.export_checkpoint_state(_Bundle(wrapped))["transformer"]
    assert any("head" in key for key in checkpoint)


def test_ddp_checkpoint_state_excludes_unregistered_frozen_params(cpu_process_group) -> None:
    net = _ToyTransformer()
    net.head.requires_grad_(False)
    strategy = _ddp_strategy(_cpu_ddp_context())

    checkpoint = strategy.export_checkpoint_state(_Bundle(_ddp_wrap(net)))["transformer"]

    assert not any("head" in key for key in checkpoint)


def test_ddp_export_then_load_checkpoint_state_round_trip(cpu_process_group) -> None:
    strategy = _ddp_strategy(_cpu_ddp_context())
    src = _ddp_wrap(_ToyTransformer())
    with torch.no_grad():
        for p in src.parameters():
            p.fill_(3.0)

    snapshot = strategy.export_checkpoint_state(_Bundle(src))
    assert set(snapshot) == {"transformer"}
    first_name, first_value = next(iter(snapshot["transformer"].items()))
    live = dict(src.module.state_dict())[first_name]
    assert first_value.data_ptr() != live.data_ptr()

    dst = _ddp_wrap(_ToyTransformer())
    strategy.load_checkpoint_state(_Bundle(dst), snapshot)
    for value in dst.module.state_dict().values():
        assert torch.allclose(value, torch.full_like(value, 3.0))


def test_ddp_restore_protocol_loads_schema_v1_full_frozen_state(
    cpu_process_group,
    tmp_path,
) -> None:
    from vrl.trainers.checkpointing import TrainingCheckpoint, restore_model_checkpoint

    source_module = _ToyTransformer()
    source_module.head.requires_grad_(False)
    with torch.no_grad():
        for parameter in source_module.parameters():
            parameter.fill_(7.0)
    source = _ddp_wrap(source_module)

    restored_module = _ToyTransformer()
    restored_module.head.requires_grad_(False)
    with torch.no_grad():
        for parameter in restored_module.parameters():
            parameter.fill_(3.0)
    restored = _ddp_wrap(restored_module)
    checkpoint = TrainingCheckpoint(
        checkpoint_dir=tmp_path,
        checkpoint_path=tmp_path / "checkpoint.pt",
        payload={
            "schema_version": 1,
            "family": "toy",
            "trainer": {},
            "model": {"trainable_modules": {"transformer": dict(source.module.state_dict())}},
            "progress": {},
            "rng": {},
        },
        meta={},
    )

    restore_model_checkpoint(
        checkpoint,
        bundle=_Bundle(restored),
        family="toy",
        strict=True,
        strategy=_ddp_strategy(_cpu_ddp_context()),
    )

    assert all(
        torch.allclose(value, torch.full_like(value, 7.0))
        for value in restored.module.state_dict().values()
    )


def test_ddp_adapter_export_uses_gathered_checkpoint_state(
    cpu_process_group,
    tmp_path,
) -> None:
    from types import SimpleNamespace

    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file

    from vrl.trainers.checkpointing import (
        LORA_WEIGHTS_NAME,
        AdapterExport,
        save_training_checkpoint,
    )

    module = get_peft_model(
        _ToyTransformer(),
        LoraConfig(r=2, lora_alpha=4, target_modules=["lin"]),
    )
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.requires_grad:
                parameter.fill_(5.0)
    wrapped = _ddp_wrap(module)

    save_training_checkpoint(
        tmp_path,
        trainer=SimpleNamespace(state_dict=lambda: {"step": 0, "global_step": 0}),
        bundle=_Bundle(wrapped),
        family="toy",
        model_identity={"schema": "toy/v1"},
        progress={"next_epoch": 1},
        rng_state={},
        adapter_exports={LORA_WEIGHTS_NAME: AdapterExport(module)},
        strategy=_ddp_strategy(_cpu_ddp_context()),
    )

    artifact = load_file(
        str(tmp_path / LORA_WEIGHTS_NAME / "adapter_model.safetensors"),
    )
    assert artifact
    assert all(torch.equal(value, torch.full_like(value, 5.0)) for value in artifact.values())
