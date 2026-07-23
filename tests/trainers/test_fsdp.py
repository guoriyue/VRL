"""FSDP2 strategy layer (SPRINT_multi_gpu_training.md).

We cannot run real multi-GPU here, but FSDP2 ``fully_shard`` runs on a single CPU
rank (``world_size=1`` + gloo): params become DTensors, forward/backward works,
and ``get_model_state_dict(full_state_dict=True)`` materializes plain full
tensors. That is enough to exercise the whole strategy path for real — wrapping,
DTensor-aware clip, full-state gather/load round-trip, and the invariant that the
rollout-facing key space is identical whether or not the trainer was sharded.

Pure helpers (unwrap / block discovery / mixed precision / mesh validation) and
the build_strategy §10 gates need no process group and run unconditionally.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import torch
from torch import nn

from vrl.config.loading import load_config
from vrl.config.schema import FSDPConfig, RootConfig, parse_config
from vrl.trainers.distributed import DistributedTrainingContext
from vrl.trainers.fsdp import (
    apply_fsdp,
    build_fsdp_mesh,
    gather_full_state_dict,
    iter_blocks,
    load_full_state_dict,
    mixed_precision_policy,
    unwrap_module,
)
from vrl.trainers.strategy import (
    FSDPStrategy,
    SingleProcessStrategy,
    build_strategy,
)

# ── fixtures / fakes ────────────────────────────────────────────────────────


def _cpu_fsdp_context() -> DistributedTrainingContext:
    return DistributedTrainingContext(
        strategy="fsdp",
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )


def _fsdp_strategy(
    context: DistributedTrainingContext,
    **overrides: Any,
) -> FSDPStrategy:
    config = FSDPConfig.model_validate(overrides)
    return FSDPStrategy(
        context,
        mesh_dims=config.mesh,
        precision_policy=config.precision_policy,
        reshard_after_forward=config.reshard_after_forward,
        cpu_offload=config.cpu_offload,
    )


def _strategy_config(
    strategy: str,
    *,
    strategy_config: dict[str, Any] | None = None,
    model: dict[str, Any] | None = None,
) -> RootConfig:
    training: dict[str, Any] = {"strategy": strategy}
    if strategy_config is not None:
        training[strategy] = strategy_config
    payload: dict[str, Any] = {"distributed": {"training": training}}
    if model is not None:
        payload["model"] = {"family": "sana", **model}
    return RootConfig.model_validate(payload)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


class _ToyTransformer(nn.Module):
    """Stands in for a diffusers DiT: per-layer blocks named in _no_split_modules."""

    _no_split_modules: ClassVar[list[str]] = ["_Block"]

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(2)])
        self.head = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class _FakePolicy:
    """Diffusion-policy shape the FSDP applier needs: transformer + _set_transformer."""

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


class _Bundle:
    def __init__(self, module: nn.Module) -> None:
        self.trainable_modules = {"transformer": module}


@pytest.fixture(scope="module")
def cpu_process_group():
    """One gloo world_size=1 group for the collective tests in this module.

    Uses a free ephemeral port (no fixed-port collision under parallel sessions)
    and a self-restoring MonkeyPatch so the torchrun env vars do not leak into the
    rest of the suite.
    """

    import socket

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


def _shard(module: nn.Module) -> nn.Module:
    return apply_fsdp(
        module,
        mesh=build_fsdp_mesh(_cpu_fsdp_context(), ["dp_shard"]),
        mp_policy=mixed_precision_policy("none"),  # fp32 keeps the CPU path simple
    )


# ── pure helpers (no process group) ─────────────────────────────────────────


def test_unwrap_module_peels_compile_then_peft_get_base_model() -> None:
    base = _ToyTransformer()
    peft = SimpleNamespace(get_base_model=lambda: base)
    compiled = SimpleNamespace(_orig_mod=peft)
    assert unwrap_module(compiled) is base


def test_unwrap_module_peels_peft_base_model_model() -> None:
    base = _ToyTransformer()
    peft = SimpleNamespace(base_model=SimpleNamespace(model=base))
    assert unwrap_module(peft) is base


def test_unwrap_module_returns_plain_module_unchanged() -> None:
    base = _ToyTransformer()
    assert unwrap_module(base) is base


def test_iter_blocks_yields_no_split_modules() -> None:
    net = _ToyTransformer()
    blocks = list(iter_blocks(net))
    assert len(blocks) == 2
    assert all(isinstance(b, _Block) for b in blocks)


def test_iter_blocks_fails_without_no_split_modules() -> None:
    net = nn.Linear(4, 4)  # no _no_split_modules
    with pytest.raises(ValueError, match="_no_split_modules"):
        list(iter_blocks(net))


def test_mixed_precision_policy_actor_uses_resolved_params_fp32_reduce() -> None:
    policy = mixed_precision_policy("actor", parameter_dtype=torch.float16)
    assert policy.param_dtype == torch.float16
    assert policy.reduce_dtype == torch.float32


def test_mixed_precision_policy_actor_requires_resolved_parameter_dtype() -> None:
    with pytest.raises(ValueError, match="resolved parameter dtype"):
        mixed_precision_policy("actor")


def test_mixed_precision_policy_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="precision_policy"):
        mixed_precision_policy("fp8")


def test_build_fsdp_mesh_rejects_2d_hsdp() -> None:
    with pytest.raises(ValueError, match="1D"):
        build_fsdp_mesh(_cpu_fsdp_context(), ["dp_replicate", "dp_shard"])


# ── real FSDP2 on a single CPU rank ──────────────────────────────────────────


def test_apply_fsdp_shards_params_and_runs_forward_backward(cpu_process_group) -> None:
    from torch.distributed.tensor import DTensor

    net = _shard(_ToyTransformer())
    assert any(isinstance(p, DTensor) for p in net.parameters())

    out = net(torch.randn(3, 4))
    out.sum().backward()
    assert all(p.grad is not None for p in net.parameters() if p.requires_grad)


def test_gather_full_state_dict_materializes_plain_full_tensors(cpu_process_group) -> None:
    from torch.distributed.tensor import DTensor

    ref = _ToyTransformer()
    snapshot = {k: v.detach().clone() for k, v in ref.state_dict().items()}

    sharded = _ToyTransformer()
    sharded.load_state_dict(snapshot)
    _shard(sharded)

    full = gather_full_state_dict(sharded)
    assert set(full) == set(snapshot)  # unwrapped keys, no shard/wrapper leakage
    for key, value in full.items():
        assert not isinstance(value, DTensor)  # materialized to a plain tensor
        assert torch.allclose(value, snapshot[key])


def test_load_full_state_dict_round_trips_into_sharded_module(cpu_process_group) -> None:
    sharded = _shard(_ToyTransformer())
    # set_model_state_dict shards the input in place, so keep a plain `expected`
    # snapshot and feed the loader a clone.
    expected = {k: torch.full_like(v, 0.5) for k, v in gather_full_state_dict(sharded).items()}

    load_full_state_dict(sharded, {k: v.clone() for k, v in expected.items()})

    reloaded = gather_full_state_dict(sharded)
    for key, value in reloaded.items():
        assert torch.allclose(value, expected[key])


def _full_grad_norm(net: nn.Module) -> float:
    from torch.distributed.tensor import DTensor

    grads = [
        (p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad).flatten()
        for p in net.parameters()
        if p.grad is not None
    ]
    return float(torch.linalg.vector_norm(torch.cat(grads)))


def test_fsdp_clip_grad_norm_returns_global_norm_and_actually_clips(cpu_process_group) -> None:
    net = _shard(_ToyTransformer())
    net(torch.randn(3, 4)).mul(5.0).sum().backward()  # large grads so clipping bites

    pre = _full_grad_norm(net)
    returned = _fsdp_strategy(_cpu_fsdp_context(), precision_policy="none").clip_grad_norm(
        net.parameters(),
        max_norm=0.1,
    )

    # The returned norm is the global (full-tensor) pre-clip norm, not a per-shard
    # value, and the grads are actually scaled down to max_norm.
    assert returned == pytest.approx(pre, rel=1e-4)
    assert pre > 0.1  # clipping genuinely engaged
    assert _full_grad_norm(net) <= 0.1 + 1e-4


def test_fsdp_rollout_export_matches_single_process_key_space(cpu_process_group) -> None:
    """The §9 invariant: sharded rollout state == single-process rollout state.

    Same keys, same values — so a rollout worker's ``load_trainable_state`` is
    oblivious to whether the trainer was sharded.
    """
    ref = _ToyTransformer()
    snapshot = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    sharded = _ToyTransformer()
    sharded.load_state_dict(snapshot)
    _shard(sharded)

    got = _fsdp_strategy(_cpu_fsdp_context(), precision_policy="none").export_rollout_state(
        _Bundle(sharded),
    )
    expected = SingleProcessStrategy().export_rollout_state(_Bundle(ref))

    assert got.keys() == expected.keys()
    assert all(key.startswith("transformer.") for key in got)
    for key in got:
        assert torch.allclose(got[key], expected[key])


def test_fsdp_rollout_export_unwraps_torch_compile_to_clean_keys(cpu_process_group) -> None:
    """A torch.compile()'d transformer (the default in live configs) must export
    clean ``transformer.*`` keys, not ``_orig_mod.``-prefixed ones.

    get_model_state_dict strips ``_orig_mod.`` while named_parameters keeps it, so
    without unwrapping first the trainable-key select disagrees and crashes. This
    locks the export path against the real compiled-policy shape.
    """
    ref = _ToyTransformer()
    snapshot = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    inner = _ToyTransformer()
    inner.load_state_dict(snapshot)
    compiled = torch.compile(inner)  # OptimizedModule with _orig_mod, like production
    _shard(compiled)

    got = _fsdp_strategy(_cpu_fsdp_context(), precision_policy="none").export_rollout_state(
        _Bundle(compiled),
    )
    expected = SingleProcessStrategy().export_rollout_state(_Bundle(torch.compile(ref)))

    assert all("_orig_mod" not in key for key in got)
    assert got.keys() == expected.keys()
    for key in got:
        assert torch.allclose(got[key], expected[key])


def test_fsdp_rollout_export_filters_frozen_params(cpu_process_group) -> None:
    """Rollout and checkpoint export gather only mutable LoRA-style parameters."""
    net = _ToyTransformer()
    net.head.requires_grad_(False)  # freeze the non-block head
    net.register_buffer("frozen_cache", torch.ones(4))
    _shard(net)
    strategy = _fsdp_strategy(_cpu_fsdp_context(), precision_policy="none")
    expected = {name for name, parameter in net.named_parameters() if parameter.requires_grad}

    rollout = strategy.export_rollout_state(_Bundle(net))
    assert set(rollout) == {f"transformer.{name}" for name in expected}

    checkpoint = strategy.export_trainable_state(_Bundle(net))["transformer"]
    assert set(checkpoint) == expected
    assert not any("head" in key or "frozen_cache" in key for key in checkpoint)


def test_fsdp_prepare_model_wraps_multi_transformer_model(cpu_process_group) -> None:
    """Dual-stage Wan shards both named roots and writes both aliases back."""
    from torch.distributed.tensor import DTensor

    policy = _DualStagePolicy(_ToyTransformer())
    out = _fsdp_strategy(_cpu_fsdp_context(), precision_policy="none").prepare_model(policy)

    assert out is policy
    assert policy.set_calls == 1
    assert policy.set_2_calls == 1
    assert any(isinstance(parameter, DTensor) for parameter in policy.transformer.parameters())
    assert any(isinstance(parameter, DTensor) for parameter in policy.transformer_2.parameters())


def test_fsdp_export_then_load_trainable_state_round_trip(cpu_process_group) -> None:
    strategy = _fsdp_strategy(_cpu_fsdp_context(), precision_policy="none")
    src_module = _ToyTransformer()
    src_module.head.requires_grad_(False)
    with torch.no_grad():
        for parameter in src_module.parameters():
            parameter.fill_(3.0 if parameter.requires_grad else 5.0)
    src = _shard(src_module)

    snapshot = strategy.export_trainable_state(_Bundle(src))
    assert set(snapshot) == {"transformer"}
    assert not any("head" in key for key in snapshot["transformer"])

    dst_module = _ToyTransformer()
    dst_module.head.requires_grad_(False)
    with torch.no_grad():
        dst_module.head.weight.fill_(11.0)
        dst_module.head.bias.fill_(11.0)
    dst = _shard(dst_module)
    strategy.load_trainable_state(_Bundle(dst), snapshot)
    restored = gather_full_state_dict(dst)
    for name, value in restored.items():
        expected = 11.0 if name.startswith("head.") else 3.0
        assert torch.allclose(value, torch.full_like(value, expected))


def test_fsdp_load_trainable_state_accepts_legacy_full_checkpoint(cpu_process_group) -> None:
    strategy = _fsdp_strategy(_cpu_fsdp_context(), precision_policy="none")
    src_module = _ToyTransformer()
    src_module.head.requires_grad_(False)
    with torch.no_grad():
        for parameter in src_module.parameters():
            parameter.fill_(7.0)
    src = _shard(src_module)
    legacy = {"transformer": gather_full_state_dict(src)}

    dst_module = _ToyTransformer()
    dst_module.head.requires_grad_(False)
    dst = _shard(dst_module)
    strategy.load_trainable_state(_Bundle(dst), legacy, strict=True)

    for value in gather_full_state_dict(dst).values():
        assert torch.allclose(value, torch.full_like(value, 7.0))


def test_fsdp_load_trainable_state_strictly_validates_mutable_keys(cpu_process_group) -> None:
    strategy = _fsdp_strategy(_cpu_fsdp_context(), precision_policy="none")
    net = _ToyTransformer()
    net.head.requires_grad_(False)
    sharded = _shard(net)
    snapshot = strategy.export_trainable_state(_Bundle(sharded))
    state = snapshot["transformer"]

    missing = {"transformer": dict(state)}
    missing["transformer"].pop(next(iter(state)))
    with pytest.raises(ValueError, match="missing="):
        strategy.load_trainable_state(_Bundle(sharded), missing, strict=True)

    unexpected = {"transformer": {**state, "unknown.weight": torch.ones(1)}}
    with pytest.raises(ValueError, match="unexpected="):
        strategy.load_trainable_state(_Bundle(sharded), unexpected, strict=True)


def test_fsdp_prepare_model_wraps_diffusion_handle(cpu_process_group) -> None:
    from torch.distributed.tensor import DTensor

    policy = _FakePolicy(_ToyTransformer())
    out = _fsdp_strategy(_cpu_fsdp_context(), precision_policy="none").prepare_model(policy)

    assert out is policy
    assert policy.set_calls == 1
    assert any(isinstance(p, DTensor) for p in policy.transformer.parameters())


def test_wan_fsdp_replay_build_defers_full_gpu_move_until_sharding(
    cpu_process_group,
    monkeypatch,
) -> None:
    """Wan replay attaches LoRA on CPU before FSDP2 owns materialization."""
    from omegaconf import OmegaConf
    from torch.distributed.tensor import DTensor

    from vrl.families.registry import get_model_family_entry
    from vrl.models.steps.denoise import build as denoise_build

    class _TrackingWanTransformer(_ToyTransformer):
        def __init__(self) -> None:
            super().__init__()
            self.to_calls = 0

        def to(self, *args: Any, **kwargs: Any) -> _TrackingWanTransformer:
            self.to_calls += 1
            return super().to(*args, **kwargs)

    transformer = _TrackingWanTransformer()
    monkeypatch.setattr(
        denoise_build,
        "load_diffusers_transformer",
        lambda _build, _class_name: transformer,
    )
    monkeypatch.setattr(
        denoise_build,
        "load_diffusers_scheduler",
        lambda _build, _class_name: object(),
    )
    cfg = OmegaConf.create(
        {
            "model": {
                "path": "fake/Wan2.1-I2V",
                "use_lora": True,
                "lora": {
                    "rank": 2,
                    "alpha": 4,
                    "target_modules": ["lin"],
                },
            },
            "precision": {
                "float32_precision": "ieee",
                "training": {"dtype": "fp32"},
                "rollout": {"dtype": "fp32"},
            },
            "distributed": {"training": {"strategy": "fsdp"}},
        },
    )
    entry = get_model_family_entry("wan_2_1_i2v")
    build = entry.resolve_model_build(
        cfg,
        torch.device("cpu"),
        for_rollout=False,
    )

    assert build.defer_trainable_device_move is True
    assert (
        entry.resolve_model_build(
            cfg, torch.device("cpu"), for_rollout=True
        ).defer_trainable_device_move
        is False
    )
    bundle = entry.build_replay(build)
    assert transformer.to_calls == 0
    trainable_names = [
        name for name, parameter in bundle.model.named_parameters() if parameter.requires_grad
    ]
    assert trainable_names
    assert all("lora_" in name for name in trainable_names)

    _fsdp_strategy(_cpu_fsdp_context(), precision_policy="none").prepare_model(
        bundle.model,
    )

    assert transformer.to_calls == 0
    assert any(isinstance(parameter, DTensor) for parameter in bundle.model.parameters())


def test_fsdp_prepare_model_initializes_process_group(cpu_process_group, monkeypatch) -> None:
    """prepare_model explicitly owns PG init + device bind, symmetric with DDP.

    init_device_mesh would lazily auto-init a default group, but it would NOT call
    torch.cuda.set_device(local_rank) first, so the NCCL group and per-block
    fully_shard could bind the wrong card on a single-node multi-GPU box.
    FSDPStrategy.prepare_model therefore calls init_training_process_group up front.
    Spy on it to lock the wiring in (the gloo PG already exists here, so the real
    init is a no-op — we assert the call, with the cpu-context gloo backend).
    """

    import vrl.trainers.fsdp as fsdp_mod

    calls: list[tuple] = []
    real = fsdp_mod.init_training_process_group

    def _spy(context, *, backend):
        calls.append((context.strategy, backend))
        return real(context, backend=backend)

    monkeypatch.setattr(fsdp_mod, "init_training_process_group", _spy)

    policy = _FakePolicy(_ToyTransformer())
    _fsdp_strategy(_cpu_fsdp_context(), precision_policy="none").prepare_model(policy)

    assert calls == [("fsdp", "gloo")]


# ── prepare_model family gate / build_strategy §10 gates (no process group) ──


def test_fsdp_prepare_model_rejects_model_without_transformer_handle() -> None:
    class _ARLikePolicy:
        pass

    with pytest.raises(NotImplementedError, match="trainable roots"):
        _fsdp_strategy(_cpu_fsdp_context(), precision_policy="none").prepare_model(_ARLikePolicy())


def test_build_strategy_single_process_returns_single_process() -> None:
    ctx = DistributedTrainingContext(
        strategy="single_process",
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )
    assert isinstance(build_strategy(RootConfig(), ctx), SingleProcessStrategy)


@pytest.mark.parametrize(
    ("strategy_config", "precision_policy", "reshard_after_forward", "cpu_offload"),
    [
        (None, "actor", True, False),
        (
            {
                "mesh": ["dp_shard"],
                "precision_policy": "none",
                "reshard_after_forward": False,
                "cpu_offload": True,
            },
            "none",
            False,
            True,
        ),
    ],
)
def test_build_strategy_fsdp_reads_public_defaults_and_overrides(
    strategy_config: dict[str, Any] | None,
    precision_policy: str,
    reshard_after_forward: bool,
    cpu_offload: bool,
) -> None:
    strategy = build_strategy(
        _strategy_config("fsdp", strategy_config=strategy_config),
        _cpu_fsdp_context(),
    )

    assert isinstance(strategy, FSDPStrategy)
    assert strategy._mesh_dims == ["dp_shard"]
    assert strategy._precision_policy == precision_policy
    assert strategy._reshard_after_forward is reshard_after_forward
    assert strategy._cpu_offload is cpu_offload


def test_base_fsdp_preset_omits_schema_defaults_but_resolves_them() -> None:
    cfg = load_config("base/distributed/training_fsdp")

    assert "fsdp" not in cfg.distributed.training
    training = parse_config(cfg).distributed.training

    assert training is not None
    assert training.fsdp == FSDPConfig()


def test_fsdp_strategy_constructor_requires_resolved_config() -> None:
    with pytest.raises(TypeError, match="mesh_dims"):
        FSDPStrategy(_cpu_fsdp_context())  # type: ignore[call-arg]


def test_build_strategy_rejects_config_context_mismatch() -> None:
    with pytest.raises(ValueError, match="strategy mismatch"):
        build_strategy(RootConfig(), _cpu_fsdp_context())


def test_build_strategy_rejects_raw_unvalidated_config() -> None:
    with pytest.raises(TypeError, match="must be RootConfig"):
        build_strategy({}, _cpu_fsdp_context())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("strategy", "unused_block"),
    [
        ("single_process", {"fsdp": {}}),
        ("fsdp", {"ddp": {}}),
    ],
)
def test_training_config_rejects_unselected_strategy_blocks(
    strategy: str,
    unused_block: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match=r"requires distributed\.training\.strategy"):
        RootConfig.model_validate(
            {
                "distributed": {
                    "training": {
                        "strategy": strategy,
                        **unused_block,
                    },
                },
            },
        )


def test_fsdp_accepts_shared_gpu_training_state_parking_preflight() -> None:
    """Shared-GPU parking is implemented for FSDP, so preflight must not refuse.

    Refusing here is what made symmetric-colocated multi-rank online RL
    unreachable: the rollout topology gate requires colocation and this gate
    rejected colocation, so no configuration satisfied both.
    """

    assert _fsdp_strategy(_cpu_fsdp_context()).validate_training_state_parking() is None


def test_fsdp_parking_rolls_every_rank_back_when_one_peer_fails() -> None:
    """A peer's failure must restore this rank too, not leave residency split.

    Parking itself issues no collective, so ranks cannot drift apart by parking.
    They drift apart on FAILURE: one rank rolls back while the others stay
    parked, and the next all-gather hangs instead of surfacing the real error.
    """

    import torch
    import torch.nn as nn

    from vrl.trainers.strategy import TrainingMemoryState

    strategy = _fsdp_strategy(_cpu_fsdp_context())
    # This rank parks cleanly; the agreement reports that a peer did not.
    strategy._all_ranks_succeeded = lambda ok: False

    model = nn.Linear(4, 4)
    state = TrainingMemoryState(
        model=model,
        ref_model=None,
        optimizer=None,
        ema=None,
        grad_scaler=None,
        device=torch.device("cpu"),
    )

    with pytest.raises(RuntimeError, match="a peer rank failed to park"):
        strategy.park_training_state(state)

    # Rolled back: nothing stays parked, so the world is resident again.
    assert strategy._parked_training_state is None


def test_fsdp_shutdown_releases_training_process_group(monkeypatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(
        "vrl.trainers.fsdp.shutdown_training_process_group",
        lambda: calls.append(True),
    )

    _fsdp_strategy(_cpu_fsdp_context()).shutdown()

    assert calls == [True]


def test_build_strategy_fsdp_allows_ema_and_resume() -> None:
    # The original §10 gates, now lifted: EMA updates through DTensor
    # local-shard views (EMAModuleWrapper) and optimizer resume goes through
    # the strategy's full-state export/load.
    assert isinstance(
        build_strategy(
            RootConfig.model_validate(
                {
                    "actor": {"ema": {"enable": True}},
                    "distributed": {"training": {"strategy": "fsdp"}},
                },
            ),
            _cpu_fsdp_context(),
        ),
        FSDPStrategy,
    )
    assert isinstance(
        build_strategy(
            RootConfig.model_validate(
                {
                    "trainer": {"resume_from": "/ckpt/checkpoint-10"},
                    "distributed": {"training": {"strategy": "fsdp"}},
                },
            ),
            _cpu_fsdp_context(),
        ),
        FSDPStrategy,
    )


def test_build_strategy_fsdp_rejects_train_compile() -> None:
    # torch.compile (inductor) is unsound with FSDP2 reshard-after-forward all-gathers;
    # the build_strategy §10 gate must reject it.
    with pytest.raises(NotImplementedError, match="torch_compile"):
        build_strategy(
            _strategy_config(
                "fsdp",
                model={"torch_compile": {"enable": True}},
            ),
            _cpu_fsdp_context(),
        )


# ── gathered HF-adapter export (save_pretrained under FSDP2) ─────────────────


def test_fsdp_export_modules_writes_gathered_hf_adapter(cpu_process_group, tmp_path) -> None:
    """save_pretrained under FSDP2 serializes the gathered adapter, not shards.

    The online recipe passes export_modules under fsdp too;
    save_training_checkpoint must detect the DTensor-sharded module and feed
    save_pretrained the full state it already gathered for checkpoint.pt.
    """

    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file
    from torch.distributed.tensor import DTensor

    from vrl.trainers.checkpointing import (
        LORA_WEIGHTS_NAME,
        load_training_checkpoint,
        save_training_checkpoint,
    )

    torch.manual_seed(0)
    peft_model = get_peft_model(
        _ToyTransformer(),
        LoraConfig(r=2, lora_alpha=4, init_lora_weights="gaussian", target_modules=["lin"]),
    )
    sharded = _shard(peft_model)
    assert any(isinstance(p, DTensor) for p in sharded.parameters())
    bundle = _Bundle(sharded)
    strategy = _fsdp_strategy(_cpu_fsdp_context())

    trainer = SimpleNamespace(state_dict=lambda: {"step": 0, "global_step": 0})
    meta = save_training_checkpoint(
        tmp_path,
        trainer=trainer,
        bundle=bundle,
        family="toy",
        progress={"completed_epoch": 0, "next_epoch": 1},
        export_modules={LORA_WEIGHTS_NAME: sharded},
        strategy=strategy,
        is_primary=True,
    )
    assert meta["uses_lora"] is True

    adapter_file = tmp_path / LORA_WEIGHTS_NAME / "adapter_model.safetensors"
    assert adapter_file.exists()
    adapter = load_file(str(adapter_file))
    assert adapter, "adapter export is empty"

    # Every exported tensor equals the gathered full state (PEFT strips the
    # ".default" adapter infix on save, so map keys back before comparing).
    gathered = strategy.export_trainable_state(bundle)["transformer"]
    checkpoint_state = load_training_checkpoint(tmp_path).trainable_state["transformer"]
    assert checkpoint_state.keys() == gathered.keys()
    assert all("lora_" in key for key in checkpoint_state)
    for key, tensor in adapter.items():
        assert isinstance(tensor, torch.Tensor)
        assert not isinstance(tensor, DTensor)
        if ".lora_A." in key:
            gathered_key = key.replace(".lora_A.", ".lora_A.default.")
        else:
            gathered_key = key.replace(".lora_B.", ".lora_B.default.")
        torch.testing.assert_close(tensor, gathered[gathered_key])


def test_export_modules_rejects_sharded_module_outside_bundle(cpu_process_group, tmp_path) -> None:
    """A DTensor-sharded export module with no gathered state must fail loud."""

    from peft import LoraConfig, get_peft_model

    from vrl.trainers.checkpointing import (
        LORA_WEIGHTS_NAME,
        save_training_checkpoint,
    )

    torch.manual_seed(0)
    stray = _shard(
        get_peft_model(
            _ToyTransformer(),
            LoraConfig(r=2, lora_alpha=4, target_modules=["lin"]),
        ),
    )
    bundle = _Bundle(_shard(_ToyTransformer()))
    trainer = SimpleNamespace(state_dict=lambda: {"step": 0, "global_step": 0})

    with pytest.raises(ValueError, match="DTensor-sharded"):
        save_training_checkpoint(
            tmp_path,
            trainer=trainer,
            bundle=bundle,
            family="toy",
            progress={"completed_epoch": 0, "next_epoch": 1},
            export_modules={LORA_WEIGHTS_NAME: stray},
            strategy=_fsdp_strategy(_cpu_fsdp_context()),
            is_primary=True,
        )


# ── optimizer-state resume + DTensor EMA (the lifted §10 gates) ──────────────


def _one_sgd_like_step(net: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad()
    net(torch.randn(2, 4)).sum().backward()
    optimizer.step()


def test_fsdp_optimizer_state_export_is_full_plain_cpu(cpu_process_group) -> None:
    """Exported Adam moments are FQN-keyed full CPU tensors, not DTensor shards."""

    from torch.distributed.tensor import DTensor

    torch.manual_seed(0)
    net = _shard(_ToyTransformer())
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-2)
    _one_sgd_like_step(net, optimizer)

    strategy = _fsdp_strategy(_cpu_fsdp_context(), precision_policy="none")
    state = strategy.export_optimizer_state(net, optimizer)

    moments = state["state"]
    assert moments, "no per-param optimizer state exported"
    global_shapes = {name: p.shape for name, p in net.named_parameters()}
    for fqn, entry in moments.items():
        assert fqn in global_shapes, f"non-FQN optimizer key {fqn!r}"
        for key in ("exp_avg", "exp_avg_sq"):
            moment = entry[key]
            assert isinstance(moment, torch.Tensor)
            assert not isinstance(moment, DTensor)
            assert moment.device.type == "cpu"
            assert moment.shape == global_shapes[fqn]


def test_fsdp_optimizer_state_round_trip(cpu_process_group) -> None:
    """Export -> fresh optimizer -> load reproduces the exact moment tensors."""

    torch.manual_seed(0)
    net = _shard(_ToyTransformer())
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-2)
    _one_sgd_like_step(net, optimizer)

    strategy = _fsdp_strategy(_cpu_fsdp_context(), precision_policy="none")
    exported = strategy.export_optimizer_state(net, optimizer)

    # Resume precondition: the fresh optimizer exists but no training step ran
    # (no pending gradients) — mirror it by clearing the grads the export step
    # left behind.
    for p in net.parameters():
        p.grad = None
    fresh = torch.optim.AdamW(net.parameters(), lr=1e-2)
    strategy.load_optimizer_state(net, fresh, exported)
    reexported = strategy.export_optimizer_state(net, fresh)

    assert reexported["state"].keys() == exported["state"].keys()
    for fqn, entry in exported["state"].items():
        for key, value in entry.items():
            other = reexported["state"][fqn][key]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(other, value)
            else:
                assert other == value


def test_ema_over_dtensor_params_updates_swaps_and_round_trips(cpu_process_group) -> None:
    """EMA shadows DTensor params: step moves shadows, swap/restore is exact,
    and the checkpoint state is full plain tensors that re-shard on load."""

    from torch.distributed.tensor import DTensor

    from vrl.trainers.online.ema import EMAModuleWrapper

    torch.manual_seed(0)
    net = _shard(_ToyTransformer())
    params = [p for p in net.parameters() if p.requires_grad]
    ema = EMAModuleWrapper(params, decay=0.5, update_step_interval=1)
    assert all(isinstance(p, DTensor) for p in ema.ema_parameters)

    # Move the live params, then EMA-step: shadows must move toward them.
    with torch.no_grad():
        for p in params:
            p.add_(1.0)
    before = [p.full_tensor().clone() for p in ema.ema_parameters]
    ema.step(params, optimization_step=0)
    assert ema.has_updates
    after = [p.full_tensor() for p in ema.ema_parameters]
    assert all(not torch.equal(a, b) for a, b in zip(after, before, strict=True))

    # Swap EMA weights in for eval, then restore the originals exactly.
    originals = [p.full_tensor().clone() for p in params]
    ema.copy_ema_to(params, store_temp=True)
    for p, shadow in zip(params, after, strict=True):
        torch.testing.assert_close(p.full_tensor(), shadow)
    ema.copy_temp_to(params)
    for p, original in zip(params, originals, strict=True):
        torch.testing.assert_close(p.full_tensor(), original)

    # Checkpoint round trip: full plain tensors out, re-sharded DTensors in.
    state = ema.state_dict()
    for saved, shadow in zip(state["ema_parameters"], ema.ema_parameters, strict=True):
        assert isinstance(saved, torch.Tensor)
        assert not isinstance(saved, DTensor)
        assert saved.shape == shadow.shape  # DTensor .shape is the global shape

    restored = EMAModuleWrapper(params, decay=0.5, update_step_interval=1)
    restored.load_state_dict(state)
    assert all(isinstance(p, DTensor) for p in restored.ema_parameters)
    for got, expected in zip(restored.ema_parameters, after, strict=True):
        torch.testing.assert_close(got.full_tensor(), expected)
    assert restored.num_updates == ema.num_updates
