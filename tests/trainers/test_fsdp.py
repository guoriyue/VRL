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

from tests.trainers._strategy_policies import (
    Bundle,
    DualStagePolicy,
    FakePolicy,
    ToyBlock,
    ToyTransformer,
)
from vrl.config.schema import FSDPConfig, RootConfig, parse_config
from vrl.trainers.distributed import DistributedTrainingContext, TrainerProcessGroup
from vrl.trainers.fsdp import (
    apply_fsdp,
    build_fsdp_mesh,
    iter_blocks,
    load_full_state_dict,
    mixed_precision_policy,
    normalize_fsdp_parameter_dtype,
    unwrap_module,
)
from vrl.trainers.strategy import (
    FSDPStrategy,
    SingleProcessStrategy,
    build_strategy,
)

# ── fixtures / fakes ────────────────────────────────────────────────────────


def gather_full_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Reference full CPU state dict of a sharded module, on every rank.

    ``cpu_offload`` stays False: with it DCP returns the full state only on rank 0.
    """

    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    from torch.distributed.tensor import DTensor

    state = get_model_state_dict(
        module, options=StateDictOptions(full_state_dict=True, cpu_offload=False)
    )
    return {
        key: (value.full_tensor() if isinstance(value, DTensor) else value).detach().cpu()
        for key, value in state.items()
    }


def _cpu_fsdp_context() -> DistributedTrainingContext:
    return DistributedTrainingContext(
        strategy="fsdp",
        rank=0,
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
        precision_policy=config.precision_policy,
        reshard_after_forward=config.reshard_after_forward,
        cpu_offload=config.cpu_offload,
        shard_trainable_only=config.shard_trainable_only,
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


def _shard(module: nn.Module) -> nn.Module:
    return apply_fsdp(
        module,
        mesh=build_fsdp_mesh(_cpu_fsdp_context()),
        mp_policy=mixed_precision_policy("none"),  # fp32 keeps the CPU path simple
    )


# ── pure helpers (no process group) ─────────────────────────────────────────


@pytest.mark.parametrize("compiled", [False, True])
def test_unwrap_module_peels_real_peft_and_compile(compiled) -> None:
    from peft import LoraConfig, get_peft_model

    base = ToyTransformer()
    wrapped = get_peft_model(base, LoraConfig(r=2, target_modules=["lin"]))
    if compiled:
        wrapped = torch.compile(wrapped)
    assert unwrap_module(wrapped) is base


def test_unwrap_module_preserves_ordinary_children() -> None:
    base = ToyTransformer()
    child = nn.Module()
    child.model = nn.Linear(2, 2)
    base.add_module("_orig_mod", child)
    assert unwrap_module(base) is base


def test_unwrap_module_does_not_invoke_unrelated_get_base_model() -> None:
    base = ToyTransformer()

    def unexpected():
        pytest.fail("ordinary model methods do not declare PEFT wrapper identity")

    base.get_base_model = unexpected
    assert unwrap_module(base) is base


def test_iter_blocks_yields_no_split_modules() -> None:
    net = ToyTransformer()
    blocks = list(iter_blocks(net))
    assert len(blocks) == 2
    assert all(isinstance(b, ToyBlock) for b in blocks)


def test_iter_blocks_fails_without_no_split_modules() -> None:
    net = nn.Linear(4, 4)  # no _no_split_modules
    with pytest.raises(ValueError, match="_no_split_modules"):
        list(iter_blocks(net))


def test_mixed_precision_policy_actor_reduces_in_fp32_without_compute_casts() -> None:
    # Compute stays in the stored dtypes (frozen FP32 pins included), as in
    # single-process replay; only the gradient reduction is promoted.
    policy = mixed_precision_policy("actor")
    assert policy.param_dtype is None
    assert policy.reduce_dtype == torch.float32
    assert not policy.cast_forward_inputs


def test_mixed_precision_policy_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="precision_policy"):
        mixed_precision_policy("fp8")


def test_trainable_only_fsdp_requires_native_precision() -> None:
    with pytest.raises(ValueError, match="requires precision_policy='none'"):
        _fsdp_strategy(_cpu_fsdp_context(), shard_trainable_only=True)


def test_apply_fsdp_leaves_forward_input_dtypes_to_the_model(cpu_process_group) -> None:
    # SANA keeps its timestep in fp32 under fp16 parameters. Single-process
    # replay never casts forward inputs, so FSDP must not either: a root cast
    # moved SANA's FSDP replay 2.4e-3 off its rollout log-prob.
    class TimestepDiT(nn.Module):
        _no_split_modules: ClassVar[list[str]] = ["ToyBlock"]

        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([ToyBlock()])
            self.timestep_dtypes: list[torch.dtype] = []

        def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            self.timestep_dtypes.append(timestep.dtype)
            for block in self.blocks:
                x = block(x)
            return x

    net = TimestepDiT().to(torch.bfloat16)
    apply_fsdp(
        net,
        mesh=build_fsdp_mesh(_cpu_fsdp_context()),
        mp_policy=mixed_precision_policy("actor"),
    )

    net(torch.randn(3, 4, dtype=torch.bfloat16), torch.tensor([500.0]))

    assert net.timestep_dtypes == [torch.float32]


def test_normalize_fsdp_parameter_dtype_casts_mixed_actor_sources(caplog) -> None:
    net = ToyTransformer().to(dtype=torch.bfloat16)
    net.head.to(dtype=torch.float32)

    with caplog.at_level("INFO", logger="vrl.trainers.fsdp"):
        normalize_fsdp_parameter_dtype(
            net,
            torch.bfloat16,
            allow_cast=True,
        )

    assert {parameter.dtype for parameter in net.parameters()} == {torch.bfloat16}
    assert "head.weight:torch.float32" in caplog.text


def test_normalize_fsdp_parameter_dtype_rejects_mixed_native_policy() -> None:
    net = ToyTransformer().to(dtype=torch.bfloat16)
    net.head.to(dtype=torch.float32)

    with pytest.raises(ValueError, match=r"precision_policy='none'.*head\.weight"):
        normalize_fsdp_parameter_dtype(
            net,
            torch.bfloat16,
            allow_cast=False,
        )


@pytest.mark.parametrize("allow_cast", [False, True], ids=["none", "actor"])
def test_fsdp_normalization_preserves_frozen_float32_parameters(allow_cast: bool) -> None:
    net = ToyTransformer().to(dtype=torch.bfloat16)
    net.head.to(dtype=torch.float32).requires_grad_(False)
    with torch.no_grad():
        for parameter in net.head.parameters():
            parameter.add_(0.000123)
    before = [parameter.detach().clone() for parameter in net.head.parameters()]
    pointers = [parameter.data_ptr() for parameter in net.head.parameters()]

    normalize_fsdp_parameter_dtype(net, torch.bfloat16, allow_cast=allow_cast)

    assert {p.dtype for p in net.parameters() if p.requires_grad} == {torch.bfloat16}
    for parameter, expected, pointer in zip(net.head.parameters(), before, pointers, strict=True):
        assert parameter.dtype == torch.float32 and parameter.data_ptr() == pointer
        assert torch.equal(parameter, expected)


def test_native_fsdp_still_rejects_frozen_nonfloating_parameters() -> None:
    net = ToyTransformer().to(dtype=torch.bfloat16)
    net.register_parameter(
        "integer", torch.nn.Parameter(torch.tensor([1], dtype=torch.int64), requires_grad=False)
    )

    with pytest.raises(ValueError, match="non-floating parameters"):
        normalize_fsdp_parameter_dtype(net, torch.bfloat16, allow_cast=False)


# ── real FSDP2 on a single CPU rank ──────────────────────────────────────────


def test_apply_fsdp_shards_params_and_runs_forward_backward(cpu_process_group) -> None:
    from torch.distributed.tensor import DTensor

    net = _shard(ToyTransformer())
    assert any(isinstance(p, DTensor) for p in net.parameters())

    out = net(torch.randn(3, 4))
    out.sum().backward()
    assert all(p.grad is not None for p in net.parameters() if p.requires_grad)


def test_load_full_state_dict_round_trips_into_sharded_module(cpu_process_group) -> None:
    sharded = _shard(ToyTransformer())
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
    net = _shard(ToyTransformer())
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
    ref = ToyTransformer()
    snapshot = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    sharded = ToyTransformer()
    sharded.load_state_dict(snapshot)
    _shard(sharded)

    got = _fsdp_strategy(_cpu_fsdp_context(), precision_policy="none").export_rollout_state(
        Bundle(sharded),
    )
    expected = SingleProcessStrategy().export_rollout_state(Bundle(ref))

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
    ref = ToyTransformer()
    snapshot = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    inner = ToyTransformer()
    inner.load_state_dict(snapshot)
    compiled = torch.compile(inner)  # OptimizedModule with _orig_mod, like production
    _shard(compiled)

    got = _fsdp_strategy(_cpu_fsdp_context(), precision_policy="none").export_rollout_state(
        Bundle(compiled),
    )
    expected = SingleProcessStrategy().export_rollout_state(Bundle(torch.compile(ref)))

    assert all("_orig_mod" not in key for key in got)
    assert got.keys() == expected.keys()
    for key in got:
        assert torch.allclose(got[key], expected[key])


def test_fsdp_rollout_export_filters_frozen_params(cpu_process_group) -> None:
    """Rollout and checkpoint export omit unregistered frozen base state."""
    net = ToyTransformer()
    net.head.requires_grad_(False)  # freeze the non-block head
    net.register_buffer("frozen_cache", torch.ones(4))
    _shard(net)
    strategy = _fsdp_strategy(_cpu_fsdp_context(), precision_policy="none")
    expected = {name for name, parameter in net.named_parameters() if parameter.requires_grad}

    rollout = strategy.export_rollout_state(Bundle(net))
    assert set(rollout) == {f"transformer.{name}" for name in expected}

    checkpoint = strategy.export_checkpoint_state(Bundle(net))["transformer"]
    assert set(checkpoint) == expected
    assert not any("head" in key or "frozen_cache" in key for key in checkpoint)


def test_fsdp_prepare_model_wraps_multi_transformer_model(cpu_process_group) -> None:
    """Dual-stage Wan shards both named roots and writes both aliases back."""
    from torch.distributed.tensor import DTensor

    policy = DualStagePolicy(ToyTransformer())
    out = _fsdp_strategy(_cpu_fsdp_context(), precision_policy="none").prepare_model(policy)

    assert out is policy
    assert policy.set_calls == 1
    assert policy.set_2_calls == 1
    assert any(isinstance(parameter, DTensor) for parameter in policy.transformer.parameters())
    assert any(isinstance(parameter, DTensor) for parameter in policy.transformer_2.parameters())


def test_fsdp_checkpoint_loader_rejects_unselected_legacy_full_state(cpu_process_group) -> None:
    strategy = _fsdp_strategy(_cpu_fsdp_context(), precision_policy="none")
    src_module = ToyTransformer()
    src_module.head.requires_grad_(False)
    with torch.no_grad():
        for parameter in src_module.parameters():
            parameter.fill_(7.0)
    src = _shard(src_module)
    legacy = {"transformer": gather_full_state_dict(src)}

    dst_module = ToyTransformer()
    dst_module.head.requires_grad_(False)
    dst = _shard(dst_module)
    with pytest.raises(ValueError, match="unexpected="):
        strategy.load_checkpoint_state(Bundle(dst), legacy)


def test_fsdp_restore_protocol_normalizes_schema_v1_full_state(
    cpu_process_group, tmp_path
) -> None:
    from vrl.trainers.checkpointing import TrainingCheckpoint

    strategy = _fsdp_strategy(_cpu_fsdp_context(), precision_policy="none")
    src_module = ToyTransformer()
    src_module.head.requires_grad_(False)
    with torch.no_grad():
        for parameter in src_module.parameters():
            parameter.fill_(7.0)
    src = _shard(src_module)
    legacy_full = gather_full_state_dict(src)

    dst_module = ToyTransformer()
    dst_module.head.requires_grad_(False)
    with torch.no_grad():
        for parameter in dst_module.parameters():
            parameter.fill_(3.0)
    dst = _shard(dst_module)
    trainer = SimpleNamespace(
        _strategy=strategy,
        load_state_dict=lambda state: None,
        state_dict=lambda: {},
    )
    checkpoint = TrainingCheckpoint(
        checkpoint_dir=tmp_path,
        checkpoint_path=tmp_path / "checkpoint.pt",
        payload={
            "schema_version": 1,
            "family": "toy",
            "trainer": {},
            "model": {"trainable_modules": {"transformer": legacy_full}},
            "progress": {},
            "rng": {},
        },
        meta={},
    )

    checkpoint.restore_training(
        trainer=trainer,
        bundle=Bundle(dst),
        family="toy",
    )

    restored = gather_full_state_dict(dst)
    for _name, value in restored.items():
        assert torch.allclose(value, torch.full_like(value, 7.0))


def test_fsdp_restore_preflights_global_shape_before_mutation(
    cpu_process_group,
    tmp_path,
) -> None:
    from torch.distributed.tensor import DTensor

    from vrl.trainers.checkpointing import TrainingCheckpoint

    strategy = _fsdp_strategy(_cpu_fsdp_context(), precision_policy="none")
    destination_module = ToyTransformer()
    destination_module.head.requires_grad_(False)
    destination = _shard(destination_module)
    assert any(isinstance(value, DTensor) for value in destination.state_dict().values())
    before = gather_full_state_dict(destination)

    owned_state = strategy.export_checkpoint_state(Bundle(destination))["transformer"]
    owned_state = {name: torch.full_like(value, 7.0) for name, value in owned_state.items()}
    bad_name = sorted(owned_state)[-1]
    owned_state[bad_name] = owned_state[bad_name].new_zeros(
        owned_state[bad_name].numel() + 1,
    )
    identity = {"schema": "toy/v1"}
    checkpoint = TrainingCheckpoint(
        checkpoint_dir=tmp_path,
        checkpoint_path=tmp_path / "checkpoint.pt",
        payload={
            "schema_version": 2,
            "family": "toy",
            "trainer": {},
            "model": {
                "identity": identity,
                "owned_state": {"transformer": owned_state},
            },
            "progress": {},
            "rng": {},
        },
        meta={},
    )

    with pytest.raises(ValueError, match="shape mismatch"):
        checkpoint.restore_model(
            bundle=Bundle(destination),
            family="toy",
            expected_model_identity=identity,
            strategy=strategy,
        )

    restored = gather_full_state_dict(destination)
    assert all(torch.equal(value, before[name]) for name, value in restored.items())


def test_fsdp_load_checkpoint_state_strictly_validates_owned_keys(cpu_process_group) -> None:
    strategy = _fsdp_strategy(_cpu_fsdp_context(), precision_policy="none")
    net = ToyTransformer()
    net.head.requires_grad_(False)
    sharded = _shard(net)
    snapshot = strategy.export_checkpoint_state(Bundle(sharded))
    state = snapshot["transformer"]

    missing = {"transformer": dict(state)}
    missing["transformer"].pop(next(iter(state)))
    with pytest.raises(ValueError, match="missing="):
        strategy.load_checkpoint_state(Bundle(sharded), missing)

    unexpected = {"transformer": {**state, "unknown.weight": torch.ones(1)}}
    with pytest.raises(ValueError, match="unexpected="):
        strategy.load_checkpoint_state(Bundle(sharded), unexpected)


def test_fsdp_actor_prepare_normalizes_mixed_sources_before_first_forward(
    cpu_process_group,
) -> None:
    policy = FakePolicy(ToyTransformer().to(dtype=torch.bfloat16))
    policy.transformer.head.to(dtype=torch.float32)
    policy.transformer.dtype = torch.bfloat16

    FSDPStrategy(
        _cpu_fsdp_context(),
        precision_policy="actor",
        reshard_after_forward=True,
        cpu_offload=False,
    ).prepare_model(policy)
    output = policy.transformer(torch.randn(2, 4, dtype=torch.bfloat16))

    assert output.dtype is torch.bfloat16


def test_native_fsdp_uses_trainable_dtype_when_first_parameter_is_frozen(
    cpu_process_group,
) -> None:
    class MixedNativeToy(torch.nn.Module):
        _no_split_modules = ("Linear",)

        def __init__(self) -> None:
            super().__init__()
            self.frozen_scale = torch.nn.Parameter(torch.tensor([1.000123]), requires_grad=False)
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4, dtype=torch.bfloat16)])

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.blocks[0](inputs) * self.frozen_scale.squeeze(0)

    model = MixedNativeToy()
    inputs = torch.randn(2, 4, dtype=torch.bfloat16)
    expected = model(inputs).detach().clone()
    scale = model.frozen_scale.detach().clone()
    policy = FakePolicy(model)

    _fsdp_strategy(_cpu_fsdp_context(), precision_policy="none").prepare_model(policy)

    assert torch.equal(policy.transformer(inputs), expected)
    assert policy.transformer.frozen_scale.dtype == torch.float32
    assert torch.equal(policy.transformer.frozen_scale.full_tensor(), scale)


def test_wan_fsdp_replay_build_defers_full_gpu_move_until_sharding(
    cpu_process_group,
    monkeypatch,
) -> None:
    """Wan replay attaches LoRA on CPU before FSDP2 owns materialization."""
    from omegaconf import OmegaConf
    from torch.distributed.tensor import DTensor

    from vrl.config.precision import PrecisionPolicy
    from vrl.models.families.registry import get_model_family_entry
    from vrl.models.steps.denoise import build as denoise_build

    class _TrackingWanTransformer(ToyTransformer):
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
        lambda _build, _class_name, **_kwargs: transformer,
    )
    monkeypatch.setattr(
        denoise_build,
        "load_diffusers_scheduler",
        lambda _build, _class_name: object(),
    )
    monkeypatch.setattr(
        "diffusers.DiffusionPipeline.load_config",
        lambda *_args, **_kwargs: {
            "boundary_ratio": None,
            "expand_timesteps": False,
        },
    )
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "wan_2_1_i2v",
                "path": "fake/Wan2.1-I2V",
                "revision": "a" * 40,
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
    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)
    entry = get_model_family_entry("wan_2_1_i2v")
    build = entry.resolve_model_build(
        root,
        torch.device("cpu"),
        precision=precision,
        for_rollout=False,
    )

    bundle = entry.build_replay(build, expected_identity=None)
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


def test_fsdp_prepare_model_without_a_process_group_is_refused(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="open TrainerProcessGroup"):
        _fsdp_strategy(_cpu_fsdp_context(), precision_policy="none").prepare_model(
            FakePolicy(ToyTransformer())
        )


def test_process_group_binds_resolved_device_after_per_rank_mask(monkeypatch, tmp_path) -> None:
    """Physical rank 3 still owns logical cuda:0 after launcher masking."""

    import os

    import torch.distributed as dist

    context = DistributedTrainingContext(
        strategy="fsdp",
        rank=3,
        world_size=4,
        device=torch.device("cuda:0"),
    )
    bound_devices: list[torch.device] = []
    init_calls: list[dict[str, object]] = []
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "set_device", bound_devices.append)
    monkeypatch.setattr(dist, "init_process_group", lambda **kwargs: init_calls.append(kwargs))
    monkeypatch.setattr(dist, "new_group", lambda **kwargs: None)
    # Opening an NCCL group writes flight-recorder settings into the environment.
    monkeypatch.setattr(os, "environ", dict(os.environ))

    TrainerProcessGroup.open(context, output_dir=tmp_path).shutdown()

    assert bound_devices == [torch.device("cuda:0")]
    assert init_calls == [
        {"backend": "nccl", "rank": 3, "world_size": 4, "device_id": torch.device("cuda:0")}
    ]


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
        world_size=1,
        device=torch.device("cpu"),
    )
    assert isinstance(build_strategy(RootConfig(), ctx), SingleProcessStrategy)


def test_build_strategy_rejects_config_context_mismatch() -> None:
    with pytest.raises(ValueError, match="strategy mismatch"):
        build_strategy(RootConfig(), _cpu_fsdp_context())


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

    from vrl.models.parking import TrainingMemoryState

    strategy = _fsdp_strategy(_cpu_fsdp_context())
    # This rank parks cleanly; the agreement reports that a peer did not.
    strategy.collectives.succeeded = lambda succeeded: False

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
    assert strategy._parking is None


def test_fsdp_shutdown_leaves_the_process_group_to_the_process(monkeypatch) -> None:
    import torch.distributed as dist

    from vrl.models.parking import TrainingMemoryState

    strategy = _fsdp_strategy(_cpu_fsdp_context())
    state = TrainingMemoryState(nn.Linear(2, 2), None, None, None, None, torch.device("cpu"))
    strategy.park_training_state(state)
    destroyed = []
    failure = RuntimeError("restore failed")

    def fail_restore(_state):
        raise failure

    monkeypatch.setattr(strategy, "restore_training_state", fail_restore)
    monkeypatch.setattr(dist, "destroy_process_group", lambda *a, **k: destroyed.append(a))
    with pytest.raises(RuntimeError) as caught:
        strategy.shutdown()
    assert caught.value is failure
    assert destroyed == []


def test_build_strategy_fsdp_accepts_rollout_scoped_compile() -> None:
    # scope=rollout keeps the FSDP2 replay policy eager, so the §10 gate must
    # not fire: only a compile that binds the replay role is unsound here.
    assert isinstance(
        build_strategy(
            _strategy_config(
                "fsdp",
                model={"torch_compile": {"enable": True, "scope": "rollout"}},
            ),
            _cpu_fsdp_context(),
        ),
        FSDPStrategy,
    )


# ── gathered HF-adapter export (save_pretrained under FSDP2) ─────────────────


def test_fsdp_adapter_export_writes_gathered_hf_adapter(cpu_process_group, tmp_path) -> None:
    """save_pretrained under FSDP2 serializes the gathered adapter, not shards.

    The online recipe passes adapter_exports under fsdp too;
    save_training_checkpoint must detect the DTensor-sharded module and feed
    save_pretrained the full state it already gathered for checkpoint.pt.
    """

    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file
    from torch.distributed.tensor import DTensor

    from vrl.trainers.checkpointing import (
        LORA_WEIGHTS_NAME,
        AdapterExport,
        TrainingCheckpoint,
        save_training_checkpoint,
    )

    torch.manual_seed(0)
    peft_model = get_peft_model(
        ToyTransformer(),
        LoraConfig(r=2, lora_alpha=4, init_lora_weights="gaussian", target_modules=["lin"]),
    )
    sharded = _shard(peft_model)
    assert any(isinstance(p, DTensor) for p in sharded.parameters())
    bundle = Bundle(sharded)
    strategy = _fsdp_strategy(_cpu_fsdp_context())

    trainer = SimpleNamespace(state_dict=lambda: {"step": 0, "global_step": 0})
    meta = save_training_checkpoint(
        tmp_path,
        trainer=trainer,
        bundle=bundle,
        family="toy",
        model_identity={"schema": "toy/v1"},
        progress={"completed_epoch": 0, "next_epoch": 1},
        adapter_exports={LORA_WEIGHTS_NAME: AdapterExport(sharded)},
        strategy=strategy,
    )
    assert meta["uses_lora"] is True

    adapter_file = tmp_path / LORA_WEIGHTS_NAME / "adapter_model.safetensors"
    assert adapter_file.exists()
    adapter = load_file(str(adapter_file))
    assert adapter, "adapter export is empty"

    # Every exported tensor equals the gathered full state (PEFT strips the
    # ".default" adapter infix on save, so map keys back before comparing).
    gathered = strategy.export_checkpoint_state(bundle)["transformer"]
    checkpoint_state = TrainingCheckpoint.load(tmp_path).checkpoint_state["transformer"]
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


def test_adapter_export_rejects_module_outside_bundle(cpu_process_group, tmp_path) -> None:
    """A DTensor-sharded export module with no gathered state must fail loud."""

    from peft import LoraConfig, get_peft_model

    from vrl.trainers.checkpointing import (
        LORA_WEIGHTS_NAME,
        AdapterExport,
        save_training_checkpoint,
    )

    torch.manual_seed(0)
    stray = _shard(
        get_peft_model(
            ToyTransformer(),
            LoraConfig(r=2, lora_alpha=4, target_modules=["lin"]),
        ),
    )
    bundle = Bundle(_shard(ToyTransformer()))
    trainer = SimpleNamespace(state_dict=lambda: {"step": 0, "global_step": 0})

    with pytest.raises(ValueError, match="exactly one bundle checkpoint root"):
        save_training_checkpoint(
            tmp_path,
            trainer=trainer,
            bundle=bundle,
            family="toy",
            model_identity={"schema": "toy/v1"},
            progress={"completed_epoch": 0, "next_epoch": 1},
            adapter_exports={LORA_WEIGHTS_NAME: AdapterExport(stray)},
            strategy=_fsdp_strategy(_cpu_fsdp_context()),
        )


# ── optimizer-state resume + DTensor EMA (the lifted §10 gates) ──────────────


def _one_sgd_like_step(net: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad()
    net(torch.randn(2, 4)).sum().backward()
    optimizer.step()


def test_ema_over_dtensor_params_updates_swaps_and_round_trips(cpu_process_group) -> None:
    """EMA shadows DTensor params: step moves shadows, swap/restore is exact,
    and the checkpoint state is full plain tensors that re-shard on load."""

    from torch.distributed.tensor import DTensor

    from vrl.trainers.online.ema import EMAWeights

    torch.manual_seed(0)
    net = _shard(ToyTransformer())
    params = [p for p in net.parameters() if p.requires_grad]
    ema = EMAWeights(params, decay=0.5, update_step_interval=1)
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
    ema.copy_ema_to(params)
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

    restored = EMAWeights(params, decay=0.5, update_step_interval=1)
    restored.load_state_dict(state)
    assert all(isinstance(p, DTensor) for p in restored.ema_parameters)
    for got, expected in zip(restored.ema_parameters, after, strict=True):
        torch.testing.assert_close(got.full_tensor(), expected)
    assert restored.num_updates == ema.num_updates


@pytest.mark.parametrize("declaration", ["ToyBlock", [1], [""]])
def test_iter_blocks_rejects_malformed_declaration(declaration) -> None:
    model = ToyTransformer()
    model._no_split_modules = declaration
    with pytest.raises(ValueError, match="block class names"):
        list(iter_blocks(model))


def test_apply_fsdp_rejects_unmatched_blocks_before_sharding(monkeypatch) -> None:
    import torch.distributed.fsdp as torch_fsdp

    model = ToyTransformer()
    model._no_split_modules = ["MisspelledBlock"]
    calls = []
    monkeypatch.setattr(torch_fsdp, "fully_shard", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="no modules match declared FSDP block classes"):
        apply_fsdp(model, mesh=None, mp_policy=mixed_precision_policy("none"))
    assert calls == []


def test_fsdp_rejects_undeclared_mixed_dtype_without_casting() -> None:
    model = ToyTransformer().to(torch.float32)
    model.head.to(torch.bfloat16)
    original = {name: parameter.dtype for name, parameter in model.named_parameters()}
    with pytest.raises(ValueError, match="mixed parameter dtypes"):
        _fsdp_strategy(_cpu_fsdp_context(), precision_policy="actor").prepare_model(
            FakePolicy(model)
        )
    assert {name: parameter.dtype for name, parameter in model.named_parameters()} == original


def test_fsdp_rejects_invalid_declared_dtype_instead_of_using_first_parameter() -> None:
    model = ToyTransformer()
    model.dtype = "bf16"
    with pytest.raises(TypeError, match=r"dtype must be a torch\.dtype"):
        _fsdp_strategy(_cpu_fsdp_context(), precision_policy="actor").prepare_model(
            FakePolicy(model)
        )
