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
from typing import ClassVar

import pytest
import torch
from torch import nn

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
        distributed=True,
        rank=0,
        local_rank=0,
        world_size=1,
        is_primary=True,
        device=torch.device("cpu"),
    )


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
    """Wan-style policy with a second trainable transformer the applier can't wrap."""

    @property
    def trainable_modules(self) -> dict[str, nn.Module]:
        return {"transformer": self.transformer, "transformer_2": self.transformer}


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


def test_mixed_precision_policy_actor_uses_bf16_params_fp32_reduce() -> None:
    policy = mixed_precision_policy("actor")
    assert policy.param_dtype == torch.bfloat16
    assert policy.reduce_dtype == torch.float32


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
    returned = FSDPStrategy(_cpu_fsdp_context(), precision_policy="none").clip_grad_norm(
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

    got = FSDPStrategy(_cpu_fsdp_context(), precision_policy="none").export_rollout_state(
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

    got = FSDPStrategy(_cpu_fsdp_context(), precision_policy="none").export_rollout_state(
        _Bundle(compiled),
    )
    expected = SingleProcessStrategy().export_rollout_state(_Bundle(torch.compile(ref)))

    assert all("_orig_mod" not in key for key in got)
    assert got.keys() == expected.keys()
    for key in got:
        assert torch.allclose(got[key], expected[key])


def test_fsdp_rollout_export_filters_frozen_params(cpu_process_group) -> None:
    """export_rollout_state drops frozen params (the LoRA shape) even on the
    sharded DTensor path; export_trainable_state (checkpoint) keeps them."""
    net = _ToyTransformer()
    net.head.requires_grad_(False)  # freeze the non-block head
    _shard(net)
    strategy = FSDPStrategy(_cpu_fsdp_context(), precision_policy="none")

    rollout = strategy.export_rollout_state(_Bundle(net))
    assert rollout  # blocks are still trainable
    assert not any("head" in key for key in rollout)

    checkpoint = strategy.export_trainable_state(_Bundle(net))["transformer"]
    assert any("head" in key for key in checkpoint)  # checkpoint keeps frozen state


def test_fsdp_prepare_model_rejects_multi_transformer_model(cpu_process_group) -> None:
    """Dual-stage Wan (transformer + transformer_2) must fail-fast, not silently
    shard one handle and leave the other replicated."""
    policy = _DualStagePolicy(_ToyTransformer())
    with pytest.raises(NotImplementedError, match="Multi-transformer"):
        FSDPStrategy(_cpu_fsdp_context(), precision_policy="none").prepare_model(policy)


def test_fsdp_export_then_load_trainable_state_round_trip(cpu_process_group) -> None:
    strategy = FSDPStrategy(_cpu_fsdp_context(), precision_policy="none")
    src = _shard(_ToyTransformer())
    with torch.no_grad():
        load_full_state_dict(src, {k: torch.full_like(v, 3.0) for k, v in gather_full_state_dict(src).items()})

    snapshot = strategy.export_trainable_state(_Bundle(src))
    assert set(snapshot) == {"transformer"}

    dst = _shard(_ToyTransformer())
    strategy.load_trainable_state(_Bundle(dst), snapshot)
    for value in gather_full_state_dict(dst).values():
        assert torch.allclose(value, torch.full_like(value, 3.0))


def test_fsdp_prepare_model_wraps_diffusion_handle(cpu_process_group) -> None:
    from torch.distributed.tensor import DTensor

    policy = _FakePolicy(_ToyTransformer())
    out = FSDPStrategy(_cpu_fsdp_context(), precision_policy="none").prepare_model(policy)

    assert out is policy
    assert policy.set_calls == 1
    assert any(isinstance(p, DTensor) for p in policy.transformer.parameters())


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
    FSDPStrategy(_cpu_fsdp_context(), precision_policy="none").prepare_model(policy)

    assert calls == [("fsdp", "gloo")]


# ── prepare_model family gate / build_strategy §10 gates (no process group) ──


def test_fsdp_prepare_model_rejects_model_without_transformer_handle() -> None:
    class _ARLikePolicy:
        pass

    with pytest.raises(NotImplementedError, match="trainable roots"):
        FSDPStrategy(_cpu_fsdp_context(), precision_policy="none").prepare_model(_ARLikePolicy())


def test_build_strategy_single_process_returns_single_process() -> None:
    ctx = DistributedTrainingContext(
        strategy="single_process",
        distributed=False,
        rank=0,
        local_rank=0,
        world_size=1,
        is_primary=True,
        device=torch.device("cpu"),
    )
    assert isinstance(build_strategy({}, ctx), SingleProcessStrategy)


def test_build_strategy_fsdp_reads_knobs() -> None:
    cfg = {
        "distributed": {
            "training": {
                "fsdp": {
                    "mesh": ["dp_shard"],
                    "precision_policy": "none",
                    "reshard_after_forward": False,
                },
            },
        },
    }
    strategy = build_strategy(cfg, _cpu_fsdp_context())
    assert isinstance(strategy, FSDPStrategy)
    assert strategy._precision_policy == "none"
    assert strategy._reshard_after_forward is False


def test_build_strategy_fsdp_rejects_ema() -> None:
    with pytest.raises(NotImplementedError, match="EMA"):
        build_strategy({"actor": {"ema": {"enable": True}}}, _cpu_fsdp_context())


def test_build_strategy_fsdp_rejects_optimizer_resume() -> None:
    with pytest.raises(NotImplementedError, match="resume"):
        build_strategy({"trainer": {"resume_from": "/ckpt/checkpoint-10"}}, _cpu_fsdp_context())


def test_build_strategy_fsdp_rejects_train_compile() -> None:
    # torch.compile (inductor) is unsound with FSDP2 reshard-after-forward all-gathers;
    # the build_strategy §10 gate must reject it.
    with pytest.raises(NotImplementedError, match="torch_compile"):
        build_strategy(
            {"model": {"torch_compile": {"enable": True}}}, _cpu_fsdp_context()
        )
