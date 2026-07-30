"""SingleProcessStrategy parity (readiness P3).

The strategy seam moves the existing backward / clip / state-export logic behind
``Strategy``. These lock that the moved methods reproduce the underlying
helpers exactly, so installing the seam changes nothing for a single-GPU run.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vrl.models.interfaces.runtime import register_checkpoint_owned_state
from vrl.trainers.online.ema import EMAModuleWrapper
from vrl.trainers.optimizer import FP32MasterWeightOptimizer
from vrl.trainers.strategy import SingleProcessStrategy, TrainingMemoryState
from vrl.trainers.weight_sync import build_trainable_state_sync_getter


class _Bundle:
    def __init__(self, module: nn.Module | None = None) -> None:
        self.trainable_modules = {"adapter": module or nn.Linear(2, 1, bias=False)}


def test_default_context_is_single_process() -> None:
    ctx = SingleProcessStrategy().context
    assert (ctx.strategy, ctx.rank, ctx.world_size, ctx.is_primary) == (
        "single_process",
        0,
        1,
        True,
    )


def test_prepare_model_is_identity_for_single_process() -> None:
    model = nn.Linear(3, 1)
    assert SingleProcessStrategy().prepare_model(model) is model
    x = torch.randn(4, 3, generator=torch.Generator().manual_seed(0))
    ref = nn.Linear(3, 1)
    strat = nn.Linear(3, 1)
    strat.load_state_dict(ref.state_dict())

    ref(x).sum().backward()
    SingleProcessStrategy().backward(strat(x).sum())

    for p_ref, p_strat in zip(ref.parameters(), strat.parameters(), strict=True):
        assert torch.equal(p_ref.grad, p_strat.grad)


def test_backward_scales_loss_when_grad_scaler_present() -> None:
    calls: list[str] = []

    class _Scaler:
        def scale(self, loss: torch.Tensor) -> torch.Tensor:
            calls.append("scale")
            return loss

    leaf = nn.Linear(1, 1)
    SingleProcessStrategy().backward(leaf(torch.ones(1, 1)).sum(), grad_scaler=_Scaler())

    assert calls == ["scale"]


def test_clip_grad_norm_matches_torch_in_place() -> None:
    ref = nn.Linear(4, 4)
    strat = nn.Linear(4, 4)
    strat.load_state_dict(ref.state_dict())
    for p in ref.parameters():
        p.grad = torch.ones_like(p)
    for p in strat.parameters():
        p.grad = torch.ones_like(p)

    ref_norm = float(nn.utils.clip_grad_norm_(ref.parameters(), 0.5))
    strat_norm = SingleProcessStrategy().clip_grad_norm(strat.parameters(), 0.5)

    assert strat_norm == ref_norm
    for p_ref, p_strat in zip(ref.parameters(), strat.parameters(), strict=True):
        assert torch.equal(p_ref.grad, p_strat.grad)  # clipped identically


def test_export_rollout_state_matches_helper() -> None:
    bundle = _Bundle()
    got = SingleProcessStrategy().export_rollout_state(bundle)
    expected = build_trainable_state_sync_getter(bundle)()

    assert got.keys() == expected.keys()
    for key in got:
        assert torch.equal(got[key], expected[key])


def test_export_rollout_state_is_detached_cpu_snapshot_without_live_alias() -> None:
    bundle = _Bundle()
    live = bundle.trainable_modules["adapter"].weight
    snapshot = SingleProcessStrategy().export_rollout_state(bundle)
    exported = snapshot["adapter.weight"]

    assert exported.device.type == "cpu"
    assert exported.requires_grad is False
    assert exported.data_ptr() != live.data_ptr()
    before = exported.clone()
    with torch.no_grad():
        live.add_(10)
    assert torch.equal(exported, before)


class _CountingLinear(nn.Linear):
    def __init__(self) -> None:
        super().__init__(2, 1, bias=False)
        self.to_calls: list[torch.device] = []

    def to(self, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        self.to_calls.append(torch.device(device))
        return super().to(*args, **kwargs)


@pytest.mark.real_cover(
    "tests/trainers/test_strategy.py"
    "::test_cuda_training_state_parking_round_trip_preserves_all_live_state",
    why=(
        "parking's destination IS cpu on this lane, so _move_tensor_tree_in_place is a no-op "
        "and no CPU assertion can distinguish a branch that ran from one that was skipped; only "
        "the gpu twin can watch the model, grads, optimizer, EMA and GradScaler tensors "
        "actually land on cpu and come back"
    ),
)
def test_training_state_parking_is_idempotent_and_deduplicates_model_identity() -> None:
    """Exactly one move per park and per restore, however many times each is called.

    ``model`` and ``ref_model`` are the same object, so a park that moved each
    entry separately — or a second park that re-parked already-parked state —
    shows up as extra ``to()`` calls beyond the one park + one restore asserted
    below, and the grads/optimizer moments must survive the round trip untouched.
    """
    model = _CountingLinear()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    # A real CPU GradScaler left exactly where a training step leaves it: the
    # first scale/step/update cycle gives `_scale` and `_growth_tracker` real
    # tensors, and the second scale/backward/unscale_ populates
    # `_per_optimizer_states`. Those are the three private torch attributes
    # parking reads by name (vrl/trainers/strategy.py) — a hand-written namespace
    # could only restate the names it was copied from, so the assertions below
    # are the only thing in the default lane that still checks torch has them.
    scaler = torch.amp.GradScaler("cpu", init_scale=8.0, growth_interval=2)
    scaler.scale(model(torch.ones(1, 2)).sum()).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(model(torch.full((1, 2), 2.0)).sum()).backward()
    scaler.unscale_(optimizer)
    assert isinstance(scaler._scale, torch.Tensor)
    assert isinstance(scaler._growth_tracker, torch.Tensor)
    assert scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
    grad_before = model.weight.grad.detach().clone()
    optimizer_before = _tensor_values(optimizer.state)
    ema = EMAModuleWrapper(model.parameters(), decay=0.9, device=torch.device("cpu"))
    state = TrainingMemoryState(
        model=model,
        ref_model=model,
        optimizer=optimizer,
        ema=ema,
        grad_scaler=scaler,
        device=torch.device("cpu"),
    )
    strategy = SingleProcessStrategy()

    strategy.park_training_state(state)
    strategy.park_training_state(state)
    strategy.restore_training_state(state)
    strategy.restore_training_state(state)

    assert model.to_calls == [torch.device("cpu"), torch.device("cpu")]
    assert torch.equal(model.weight.grad, grad_before)
    _assert_tensor_values_equal(optimizer.state, optimizer_before)


def test_terminal_shutdown_can_abandon_parked_state_without_gpu_restore() -> None:
    model = _CountingLinear()
    state = TrainingMemoryState(
        model=model,
        ref_model=None,
        optimizer=None,
        ema=None,
        grad_scaler=None,
        device=torch.device("cpu"),
    )
    strategy = SingleProcessStrategy()

    strategy.park_training_state(state)
    strategy.shutdown(restore_parked=False)
    strategy.restore_training_state(state)

    assert model.to_calls == [torch.device("cpu")]


def test_training_parking_cache_release_failure_rolls_back_and_does_not_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vrl.trainers.strategy as strategy_module

    model = _CountingLinear()
    state = TrainingMemoryState(
        model=model,
        ref_model=None,
        optimizer=None,
        ema=None,
        grad_scaler=None,
        device=torch.device("cpu"),
    )
    strategy = SingleProcessStrategy()
    monkeypatch.setattr(
        strategy_module,
        "_release_training_cuda_memory",
        lambda: (_ for _ in ()).throw(RuntimeError("empty cache failed")),
    )

    with pytest.raises(RuntimeError, match="empty cache failed"):
        strategy.park_training_state(state)

    assert model.to_calls == [torch.device("cpu"), torch.device("cpu")]
    strategy.restore_training_state(state)
    assert model.to_calls == [torch.device("cpu"), torch.device("cpu")]


def test_online_trainer_resolves_lazy_training_memory_state_on_every_phase() -> None:
    from vrl.trainers.online.trainer import OnlineTrainer

    trainer = object.__new__(OnlineTrainer)
    trainer.model = nn.Linear(2, 1)
    trainer.ref_model = nn.Linear(2, 1)
    trainer._optimizer = None
    trainer._ema = None
    trainer._grad_scaler = None
    trainer.device = torch.device("cpu")

    first = trainer._training_memory_state()
    assert first.optimizer is None
    assert first.ema is None

    trainer._optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.1)
    trainer._ema = EMAModuleWrapper(trainer.model.parameters(), device=torch.device("cpu"))
    second = trainer._training_memory_state()

    assert second.optimizer is trainer._optimizer
    assert second.ema is trainer._ema
    assert second is not first


@pytest.mark.gpu
def test_cuda_training_state_parking_round_trip_preserves_all_live_state() -> None:
    device = torch.device("cuda:0")
    model = nn.Linear(4, 2).to(device)
    ref_model = nn.Linear(4, 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    ema = EMAModuleWrapper(model.parameters(), decay=0.9, device=device)
    scaler = torch.amp.GradScaler("cuda")

    optimizer.zero_grad(set_to_none=True)
    first_loss = model(torch.ones(2, 4, device=device)).square().mean()
    scaler.scale(first_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    ema.step(model.parameters(), 0)

    optimizer.zero_grad(set_to_none=True)
    live_loss = model(torch.full((2, 4), 0.5, device=device)).sum()
    scaler.scale(live_loss).backward()
    scaler.unscale_(optimizer)

    model_before = [parameter.detach().cpu().clone() for parameter in model.parameters()]
    ref_before = [parameter.detach().cpu().clone() for parameter in ref_model.parameters()]
    grads_before = [parameter.grad.detach().cpu().clone() for parameter in model.parameters()]
    optimizer_before = _tensor_values(optimizer.state)
    ema_before = _tensor_values(ema.ema_parameters)
    scaler_before = _tensor_values(
        (scaler._scale, scaler._growth_tracker, scaler._per_optimizer_states),
    )
    state = TrainingMemoryState(
        model=model,
        ref_model=ref_model,
        optimizer=optimizer,
        ema=ema,
        grad_scaler=scaler,
        device=device,
    )
    strategy = SingleProcessStrategy()

    strategy.park_training_state(state)

    assert all(tensor.device.type == "cpu" for tensor in _module_tensors(model, ref_model))
    assert all(tensor.device.type == "cpu" for tensor in _tree_tensors(optimizer.state))
    assert all(tensor.device.type == "cpu" for tensor in _tree_tensors(ema.ema_parameters))
    assert all(
        tensor.device.type == "cpu"
        for tensor in _tree_tensors(
            (scaler._scale, scaler._growth_tracker, scaler._per_optimizer_states),
        )
    )

    strategy.restore_training_state(state)

    assert all(tensor.device.type == "cuda" for tensor in _module_tensors(model, ref_model))
    _assert_tensor_values_equal(model.parameters(), model_before)
    _assert_tensor_values_equal(ref_model.parameters(), ref_before)
    _assert_tensor_values_equal((parameter.grad for parameter in model.parameters()), grads_before)
    _assert_tensor_values_equal(optimizer.state, optimizer_before)
    _assert_tensor_values_equal(ema.ema_parameters, ema_before)
    _assert_tensor_values_equal(
        (scaler._scale, scaler._growth_tracker, scaler._per_optimizer_states),
        scaler_before,
    )

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


@pytest.mark.gpu
def test_bitsandbytes_adamw8bit_state_survives_cuda_parking_round_trip() -> None:
    """The production 8-bit optimizer can step again after a phase lease."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for bitsandbytes optimizer state")
    bitsandbytes = pytest.importorskip("bitsandbytes")
    device = torch.device("cuda:0")
    model = nn.Linear(128, 128, bias=False).to(device)
    optimizer = bitsandbytes.optim.AdamW8bit(model.parameters(), lr=0.01)

    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(2, 128, device=device)).square().mean().backward()
    optimizer.step()
    state_tensors = list(_tree_tensors(optimizer.state))
    assert state_tensors
    assert all(tensor.device.type == "cuda" for tensor in state_tensors)

    state = TrainingMemoryState(
        model=model,
        ref_model=None,
        optimizer=optimizer,
        ema=None,
        grad_scaler=None,
        device=device,
    )
    strategy = SingleProcessStrategy()
    strategy.park_training_state(state)

    assert all(tensor.device.type == "cpu" for tensor in _tree_tensors(optimizer.state))

    strategy.restore_training_state(state)

    assert all(tensor.device.type == "cuda" for tensor in _tree_tensors(optimizer.state))
    optimizer.zero_grad(set_to_none=True)
    model(torch.full((2, 128), 0.5, device=device)).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


@pytest.mark.gpu
def test_fp32_master_parameters_and_live_grads_survive_cuda_parking() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FP32 master parking")
    bitsandbytes = pytest.importorskip("bitsandbytes")
    device = torch.device("cuda:0")
    model = nn.Linear(128, 128, bias=False).to(device=device, dtype=torch.float16)
    optimizer = FP32MasterWeightOptimizer(
        model.parameters(),
        lambda parameters: bitsandbytes.optim.AdamW8bit(parameters, lr=0.01),
    )
    scaler = torch.amp.GradScaler("cuda")

    first_loss = (
        model(
            torch.ones(2, 128, device=device, dtype=torch.float16),
        )
        .square()
        .mean()
    )
    scaler.scale(first_loss).backward()
    optimizer.prepare_gradients()
    scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Keep a live prepared master gradient across the phase handoff. This is an
    # exceptional but supported state (for example cancellation between
    # backward and step) and must not pin the rollout GPU.
    live_loss = model(
        torch.full((2, 128), 0.5, device=device, dtype=torch.float16),
    ).sum()
    scaler.scale(live_loss).backward()
    optimizer.prepare_gradients()
    scaler.unscale_(optimizer)
    masters = list(optimizer.parameters())
    assert all(parameter.device.type == "cuda" for parameter in masters)
    assert all(parameter.grad is not None for parameter in masters)

    state = TrainingMemoryState(
        model=model,
        ref_model=None,
        optimizer=optimizer,
        ema=None,
        grad_scaler=scaler,
        device=device,
    )
    strategy = SingleProcessStrategy()
    strategy.park_training_state(state)

    assert all(parameter.device.type == "cpu" for parameter in masters)
    assert all(
        parameter.grad is not None and parameter.grad.device.type == "cpu" for parameter in masters
    )
    assert all(tensor.device.type == "cpu" for tensor in _tree_tensors(optimizer.state))

    strategy.restore_training_state(state)

    assert all(parameter.device.type == "cuda" for parameter in masters)
    assert all(
        parameter.grad is not None and parameter.grad.device.type == "cuda"
        for parameter in masters
    )
    assert all(tensor.device.type == "cuda" for tensor in _tree_tensors(optimizer.state))
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def _module_tensors(*modules: nn.Module):
    for module in modules:
        for parameter in module.parameters():
            yield parameter
            if parameter.grad is not None:
                yield parameter.grad
        yield from module.buffers()


def _tree_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _tree_tensors(child)
    elif isinstance(value, (list, tuple)) or (
        hasattr(value, "__iter__") and not isinstance(value, (str, bytes))
    ):
        for child in value:
            yield from _tree_tensors(child)


def _tensor_values(value):
    return [tensor.detach().cpu().clone() for tensor in _tree_tensors(value)]


def _assert_tensor_values_equal(value, expected) -> None:
    actual = _tensor_values(value)
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert torch.equal(got, want)


def test_export_and_load_checkpoint_state_round_trip() -> None:
    strat = SingleProcessStrategy()
    src = _Bundle()
    with torch.no_grad():
        src.trainable_modules["adapter"].weight.fill_(3.0)
    snapshot = strat.export_checkpoint_state(src)
    assert set(snapshot) == {"adapter"}

    dst = _Bundle()
    strat.load_checkpoint_state(dst, snapshot)
    assert torch.equal(
        dst.trainable_modules["adapter"].weight,
        src.trainable_modules["adapter"].weight,
    )


def test_checkpoint_and_rollout_state_have_distinct_ownership() -> None:
    module = nn.Linear(2, 1, bias=False)
    module.register_buffer("previous", torch.ones(1))
    module.register_buffer("cache", torch.zeros(1))
    register_checkpoint_owned_state(module, ["previous"])
    bundle = _Bundle(module)
    strategy = SingleProcessStrategy()

    checkpoint = strategy.export_checkpoint_state(bundle)["adapter"]
    rollout = strategy.export_rollout_state(bundle)

    assert set(checkpoint) == {"weight", "previous"}
    assert set(rollout) == {"adapter.weight"}
    assert "adapter.previous" not in rollout
    assert "cache" not in checkpoint


def test_barrier_is_noop() -> None:
    assert SingleProcessStrategy().barrier() is None


def test_shutdown_is_noop() -> None:
    assert SingleProcessStrategy().shutdown() is None
