"""Driver-model offload moves frozen components too (SPRINT_frozen_component_preservation).

The colocated single-GPU path parks the driver model on CPU during the rollout
window. It must also offload the model's frozen VAE / text-encoders (which
nn.Module.to leaves behind) and restore them after — getattr-guarded so AR
families without the hook are unaffected.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from vrl.rollouts.orchestration.lifecycle import RolloutLifecycle


class _FakeDriverModel:
    def __init__(self, *, with_frozen_hook: bool) -> None:
        self.to_calls: list[Any] = []
        self.frozen_calls: list[Any] = []
        if with_frozen_hook:
            self.move_frozen_components = self._move_frozen

    def to(self, device: Any) -> "_FakeDriverModel":
        self.to_calls.append(device)
        return self

    def _move_frozen(self, device: Any) -> None:
        self.frozen_calls.append(device)


def _lifecycle(model: _FakeDriverModel) -> RolloutLifecycle:
    # A runtime that advertises colocation-driven driver offload; cuda-typed
    # device so should_offload_driver_model_for_rollout() is True on a CPU box.
    runtime = SimpleNamespace(requires_driver_model_offload=True, current_policy_version=None)
    return RolloutLifecycle(
        collector=SimpleNamespace(runtime=runtime),
        model=model,
        device=SimpleNamespace(type="cuda"),
        weight_syncer=None,
        sync_state_getter=None,
        weights_initialized=lambda: True,
        set_weights_initialized=lambda _v: None,
    )


def test_offload_and_restore_move_frozen_components() -> None:
    model = _FakeDriverModel(with_frozen_hook=True)
    lifecycle = _lifecycle(model)
    phases: dict[str, float] = {}

    assert lifecycle.offload_driver_model_for_rollout(phases) is True
    assert model.to_calls == ["cpu"]
    assert model.frozen_calls == ["cpu"]  # frozen components parked, not left resident

    lifecycle.restore_driver_model_after_rollout(phases)
    assert model.to_calls[-1].type == "cuda"
    assert model.frozen_calls[-1].type == "cuda"  # frozen components restored


def test_model_without_frozen_hook_is_unaffected() -> None:
    """AR families register their VAE directly and expose no hook — no crash."""
    model = _FakeDriverModel(with_frozen_hook=False)
    lifecycle = _lifecycle(model)

    assert lifecycle.offload_driver_model_for_rollout({}) is True
    lifecycle.restore_driver_model_after_rollout({})
    assert model.to_calls == ["cpu", model.to_calls[-1]]  # only the model.to moves
    assert model.frozen_calls == []
