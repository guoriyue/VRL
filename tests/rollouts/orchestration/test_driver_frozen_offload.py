"""Driver-model offload moves frozen components too (SPRINT_frozen_component_preservation).

The colocated single-GPU path parks the driver model on CPU during the rollout
window. It must also offload the model's frozen VAE / text-encoders (which
nn.Module.to leaves behind) and restore them after — getattr-guarded so AR
families without the hook are unaffected.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from vrl.rollouts.orchestration.rollout_runtime import RolloutRuntimeCoordinator
from vrl.rollouts.stats import RolloutStats
from vrl.trainers.strategy import SingleProcessStrategy, TrainingMemoryState


class _FakeDriverModel:
    def __init__(self, *, with_frozen_hook: bool) -> None:
        self.to_calls: list[Any] = []
        self.frozen_calls: list[Any] = []
        if with_frozen_hook:
            self.move_frozen_components = self._move_frozen

    def to(self, device: Any) -> _FakeDriverModel:
        self.to_calls.append(device)
        return self

    def _move_frozen(self, device: Any) -> None:
        self.frozen_calls.append(device)


def _coordinator(model: _FakeDriverModel) -> RolloutRuntimeCoordinator:
    # A runtime that advertises colocation-driven driver offload; cuda-typed
    # device so should_offload_driver_model_for_rollout() is True on a CPU box.
    runtime = SimpleNamespace(requires_driver_model_offload=True, current_policy_version=None)
    strategy = SingleProcessStrategy()
    state = TrainingMemoryState(
        model=model,  # type: ignore[arg-type]
        ref_model=None,
        optimizer=None,
        ema=None,
        grad_scaler=None,
        device=torch.device("cuda"),
    )

    class _CollectorControl:
        def __init__(self, runtime: Any) -> None:
            self.runtime = runtime
            self.requires_runtime_offload_before_reward = False
            self.requires_driver_model_offload_for_reward = False
            self.supports_reward_generation_overlap = False

        async def activate_runtime(self) -> None:
            return None

        async def offload_runtime_memory(self) -> None:
            return None

        async def shutdown(self) -> None:
            return None

    return RolloutRuntimeCoordinator(
        collector=_CollectorControl(runtime),
        strategy=strategy,
        training_state_getter=lambda: state,
        weight_syncer=None,
        sync_state_getter=None,
        weights_initialized=lambda: True,
        set_weights_initialized=lambda _v: None,
    )


def test_offload_and_restore_move_frozen_components() -> None:
    model = _FakeDriverModel(with_frozen_hook=True)
    lifecycle = _coordinator(model)
    stats = RolloutStats()

    assert lifecycle.park_training_state_for_rollout(stats) is True
    assert model.to_calls == [torch.device("cpu")]
    assert model.frozen_calls == [torch.device("cpu")]  # frozen components parked

    lifecycle.restore_training_state_after_rollout(stats)
    assert model.to_calls[-1].type == "cuda"
    assert model.frozen_calls[-1].type == "cuda"  # frozen components restored


def test_model_without_frozen_hook_is_unaffected() -> None:
    """AR families register their VAE directly and expose no hook — no crash."""
    model = _FakeDriverModel(with_frozen_hook=False)
    lifecycle = _coordinator(model)

    assert lifecycle.park_training_state_for_rollout(RolloutStats()) is True
    lifecycle.restore_training_state_after_rollout(RolloutStats())
    assert model.to_calls == [torch.device("cpu"), torch.device("cuda")]
    assert model.frozen_calls == []
