"""inference_runtime selects the reward transport (P6 wiring)."""

from __future__ import annotations

from vrl.rewards.functions.pickscore import PickScoreReward
from vrl.rewards.ray.runtime import RayRewardRuntime
from vrl.rewards.runtime import LocalRewardRuntime, make_reward_runtime

_FACTORY = "vrl.rewards.models.pickscore:pickscore_reward_model"


def test_make_reward_runtime_local() -> None:
    """Checks make reward runtime local."""
    runtime = make_reward_runtime(
        "local", model_factory=_FACTORY, worker_config={"device": "cpu"},
    )
    assert isinstance(runtime, LocalRewardRuntime)


def test_make_reward_runtime_ray_nests_worker_config() -> None:
    """Checks make reward runtime Ray nests worker config."""
    runtime = make_reward_runtime(
        "ray",
        model_factory=_FACTORY,
        worker_config={"device": "cpu", "num_workers": 1},
    )
    assert isinstance(runtime, RayRewardRuntime)
    assert runtime.worker_config["model_factory"] == _FACTORY


def test_make_reward_runtime_rejects_unknown() -> None:
    """Checks make reward runtime rejects unknown."""
    import pytest

    with pytest.raises(ValueError, match="inference_runtime"):
        make_reward_runtime("inline", model_factory=_FACTORY)


def test_reward_class_defaults_to_local_transport() -> None:
    """Checks reward class defaults to local transport."""
    reward = PickScoreReward(device="cpu")
    assert isinstance(reward.runtime, LocalRewardRuntime)


def test_reward_class_opts_into_ray_transport() -> None:
    """Checks reward class opts into Ray transport."""
    reward = PickScoreReward(device="cpu", inference_runtime="ray", num_workers=1)
    assert isinstance(reward.runtime, RayRewardRuntime)
