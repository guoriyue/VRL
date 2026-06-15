"""execution selects the reward transport (inline vs pool)."""

from __future__ import annotations

from vrl.rewards.functions.pickscore import PickScoreReward
from vrl.rewards.ray.runtime import RayRewardRuntime
from vrl.rewards.runtime import LocalRewardRuntime, make_reward_runtime

_FACTORY = "vrl.rewards.models.pickscore:pickscore_reward_model"


def test_make_reward_runtime_inline() -> None:
    """Checks make reward runtime inline -> in-process transport."""
    runtime = make_reward_runtime(
        "inline", model_factory=_FACTORY, worker_config={"device": "cpu"},
    )
    assert isinstance(runtime, LocalRewardRuntime)


def test_make_reward_runtime_pool_nests_worker_config() -> None:
    """Checks make reward runtime pool -> worker-pool transport, nests worker config."""
    runtime = make_reward_runtime(
        "pool",
        model_factory=_FACTORY,
        worker_config={"device": "cpu", "num_workers": 1},
    )
    assert isinstance(runtime, RayRewardRuntime)
    assert runtime.worker_config["model_factory"] == _FACTORY


def test_make_reward_runtime_rejects_unknown() -> None:
    """Checks make reward runtime rejects unknown."""
    import pytest

    with pytest.raises(ValueError, match="execution"):
        make_reward_runtime("bogus", model_factory=_FACTORY)


def test_reward_class_defaults_to_inline_transport() -> None:
    """Checks reward class defaults to inline (in-process) transport."""
    reward = PickScoreReward(device="cpu")
    assert isinstance(reward.runtime, LocalRewardRuntime)


def test_reward_class_opts_into_pool_transport() -> None:
    """Checks reward class opts into pool (Ray worker) transport."""
    reward = PickScoreReward(device="cpu", execution="pool", num_workers=1)
    assert isinstance(reward.runtime, RayRewardRuntime)
