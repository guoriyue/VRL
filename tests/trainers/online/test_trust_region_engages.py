"""L1 invariant: a trust-region algorithm's ratio term must be able to engage.

Flow-DPPO / GRPO-Guard are *defined* by a clipped/guarded importance ratio
r = pi_new/pi_old. With strict_on_policy + ppo_epochs=1 the behavior and target
policy are identical on the single replay pass, so r==1, the trust-region term is
identically zero, and the run is byte-equivalent to plain GRPO (the documented
flat-curve root cause). The trainer must refuse that config at construction
instead of silently training a no-op mechanism. Plain GRPO's clip is only a
safety rail, so it stays valid at ppo_epochs=1 (honest REINFORCE+group-baseline).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tests.trainers.online._collector_control import CollectorControlFake
from tests.trainers.online._helpers import (
    _stamp_model_precision,
    _trajectory_signals,
)
from vrl.algorithms.grpo.continuous import GRPO, FlowDPPO, GRPOGuard
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.base import Evaluator
from vrl.trainers.core.types import (
    ContinuousRolloutConfig,
    EMAConfig,
    OptimConfig,
    RolloutOrchestrationConfig,
)
from vrl.trainers.online import OnlineTrainer
from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig


class _Collector(CollectorControlFake):
    async def score_rollouts(self, pendings):
        return list(pendings)

    async def collect_unscored(self, prompts, **kwargs):
        group_size = int(kwargs["group_size"])
        return RolloutBatch(
            observations=torch.zeros(group_size, 2, 1),
            actions=torch.zeros(group_size, 2, 1),
            rewards=torch.arange(group_size, dtype=torch.float32),
            group_ids=torch.zeros(group_size, dtype=torch.long),
            context={},
        )


class _Evaluator(Evaluator):
    def evaluate(self, model, batch, timestep_idx, **kw):
        del kw
        return _trajectory_signals(
            batch,
            model.weight.view(1).expand(batch.rewards.shape[0]),
            timestep_idx,
        )


def _build_trainer(
    tmp_path,
    *,
    algorithm,
    ppo_epochs: int = 1,
    schedule_mode: str = "strict_on_policy",
    max_stale_policy_versions: int = 1,
) -> OnlineTrainer:
    model = nn.Linear(1, 1, bias=False)
    _stamp_model_precision(model)
    with torch.no_grad():
        model.weight.fill_(1.0)
    return OnlineTrainer(
        algorithm=algorithm,
        collector=_Collector(),
        evaluator=_Evaluator(),
        model=model,
        config=TrainerConfig(
            batch_plan=OnlineBatchPlan(prompts_per_batch=1, n_samples_per_prompt=2),
            timestep_fraction=1.0,
            ppo_epochs=ppo_epochs,
            drop_zero_advantage=False,
            optim=OptimConfig(lr=0.01),
            ema=EMAConfig(),
            rollout_orchestration=RolloutOrchestrationConfig(
                schedule_mode=schedule_mode,
                continuous=ContinuousRolloutConfig(
                    max_stale_policy_versions=max_stale_policy_versions,
                ),
            ),
            train_precision="no",
            output_dir=str(tmp_path),
        ),
        device="cpu",
    )


def test_flag_values_match_objective() -> None:
    # Plain GRPO: clip is a safety rail, valid at one epoch.
    assert GRPO().requires_active_trust_region is False
    # Subclasses whose loss IS the ratio term must require a moving policy.
    assert FlowDPPO().requires_active_trust_region is True
    assert GRPOGuard().requires_active_trust_region is True


def test_flow_dppo_strict_single_epoch_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="trust region"):
        _build_trainer(tmp_path, algorithm=FlowDPPO(), ppo_epochs=1)


def test_grpo_guard_strict_single_epoch_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="ppo_epochs"):
        _build_trainer(tmp_path, algorithm=GRPOGuard(), ppo_epochs=1)


def test_trust_region_with_multi_epoch_is_allowed(tmp_path) -> None:
    # ppo_epochs>1 lets the policy move between epochs, so the ratio engages.
    trainer = _build_trainer(tmp_path, algorithm=FlowDPPO(), ppo_epochs=2)
    assert trainer is not None


def test_plain_grpo_single_epoch_is_allowed(tmp_path) -> None:
    # Plain GRPO at one epoch is honest REINFORCE+group-baseline, not a no-op.
    trainer = _build_trainer(tmp_path, algorithm=GRPO(), ppo_epochs=1)
    assert trainer is not None


def test_continuous_zero_staleness_is_rejected_by_typed_config() -> None:
    with pytest.raises(ValueError, match=r"max_stale_policy_versions must be >= 1"):
        ContinuousRolloutConfig(max_stale_policy_versions=0)


def test_continuous_with_staleness_allows_single_epoch_trust_region(tmp_path) -> None:
    trainer = _build_trainer(
        tmp_path,
        algorithm=FlowDPPO(),
        schedule_mode="continuous",
        max_stale_policy_versions=1,
    )
    assert trainer is not None
