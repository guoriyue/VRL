"""Shared OnlineTrainer test helpers.

``real_trainer`` is the construction trainer tests start from: the online
recipe's own wiring (algorithm/evaluator pair from config, the run's replay
bundle as policy and reference, ``SingleProcessStrategy``, the real weight
syncer and strategy export) over ``real_collector`` on the tiny SANA stack.
A trainer knob is a config override, never an attribute poke.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from tests.rollouts.collector._helpers import CollectorBench, real_collector
from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.rollouts.batch import RolloutBatch
from vrl.trajectory.builders import build_diffusion_trajectory


def _diffusion_rollout_batch(
    *,
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    num_steps: int | None = None,
    observations: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    timesteps: torch.Tensor | None = None,
    extras: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> RolloutBatch:
    """Build a canonical typed diffusion batch for trainer tests."""

    batch_size = int(rewards.shape[0])
    if tuple(group_ids.shape) != (batch_size,):
        raise ValueError("group_ids must have one value per reward")
    if observations is None:
        step_count = 2 if num_steps is None else int(num_steps)
        observations = torch.zeros(
            batch_size,
            step_count,
            1,
            device=rewards.device,
        )
    else:
        if observations.ndim < 2 or int(observations.shape[0]) != batch_size:
            raise ValueError("observations must start with [sample, denoise]")
        step_count = int(observations.shape[1])
        if num_steps is not None and int(num_steps) != step_count:
            raise ValueError("num_steps does not match observations")
    if actions is None:
        actions = torch.zeros_like(observations)
    if tuple(actions.shape[:2]) != (batch_size, step_count):
        raise ValueError("actions must start with [sample, denoise]")

    old_log_prob = torch.zeros(
        batch_size,
        step_count,
        dtype=torch.float32,
        device=observations.device,
    )
    if timesteps is None:
        timesteps = (
            torch.arange(step_count, device=observations.device)
            .unsqueeze(0)
            .expand(batch_size, step_count)
        )

    group_values = group_ids.detach().cpu().tolist()
    prompt_indices: dict[Any, int] = {}
    sample_counts: dict[Any, int] = {}
    row_specs: list[tuple[int, int]] = []
    for group_id in group_values:
        prompt_index = prompt_indices.setdefault(group_id, len(prompt_indices))
        sample_index = sample_counts.get(group_id, 0)
        sample_counts[group_id] = sample_index + 1
        row_specs.append((prompt_index, sample_index))
    prompts = [f"trainer test prompt {index}" for index in range(len(prompt_indices))]
    request = GenerationRequest(
        request_id="trainer-test",
        family="test-diffusion",
        task="t2i",
        inputs=prompts,
        samples_per_prompt=max(sample_counts.values(), default=1),
    )
    sample_rows = [
        GenerationSampleRow(
            prompt_index=prompt_index,
            sample_index=sample_index,
            prompt=prompts[prompt_index],
            sample_id=f"trainer-test:sample:{index}",
        )
        for index, (prompt_index, sample_index) in enumerate(row_specs)
    ]
    batch_context = dict(context or {})
    trajectory = build_diffusion_trajectory(
        request=request,
        sample_rows=sample_rows,
        observations=observations,
        actions=actions,
        old_log_prob=old_log_prob,
        timesteps=timesteps,
        replay_tensors={},
        context=batch_context,
    )
    return RolloutBatch(
        rewards=rewards,
        group_ids=group_ids,
        extras=dict(extras or {}),
        context=batch_context,
        trajectory=trajectory,
    )


def bare_trainer(**attributes):
    """An ``OnlineTrainer`` with only the attributes a single method reads.

    The full constructor wires a rollout schedule, strategy, and algorithm; the
    index-selection and optimizer-derivation methods read one or two fields.
    Every test that needs such a shell builds it here, so the constructor bypass
    has one owner and the attribute names appear in one place.
    """

    from vrl.trainers.online.trainer import OnlineTrainer

    trainer = object.__new__(OnlineTrainer)
    for name, value in attributes.items():
        setattr(trainer, name, value)
    return trainer


@dataclass
class TrainerBench:
    """A real ``OnlineTrainer`` and the stack under it."""

    trainer: Any
    collector: CollectorBench
    bundle: Any
    strategy: Any

    @property
    def model(self) -> Any:
        return self.bundle.model

    def trainable_parameters(self) -> dict[str, torch.Tensor]:
        transformer = self.bundle.model.trainable_modules["transformer"]
        return {
            name: parameter
            for name, parameter in transformer.named_parameters()
            if parameter.requires_grad
        }


def real_trainer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    *,
    reward: Any = None,
    overrides: tuple[str, ...] = (),
    versioned_slots: bool = False,
    lifecycle: Any = None,
    strategy: Any = None,
) -> TrainerBench:
    """Build ``OnlineTrainer`` exactly as the online recipe wires it, on tiny SANA.

    ``strategy`` defaults to ``SingleProcessStrategy``; a distributed strategy
    needs its process group initialized by the caller.
    """

    from vrl.scripts.common.factory import AlgorithmEvaluatorPair
    from vrl.trainers.online.trainer import OnlineTrainer
    from vrl.trainers.strategy import SingleProcessStrategy
    from vrl.trainers.weight_sync import RayRuntimeWeightSyncer

    bench = real_collector(
        monkeypatch,
        tmp_path,
        reward=reward,
        lifecycle=lifecycle,
        versioned_slots=versioned_slots,
        overrides=overrides,
    )
    stack = bench.stack
    built = stack.resolved.built
    bundle = stack.trainer_bundle()
    strategy = strategy if strategy is not None else SingleProcessStrategy()
    pair = AlgorithmEvaluatorPair.from_configs(
        family_entry=stack.family,
        built=built,
        collector_config=stack.collector_config(),
        scheduler=getattr(bundle, "scheduler", None),
    )
    contract = built.root.algorithm.hyperparameters.config_contract
    uses_reference = (pair.evaluator is not None and pair.algorithm.kl_coef > 0) or (
        contract.requires_reference_policy
    )
    trainer = OnlineTrainer(
        algorithm=pair.algorithm,
        collector=bench.collector,
        evaluator=pair.evaluator,
        model=bundle.model,
        ref_model=bundle.model if uses_reference else None,
        weight_syncer=RayRuntimeWeightSyncer(bench.runtime),
        sync_state_getter=lambda: strategy.export_rollout_state(bundle),
        config=built.trainer,
        device=torch.device("cpu"),
        strategy=strategy,
    )
    return TrainerBench(trainer=trainer, collector=bench, bundle=bundle, strategy=strategy)
