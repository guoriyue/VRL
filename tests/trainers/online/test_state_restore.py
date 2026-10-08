"""OnlineTrainer resume: optimizer/EMA/rollout-init state restore and pre-collect driver-weight push.

Every trainer is the real online wiring on tiny SANA (``real_trainer``). The
state a test restores is produced by a real training step on the real policy;
a restored trainer is a second, freshly built stack.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from tests.rollouts.collector._helpers import Trace
from tests.trainers.online._helpers import TrainerBench, real_trainer
from vrl.algorithms.types import InitialReplayStats
from vrl.trainers.distributed import DistributedTrainingContext
from vrl.trainers.online.trainer import OnlineTrainer
from vrl.trainers.strategy import DDPStrategy, SingleProcessStrategy


def _trainer(
    monkeypatch, tmp_path, *, ema: bool = False, overrides: tuple[str, ...] = ()
) -> TrainerBench:
    return real_trainer(
        monkeypatch,
        tmp_path,
        overrides=(
            "actor.ppo_epochs=1",
            "actor.drop_zero_advantage=false",
            f"actor.ema.enable={str(ema).lower()}",
            *overrides,
        ),
    )


def _trained(monkeypatch, tmp_path, *, ema: bool = False) -> TrainerBench:
    """A trainer after one real training step: optimizer, EMA and counters are live."""

    tb = _trainer(monkeypatch, tmp_path, ema=ema)
    asyncio.run(tb.trainer.step(["a cat"]))
    return tb


def _progress(trainer: OnlineTrainer) -> tuple[int, int]:
    return trainer.state.step, trainer.state.global_step


@pytest.mark.parametrize("field", ["step", "global_step"])
@pytest.mark.parametrize("value", [1.9, "2", True, -1])
def test_invalid_progress_does_not_modify_trainer(monkeypatch, tmp_path, field, value):
    tb = _trained(monkeypatch, tmp_path)
    progress = _progress(tb.trainer)
    state = {"step": 3, "global_step": 5, field: value}

    with pytest.raises(ValueError, match=rf"trainer_state\.{field}"):
        tb.trainer.load_state_dict(state)

    assert _progress(tb.trainer) == progress


class TestOnlineTrainerResumeState:
    """Groups tests for online trainer resume state."""

    def test_incompatible_ema_shape_preserves_existing_shadows(self, monkeypatch, tmp_path):
        tb = _trained(monkeypatch, tmp_path, ema=True)
        ema = tb.trainer._ema
        shadows = ema.ema_parameters
        originals = [shadow.clone() for shadow in shadows]
        num_updates = ema.num_updates
        state = tb.trainer.state_dict()
        state["ema"] = {
            "decay": 0.5,
            "num_updates": 10,
            # One shadow per trainable, each a different shape than its parameter.
            "ema_parameters": [torch.ones(shadow.numel() + 1) for shadow in shadows],
        }

        with pytest.raises(ValueError, match="EMA parameter shape mismatch"):
            tb.trainer.load_state_dict(state)

        assert ema.ema_parameters is shadows
        for shadow, original in zip(shadows, originals, strict=True):
            torch.testing.assert_close(shadow, original)
        assert ema.num_updates == num_updates
        assert ema.decay == tb.trainer.config.ema.decay

    def test_load_state_dict_accepts_legacy_total_keys(self, monkeypatch, tmp_path) -> None:
        """Checkpoints written before the totals were dropped still resume."""

        source = _trained(monkeypatch, tmp_path / "source")
        state = source.trainer.state_dict()
        state.update({"total_reward": 99.0, "total_loss": 101.0})
        restored = _trainer(monkeypatch, tmp_path / "restored")

        restored.trainer.load_state_dict(state)

        assert _progress(restored.trainer) == _progress(source.trainer)

    def test_strict_resume_rejects_master_state_for_plain_optimizer(
        self, monkeypatch, tmp_path
    ) -> None:
        source = _trained(monkeypatch, tmp_path / "source")
        state = source.trainer.state_dict()
        state["optimizer"]["fp32_master_weights"] = {
            "version": 1,
            "parameters": [],
        }
        restored = _trainer(monkeypatch, tmp_path / "restored")

        with pytest.raises(ValueError, match="master-weight state does not match"):
            restored.trainer.load_state_dict(state)

    def test_load_state_dict_initializes_and_restores_ema_state(
        self, monkeypatch, tmp_path
    ) -> None:
        """``load_state_dict`` creates the EMA on demand when EMA is enabled and restores its
        shadow parameters.
        """

        source = _trained(monkeypatch, tmp_path / "source", ema=True)
        state = source.trainer.state_dict()
        restored = _trainer(monkeypatch, tmp_path / "restored", ema=True)

        restored.trainer.load_state_dict(state)

        assert restored.trainer._ema is not None
        assert restored.trainer._ema.num_updates == source.trainer._ema.num_updates
        for actual, expected in zip(
            restored.trainer._ema.ema_parameters,
            source.trainer._ema.ema_parameters,
            strict=True,
        ):
            assert torch.equal(actual, expected)

    def test_state_dict_includes_initial_ema_state_before_first_step(
        self, monkeypatch, tmp_path
    ) -> None:
        """Checks zero-step EMA checkpoints can resume strictly."""

        source = _trainer(monkeypatch, tmp_path / "source", ema=True)
        state = source.trainer.state_dict()
        restored = _trainer(monkeypatch, tmp_path / "restored", ema=True)

        restored.trainer.load_state_dict(state)

        assert state["ema"]["num_updates"] == 0
        assert restored.trainer._ema is not None

    def test_load_state_dict_rejects_ema_state_when_ema_is_disabled(
        self, monkeypatch, tmp_path
    ) -> None:
        source = _trained(monkeypatch, tmp_path / "source", ema=True)
        state = source.trainer.state_dict()
        restored = _trainer(monkeypatch, tmp_path / "restored", ema=False)

        with pytest.raises(ValueError, match="EMA state"):
            restored.trainer.load_state_dict(state)

    def test_load_state_dict_resets_rollout_weight_initialization(
        self, monkeypatch, tmp_path
    ) -> None:
        """Loading a checkpoint resets the rollout-weights-initialized and replay-parity flags, so
        the next step re-pushes weights and re-checks parity against the restored state.
        """

        tb = _trained(monkeypatch, tmp_path)
        assert tb.trainer._rollout_runtime.weights_initialized is True
        assert tb.trainer._replay_parity_passed is True

        tb.trainer.load_state_dict(tb.trainer.state_dict())

        assert tb.trainer._rollout_runtime.weights_initialized is False
        assert tb.trainer._replay_parity_passed is False

    def test_resume_rechecks_parity_at_nonzero_training_step(self, monkeypatch, tmp_path) -> None:
        tb = _trained(monkeypatch, tmp_path)
        assert tb.trainer._replay_parity_passed is True
        state = tb.trainer.state_dict()
        assert "_replay_parity_passed" not in state

        tb.trainer.load_state_dict(state)

        with pytest.raises(RuntimeError, match="replay parity failed"):
            tb.trainer._validate_first_update_parity(
                InitialReplayStats(logprob_abs_diff_max=0.02, finite=True),
                local_weight=1.0,
            )
        assert _progress(tb.trainer) == (state["step"], state["global_step"])
        assert tb.trainer._replay_parity_passed is False

    def test_strict_resume_requires_master_optimizer_state_after_first_step(
        self, monkeypatch, tmp_path
    ) -> None:
        tb = _trainer(monkeypatch, tmp_path, overrides=("precision.training.dtype=bf16",))
        assert next(iter(tb.trainable_parameters().values())).dtype is torch.bfloat16

        with pytest.raises(ValueError, match=r"missing optimizer state.*master residuals"):
            tb.trainer.load_state_dict({"step": 1, "global_step": 1})

        tb.trainer.load_state_dict({"step": 0, "global_step": 0})

    def test_low_precision_master_gate_runs_before_distributed_prepare(
        self, monkeypatch, tmp_path
    ) -> None:
        tb = _trainer(monkeypatch, tmp_path, overrides=("precision.training.dtype=fp16",))
        assert next(iter(tb.trainable_parameters().values())).dtype is torch.float16

        def rewire(strategy) -> OnlineTrainer:
            """The same trainer wiring under another strategy."""

            trainer = tb.trainer
            return OnlineTrainer(
                algorithm=trainer.algorithm,
                collector=trainer.collector,
                evaluator=trainer.evaluator,
                model=tb.model,
                weight_syncer=trainer.weight_syncer,
                sync_state_getter=trainer.sync_state_getter,
                config=trainer.config,
                strategy=strategy,
            )

        prepared = Trace(monkeypatch)
        # A DDP rank: the constructor needs no process group before prepare_model.
        distributed = DDPStrategy(
            DistributedTrainingContext(
                strategy="ddp", rank=0, world_size=1, device=torch.device("cpu")
            ),
            find_unused_parameters=False,
        )
        prepared.watch(distributed, "prepare_model", "ddp.prepare_model")
        with pytest.raises(NotImplementedError, match="use BF16 or single_process"):
            rewire(distributed)
        assert prepared.events == []

        compatible = SingleProcessStrategy()
        prepared.watch(compatible, "prepare_model", "single.prepare_model")
        rewire(compatible)
        assert prepared.events == ["single.prepare_model"]
        assert next(iter(tb.trainable_parameters().values())).dtype is torch.float16

    def test_resume_pushes_restored_driver_weights_before_next_collect(
        self, monkeypatch, tmp_path
    ) -> None:
        """After a resume the restored driver weights are pushed to the rollout before the first
        collect: the collector observes exactly one sync, carrying the restored weights.
        """

        tb = _trained(monkeypatch, tmp_path)
        state = tb.trainer.state_dict()
        # Model restoration is separate from trainer counters in checkpoint resume:
        # the checkpoint's policy differs from what the rollout last received.
        with torch.no_grad():
            for value in tb.trainable_parameters().values():
                value.add_(1e-3)
        restored_weights = tb.strategy.export_rollout_state(tb.bundle)
        trace = tb.collector.trace
        events_before = len(trace.calls)

        tb.trainer.load_state_dict(state)
        asyncio.run(tb.trainer.step(["a cat"]))

        resumed = trace.calls[events_before:]
        first_generate = next(i for i, (event, _) in enumerate(resumed) if event == "generate")
        pushes = [args for event, args in resumed[:first_generate] if event == "update_weights"]
        assert len(pushes) == 1
        payload = pushes[0][0]
        assert payload.keys() == restored_weights.keys()
        for name, value in restored_weights.items():
            torch.testing.assert_close(payload[name], value, rtol=0, atol=0)


def _cuda_fp16_trainer(monkeypatch, tmp_path) -> OnlineTrainer:
    """The real FP16 policy on the card: native FP16 gradients need the GradScaler."""

    tb = _trainer(monkeypatch, tmp_path, overrides=("precision.training.dtype=fp16",))
    tb.model.to("cuda")
    trainer = tb.trainer
    return OnlineTrainer(
        algorithm=trainer.algorithm,
        collector=trainer.collector,
        evaluator=trainer.evaluator,
        model=tb.model,
        weight_syncer=trainer.weight_syncer,
        sync_state_getter=trainer.sync_state_getter,
        config=trainer.config,
        strategy=SingleProcessStrategy(
            DistributedTrainingContext("single_process", 0, 1, torch.device("cuda"))
        ),
    )


@pytest.mark.gpu
def test_fp16_cuda_state_dict_round_trips_grad_scaler(monkeypatch, tmp_path) -> None:
    """CUDA fp16 training must save and restore GradScaler state."""

    source = _cuda_fp16_trainer(monkeypatch, tmp_path / "source")
    assert source._grad_scaler is not None
    optimizer = source._ensure_optimizer()
    trainables = [p for p in source.model.parameters() if p.requires_grad]
    source._backward(sum(value.float().square().sum() for value in trainables))
    source._clip_and_step(optimizer)
    state = source.state_dict()

    assert "grad_scaler" in state

    restored = _cuda_fp16_trainer(monkeypatch, tmp_path / "restored")
    restored.load_state_dict(state)

    assert restored._grad_scaler is not None
    assert restored._grad_scaler.state_dict()["scale"] == state["grad_scaler"]["scale"]


@pytest.mark.gpu
def test_strict_resume_rejects_nonzero_fp16_checkpoint_without_scaler(
    monkeypatch, tmp_path
) -> None:
    trainer = _cuda_fp16_trainer(monkeypatch, tmp_path)
    optimizer = trainer._ensure_optimizer()
    trainables = [p for p in trainer.model.parameters() if p.requires_grad]
    trainer._backward(sum(value.float().square().sum() for value in trainables))
    trainer._clip_and_step(optimizer)
    state = trainer.state_dict()
    # A nonzero-step checkpoint written without the scaler, e.g. by a non-fp16 run.
    state.update(step=1, global_step=1)
    del state["grad_scaler"]

    with pytest.raises(ValueError, match="missing GradScaler state"):
        trainer.load_state_dict(state)


def test_online_trainer_standard_adamw_roundtrip(monkeypatch, tmp_path) -> None:
    source = _trained(monkeypatch, tmp_path / "source")
    optimizer = source.trainer._optimizer
    assert type(optimizer) is torch.optim.AdamW
    transformer = source.bundle.model.trainable_modules["transformer"]
    checkpoint = tmp_path / "state.pt"
    torch.save(
        {"trainer": source.trainer.state_dict(), "model": transformer.state_dict()}, checkpoint
    )
    saved = torch.load(checkpoint, weights_only=True)

    restored = _trainer(monkeypatch, tmp_path / "restored")
    restored.bundle.model.trainable_modules["transformer"].load_state_dict(saved["model"])
    restored.trainer.load_state_dict(saved["trainer"])

    assert _progress(restored.trainer) == _progress(source.trainer)
    assert restored.trainer._optimizer is not None
    # One more identical update on both: the restored moments reproduce it bit for bit.
    for tb in (source, restored):
        loss = sum(value.square().sum() for value in tb.trainable_parameters().values())
        loss.backward()
        tb.trainer._ensure_optimizer().step()
    for name, value in source.trainable_parameters().items():
        torch.testing.assert_close(restored.trainable_parameters()[name], value, rtol=0, atol=0)
    expected = optimizer.state_dict()["state"]
    actual = restored.trainer._optimizer.state_dict()["state"]
    assert actual.keys() == expected.keys()
    for index in expected:
        for key in expected[index]:
            torch.testing.assert_close(actual[index][key], expected[index][key], rtol=0, atol=0)


def test_optimizer_restore_failure_preserves_progress(monkeypatch, tmp_path):
    tb = _trained(monkeypatch, tmp_path)
    state = tb.trainer.state_dict()
    state.update(step=3, global_step=5)
    progress = _progress(tb.trainer)
    restore = Trace(monkeypatch)
    restore.watch(tb.strategy, "load_optimizer_state", "load_optimizer_state")
    restore.fail("load_optimizer_state", "optimizer restore failed")

    with pytest.raises(RuntimeError, match="optimizer restore failed"):
        tb.trainer.load_state_dict(state)

    assert _progress(tb.trainer) == progress
