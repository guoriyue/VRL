"""Online recipe controller configuration ownership tests."""

from __future__ import annotations

import pytest

from vrl.config.schema import RootConfig
from vrl.run import OnlineRunConfig


def _root(**trainer: object) -> RootConfig:
    return RootConfig.model_validate({"trainer": trainer})


def test_online_run_config_uses_controller_defaults() -> None:
    run = OnlineRunConfig.from_root(_root(total_epochs=0))

    assert run.total_epochs == 0
    assert run.save_freq == 50
    assert run.seed == 0


def test_online_run_config_preserves_explicit_values() -> None:
    run = OnlineRunConfig.from_root(
        _root(total_epochs=12, save_freq=3, seed=-7),
    )

    assert run == OnlineRunConfig(total_epochs=12, save_freq=3, seed=-7)


@pytest.mark.parametrize(
    ("trainer", "path"),
    [
        ({}, "trainer.total_epochs"),
        ({"total_epochs": -1}, "trainer.total_epochs"),
        ({"total_epochs": 1, "save_freq": -1}, "trainer.save_freq"),
    ],
)
def test_online_run_config_rejects_invalid_controller_values(
    trainer: dict[str, object],
    path: str,
) -> None:
    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        OnlineRunConfig.from_root(_root(**trainer))


def test_process_seed_repeats_initialization_and_separates_rank_streams():
    import random

    import numpy as np
    import torch

    from vrl.trainers.checkpointing import capture_rng_state, restore_rng_state

    previous = capture_rng_state()
    try:
        run = OnlineRunConfig(total_epochs=1, seed=-7)
        run.initialize_process_rng()
        weights = torch.nn.Linear(4, 3).weight.detach().clone()
        first = (random.random(), np.random.rand(), torch.rand(3))
        run.initialize_process_rng()
        assert torch.equal(torch.nn.Linear(4, 3).weight, weights)
        assert random.random() == first[0]
        assert np.random.rand() == first[1]
        assert torch.equal(torch.rand(3), first[2])
        run.initialize_process_rng(rank=1)
        other_rank = torch.rand(3)
        run.initialize_process_rng(rank=1)
        assert torch.equal(other_rank, torch.rand(3))
        run.initialize_process_rng(rank=0)
        assert not torch.equal(other_rank, torch.rand(3))
        saved = capture_rng_state()
        expected = torch.rand(3)
        run.initialize_process_rng(rank=1)
        restore_rng_state(saved)
        assert torch.equal(torch.rand(3), expected)
    finally:
        restore_rng_state(previous)


def test_deterministic_policy_is_explicit_and_strict(monkeypatch):
    import torch

    from vrl.trainers.checkpointing import capture_rng_state, restore_rng_state

    previous = capture_rng_state()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        run = OnlineRunConfig.from_root(_root(total_epochs=1, deterministic=True))
        run.initialize_process_rng()
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        # A false/default option must not weaken an externally established policy.
        OnlineRunConfig(total_epochs=1).initialize_process_rng()
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn
        restore_rng_state(previous)


def test_late_or_incompatible_workspace_fails_before_reseeding(monkeypatch):
    import torch

    run = OnlineRunConfig(total_epochs=1, deterministic=True)
    before = torch.get_rng_state()
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="before launching"):
        run.initialize_process_rng()
    assert torch.equal(before, torch.get_rng_state())
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":bad")
    with pytest.raises(ValueError, match="workspace config"):
        run.initialize_process_rng()


@pytest.mark.parametrize("seed", [-(2**63) - 1, 2**64])
def test_seed_range_is_rejected_at_config_boundary(seed):
    with pytest.raises(ValueError, match="64-bit seed range"):
        OnlineRunConfig(total_epochs=1, seed=seed)


def test_strict_cuda_seed_repeats_small_optimizer_updates(monkeypatch):
    import torch

    from vrl.trainers.checkpointing import capture_rng_state, restore_rng_state

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    previous = capture_rng_state()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run = OnlineRunConfig(total_epochs=3, seed=71, deterministic=True)
    results = []
    try:
        for _ in range(2):
            run.initialize_process_rng()
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4), torch.nn.Dropout(0.25), torch.nn.Linear(4, 1)
            ).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False, fused=False)
            inputs = torch.randn(8, 4, device="cuda")
            losses = []
            for _step in range(3):
                optimizer.zero_grad()
                loss = model(inputs).square().mean()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            results.append((losses, [p.detach().cpu().clone() for p in model.parameters()]))
        assert results[0][0] == results[1][0]
        assert all(torch.equal(a, b) for a, b in zip(results[0][1], results[1][1], strict=True))
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn
        restore_rng_state(previous)
