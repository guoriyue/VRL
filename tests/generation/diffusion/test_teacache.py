"""Unit tests for the diffusion rollout TeaCache decision machine (CPU, no model)."""

from __future__ import annotations

import pytest
import torch

from vrl.generation.diffusion.teacache import (
    TeaCacheConfig,
    TeaCacheState,
    teacache_signal,
)


def test_from_sampling_off_variants_return_none():
    assert TeaCacheConfig.from_sampling(None) is None
    assert TeaCacheConfig.from_sampling(False) is None
    assert TeaCacheConfig.from_sampling({"enabled": False}) is None


def test_from_sampling_true_uses_defaults():
    cfg = TeaCacheConfig.from_sampling(True)
    assert cfg is not None
    assert cfg.enabled and cfg.signal == "latent"
    assert cfg.threshold > 0 and cfg.warmup_steps >= 0


def test_from_sampling_mapping_overrides():
    cfg = TeaCacheConfig.from_sampling({"threshold": 0.3, "warmup_steps": 1})
    assert cfg is not None
    assert cfg.threshold == 0.3 and cfg.warmup_steps == 1


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        TeaCacheConfig(enabled=True, threshold=0.1, warmup_steps=0, signal="bogus")
    with pytest.raises(ValueError):
        TeaCacheConfig(enabled=True, threshold=0.0, warmup_steps=0, signal="latent")


def test_warmup_and_last_step_always_run():
    cfg = TeaCacheConfig(enabled=True, threshold=999.0, warmup_steps=2, signal="latent")
    state = TeaCacheState(cfg, num_steps=4)
    sig = torch.ones(1, 4)
    # warmup steps 0,1 forced to run even though change is zero (caller caches)
    assert state.should_run(sig, 0) is True
    state.cache_noise_pred(torch.zeros(1, 4))
    assert state.should_run(sig, 1) is True
    state.cache_noise_pred(torch.zeros(1, 4))
    # step 2: huge threshold + tiny change + a cache present -> skip
    assert state.should_run(sig, 2) is False
    # last step (3) forced to run
    assert state.should_run(sig, 3) is True
    assert state.runs == 3 and state.skips == 1


def test_low_change_skips_high_change_runs():
    cfg = TeaCacheConfig(enabled=True, threshold=0.5, warmup_steps=1, signal="latent")
    state = TeaCacheState(cfg, num_steps=6)
    base = torch.ones(1, 8)
    state.should_run(base, 0)  # warmup run -> caches
    state.cache_noise_pred(torch.zeros(1, 8))
    # tiny change (rel-L1 ~0.01) stays under threshold -> skip
    assert state.should_run(base * 1.01, 1) is False
    # accumulates; another tiny change still under -> skip
    assert state.should_run(base * 1.02, 2) is False
    # a large jump pushes accumulated change over threshold -> run + reset
    assert state.should_run(base * 2.0, 3) is True


def test_no_cache_yet_forces_run():
    cfg = TeaCacheConfig(enabled=True, threshold=0.0001, warmup_steps=0, signal="latent")
    state = TeaCacheState(cfg, num_steps=4)
    # warmup=0 but cache is empty on the first step -> must run
    assert state.should_run(torch.ones(1, 4), 0) is True


def test_skip_ratio_and_counters():
    cfg = TeaCacheConfig(enabled=True, threshold=999.0, warmup_steps=1, signal="latent")
    state = TeaCacheState(cfg, num_steps=5)
    sig = torch.ones(1, 4)
    state.should_run(sig, 0)  # run (warmup)
    state.cache_noise_pred(torch.zeros(1, 4))
    state.should_run(sig, 1)  # skip
    state.should_run(sig, 2)  # skip
    state.should_run(sig, 3)  # skip
    state.should_run(sig, 4)  # run (last)
    c = state.counters()
    assert c["teacache_runs"] == 2 and c["teacache_skips"] == 3
    assert c["teacache_skip_ratio"] == pytest.approx(3 / 5)


def test_signal_latent_passthrough_and_unknown_raises():
    lat = torch.randn(2, 3)
    assert teacache_signal(lat, "latent") is lat
    with pytest.raises(ValueError):
        teacache_signal(lat, "modulated")
