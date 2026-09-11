"""Unit tests for the diffusion rollout TeaCache decision machine (CPU, no model)."""

from __future__ import annotations

import pytest
import torch

from vrl.generation.steps.denoise.teacache import (
    TeaCacheConfig,
    TeaCacheState,
    relative_l1_change,
)


def test_from_sampling_off_variants_return_none():
    assert TeaCacheConfig.from_sampling(None) is None
    assert TeaCacheConfig.from_sampling(False) is None
    assert TeaCacheConfig.from_sampling({"enabled": False}) is None


def test_from_sampling_true_uses_defaults():
    cfg = TeaCacheConfig.from_sampling(True)
    assert cfg is not None
    assert cfg.threshold > 0 and cfg.warmup_steps >= 0


def test_from_sampling_mapping_overrides():
    cfg = TeaCacheConfig.from_sampling({"threshold": 0.3, "warmup_steps": 1})
    assert cfg is not None
    assert cfg.threshold == 0.3 and cfg.warmup_steps == 1


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        TeaCacheConfig(threshold=0.0, warmup_steps=0)


def test_enabled_sampling_rejects_unknown_parameter():
    with pytest.raises(TypeError, match="threshhold"):
        TeaCacheConfig.from_sampling({"threshhold": 0.3})


def test_warmup_and_last_step_always_run():
    cfg = TeaCacheConfig(threshold=999.0, warmup_steps=2)
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
    cfg = TeaCacheConfig(threshold=0.5, warmup_steps=1)
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


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_nonfinite_signal_never_grants_cache_reuse(invalid_value):
    state = TeaCacheState(TeaCacheConfig(threshold=0.5, warmup_steps=1), num_steps=6)
    finite = torch.ones(1, 4)
    assert state.should_run(finite, 0)
    state.cache_noise_pred(torch.zeros_like(finite))

    assert state.should_run(torch.full_like(finite, invalid_value), 1)
    # The previous signal is still invalid, so this comparison cannot authorize
    # reuse either. Once two finite signals are compared, normal skipping resumes.
    assert state.should_run(finite, 2)
    assert not state.should_run(finite, 3)
    assert state.runs == 3
    assert state.skips == 1


def test_no_cache_yet_forces_run():
    cfg = TeaCacheConfig(threshold=0.0001, warmup_steps=0)
    state = TeaCacheState(cfg, num_steps=4)
    # warmup=0 but cache is empty on the first step -> must run
    assert state.should_run(torch.ones(1, 4), 0) is True


def test_skip_ratio_and_counters():
    cfg = TeaCacheConfig(threshold=999.0, warmup_steps=1)
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


@pytest.mark.parametrize("previous,current,expected", [(100, 101, 0.01), (-40000, 40000, 2.0)])
def test_relative_l1_avoids_half_precision_overflow(previous, current, expected):
    previous_signal = torch.full((1024,), previous, dtype=torch.float16)
    current_signal = torch.full((1024,), current, dtype=torch.float16)
    assert relative_l1_change(current_signal, previous_signal) == pytest.approx(expected)


def test_half_precision_change_above_threshold_runs_forward():
    state = TeaCacheState(TeaCacheConfig(threshold=0.005, warmup_steps=1), num_steps=4)
    previous = torch.full((1024,), 100, dtype=torch.float16)
    assert state.should_run(previous, 0)
    state.cache_noise_pred(torch.zeros_like(previous))
    assert state.should_run(previous + 1, 1)
    assert state.skips == 0


@pytest.mark.parametrize("enabled", ["false", "true", 0, 1, None])
def test_sampling_enabled_requires_boolean(enabled):
    with pytest.raises(ValueError, match=r"teacache\.enabled"):
        TeaCacheConfig.from_sampling({"enabled": enabled})


@pytest.mark.parametrize(
    "field,value",
    [
        ("threshold", float("nan")),
        ("threshold", float("inf")),
        ("threshold", -1.0),
        ("threshold", True),
        ("threshold", "0.15"),
        ("warmup_steps", True),
        ("warmup_steps", 1.5),
        ("warmup_steps", "2"),
        ("warmup_steps", -1),
    ],
)
@pytest.mark.parametrize("from_mapping", [False, True])
def test_config_boundaries_reject_invalid_cache_parameters(field, value, from_mapping):
    with pytest.raises(ValueError, match=rf"teacache\.{field}"):
        if from_mapping:
            TeaCacheConfig.from_sampling({field: value})
        else:
            TeaCacheConfig(**{field: value})
