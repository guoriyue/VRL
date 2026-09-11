"""Early argument validation for the generation bottleneck CLI."""

import pytest

from vrl.scripts.perf import generation_bottleneck_profile as profiler


@pytest.mark.parametrize(
    "option, value", [("--steps", "0"), ("--steps", "-1"), ("--warmup", "-1")]
)
def test_invalid_counts_fail_before_config_loading(monkeypatch, capsys, option, value):
    def load(*args, **kwargs):
        pytest.fail("invalid counts must not load configuration or models")

    monkeypatch.setattr(profiler, "load_config", load)
    with pytest.raises(SystemExit) as caught:
        profiler.main(["--config", "unused", option, value])
    assert caught.value.code == 2
    assert option in capsys.readouterr().err


@pytest.mark.parametrize("stage", ["forward", "profiler", "monitor_wait"])
def test_monitor_is_reaped_when_profiling_fails(monkeypatch, stage):
    from contextlib import contextmanager
    from types import SimpleNamespace

    failure = RuntimeError("profile failed")
    if stage == "monitor_wait":
        failure = profiler.subprocess.TimeoutExpired("dmon", 30)
    calls = []

    class Monitor:
        def kill(self):
            calls.append("kill")

        def communicate(self, timeout=None):
            calls.append(("communicate", timeout))
            if timeout is not None:
                raise failure
            return "", None

    @contextmanager
    def profile(**kwargs):
        if stage == "profiler":
            raise failure
        yield None

    def step(index):
        if stage == "forward":
            raise failure

    root = SimpleNamespace(
        sampling=SimpleNamespace(width=8, height=8, num_steps=1), precision=None
    )
    monkeypatch.setattr(profiler, "load_config", lambda *a, **kw: root)
    monkeypatch.setattr(profiler, "parse_config", lambda cfg: cfg)
    monkeypatch.setattr(
        profiler, "PrecisionPolicy", SimpleNamespace(from_section=lambda value: None)
    )
    monkeypatch.setattr(profiler, "build_runtime", lambda *a, **kw: SimpleNamespace(model=None))
    monkeypatch.setattr(profiler, "make_step_fn", lambda *a: step)
    monkeypatch.setattr(profiler.torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(profiler.torch.cuda, "reset_peak_memory_stats", lambda *a: None)
    monkeypatch.setattr(profiler.torch.cuda, "max_memory_allocated", lambda *a: 0)
    monkeypatch.setattr(profiler.torch.profiler, "profile", profile)
    monkeypatch.setattr(profiler.subprocess, "Popen", lambda *a, **kw: Monitor())
    with pytest.raises(type(failure)) as caught:
        profiler.main(["--config", "unused", "--warmup", "0", "--steps", "1"])
    assert caught.value is failure
    assert calls[-2:] == ["kill", ("communicate", None)]
