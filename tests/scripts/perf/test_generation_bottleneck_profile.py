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
