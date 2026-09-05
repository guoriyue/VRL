"""The dead-flag lint must stay green — and must be able to go red.

A hygiene gate that cannot fail is worse than no gate: it reads as coverage
while proving nothing. The negative control is therefore the load-bearing
test here, and it runs the real detector over a real declaration.
"""

from __future__ import annotations

from vrl.scripts.lint.dead_flags import dead_flags, declared_flags


def test_no_dead_flags_in_the_repo() -> None:
    """Every declared CLI flag has a consumer (the 2026-08 audit's end state)."""

    findings = dead_flags()
    assert not findings, "dead CLI flags: " + ", ".join(
        f"{file}:{line} {option}" for file, line, option, _ in findings
    )


def test_the_sweep_actually_sees_the_repo() -> None:
    """Guard against a detector that passes because it scanned nothing."""

    flags = declared_flags()
    assert len(flags) > 100
    assert any(option.startswith("--") for _, _, option, _ in flags)


def test_detector_fires_on_an_injected_dead_flag(tmp_path, monkeypatch) -> None:
    """Negative control: an unconsumed declaration must be reported.

    Rebuilds the module against a temporary repo root holding one script with
    one consumed and one unconsumed flag, so the assertion exercises the real
    AST scan and the real consumer search rather than a stub.
    """

    import vrl.scripts.lint.dead_flags as lint

    (tmp_path / "vrl").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "vrl" / "probe.py").write_text(
        "import argparse\n"
        "def main():\n"
        "    p = argparse.ArgumentParser()\n"
        '    p.add_argument("--live-knob", type=int, default=1)\n'
        '    p.add_argument("--dead-knob", type=int, default=2)\n'
        "    args = p.parse_args()\n"
        "    return args.live_knob\n",
    )
    monkeypatch.setattr(lint, "REPO_ROOT", tmp_path)

    findings = lint.dead_flags()

    assert [option for _, _, option, _ in findings] == ["--dead-knob"]
