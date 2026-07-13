"""mps_colocation_probe: pin the attachment check, which is the probe's whole safety
property. A client that never reaches the MPS control daemon silently falls back to
context time-slicing, so a run mislabelled "MPS" measures the exact opposite of what
it claims. The probe may only call a run "MPS" when an mps-server actually holds the GPU."""

from __future__ import annotations

import subprocess

from vrl.scripts.perf import mps_colocation_probe as probe

_MPS_ROW = "1572627, nvidia-cuda-mps-server, 56 MiB"
_CLIENT_ROW = "1573072, /home/u/.venv/bin/python, 528 MiB"


def _stub_nvidia_smi(monkeypatch, stdout: str) -> None:
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout),
    )


def test_no_mps_server_means_not_attached(monkeypatch) -> None:
    """Two python clients on the GPU and no mps-server == plain time-slicing, not MPS."""
    _stub_nvidia_smi(monkeypatch, f"{_CLIENT_ROW}\n{_CLIENT_ROW}\n")
    assert probe.mps_server_pids() == []


def test_mps_server_is_detected(monkeypatch) -> None:
    _stub_nvidia_smi(monkeypatch, f"{_MPS_ROW}\n{_CLIENT_ROW}\n")
    assert probe.mps_server_pids() == ["1572627"]


def test_empty_gpu_is_not_attached(monkeypatch) -> None:
    _stub_nvidia_smi(monkeypatch, "\n")
    assert probe.mps_server_pids() == []


def test_missing_nvidia_smi_degrades_to_not_attached(monkeypatch) -> None:
    """On a CPU box the probe must report 'no MPS', never crash the caller."""

    def _boom(*a, **kw):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(probe.subprocess, "run", _boom)
    assert probe.mps_server_pids() == []
