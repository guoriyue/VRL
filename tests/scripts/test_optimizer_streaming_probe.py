"""Run the comparison CLI with actual isolated processes and disk state."""

import json
import subprocess
import sys


def test_cpu_probe_compares_content_and_publishes_once(tmp_path):
    report = tmp_path / "report.json"
    command = [
        sys.executable,
        "-m",
        "vrl.scripts.perf.optimizer_streaming_probe",
        "--device",
        "cpu",
        "--parameters",
        "4",
        "--elements",
        "32",
        "--steps",
        "2",
        "--bucket-bytes",
        "260",
        "--directory",
        str(tmp_path / "scratch"),
        "--report",
        str(report),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    result = json.loads(report.read_text())
    assert result["exact_final_state_match"] is True
    assert result["moment_bytes"] == 4 * 260
    resident, disk = result["arms"]
    assert resident["arm"] == "resident" and disk["arm"] == "disk"
    assert disk["window_io"]["wchar"] > resident["window_io"]["wchar"]
    assert len(disk["steady_step_s"]) == 2
    assert disk["cuda_peak_allocated_bytes"] is None
    assert not list((tmp_path / "scratch").iterdir())
    before = report.read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert repeated.returncode != 0
    assert report.read_bytes() == before
