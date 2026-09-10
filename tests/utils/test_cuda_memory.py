"""Physical memory ownership; device totals cannot prove per-process parking."""

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from vrl.utils.cuda_memory import gpu_process_used_bytes


@pytest.fixture
def nvml_probe(monkeypatch):
    import pynvml

    state = {"entries": [SimpleNamespace(pid=os.getpid(), usedGpuMemory=1234)], "closed": 0}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid="GPU-selected-by-cuda")
    )
    monkeypatch.setattr(pynvml, "nvmlInit", lambda: None)

    def close():
        state["closed"] += 1

    def handle(uuid):
        assert uuid == "GPU-selected-by-cuda"
        return "selected-handle"

    monkeypatch.setattr(pynvml, "nvmlShutdown", close)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetHandleByUUID", handle)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetComputeRunningProcesses", lambda _: state["entries"])
    return state


def test_other_process_memory_does_not_change_owner_reading(nvml_probe):
    nvml_probe["entries"].append(SimpleNamespace(pid=os.getpid() + 1, usedGpuMemory=1000000000))
    assert gpu_process_used_bytes("cuda:2") == 1234
    nvml_probe["entries"][1].usedGpuMemory = 0
    assert gpu_process_used_bytes("cuda:2") == 1234
    assert nvml_probe["closed"] == 2


@pytest.mark.parametrize("entries", [[], [None], [-1], [2**64 - 1], [1, 2]])
def test_missing_or_ambiguous_accounting_never_certifies_zero(nvml_probe, entries):
    nvml_probe["entries"] = [
        SimpleNamespace(pid=os.getpid(), usedGpuMemory=value) for value in entries
    ]
    with pytest.raises(RuntimeError, match="NVML"):
        gpu_process_used_bytes()
    assert nvml_probe["closed"] == 1


def test_cpu_does_not_require_nvml(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert gpu_process_used_bytes() == 0
    assert gpu_process_used_bytes("cpu") == 0


@pytest.mark.gpu
def test_real_cumem_parking_with_another_process_allocation():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("vllm.device_allocator.cumem")
    code = textwrap.dedent("""
        import gc, subprocess, sys
        import torch
        from vrl.utils.cuda_memory import CumemPool, gpu_process_used_bytes
        pool = CumemPool.require('process-ownership-acceptance')
        warm = torch.ones(1, device='cuda').cpu()
        torch.cuda.empty_cache()
        baseline = gpu_process_used_bytes()
        with pool.building():
            value = torch.ones(16*1024*1024, device='cuda')
        expected = value.cpu()
        loaded = gpu_process_used_bytes()
        assert loaded >= baseline + 60*1024*1024
        child = subprocess.Popen([sys.executable, '-c',
            "import torch,sys; x=torch.ones(256*1024*1024,device='cuda'); "
            "torch.cuda.synchronize(); print('ready',flush=True); sys.stdin.readline()"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            assert child.stdout.readline().strip() == 'ready'
            pool.sleep()
            parked = gpu_process_used_bytes()
            assert parked <= baseline + 16*1024*1024, (baseline, loaded, parked)
            # Logical allocation survives CuMem unmapping; it cannot prove release.
            assert torch.cuda.memory_allocated() >= value.numel()*value.element_size()
            pool.wake()
            assert torch.equal(value.cpu(), expected)
            del value
            gc.collect()
            pool.close()
        finally:
            child.communicate('\\n', timeout=30)
        print(baseline, loaded, parked)
    """)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, result.stderr


def test_query_failure_is_not_hidden_and_releases_nvml(nvml_probe, monkeypatch):
    import pynvml

    def fail(_):
        raise RuntimeError("injected driver query error")

    monkeypatch.setattr(pynvml, "nvmlDeviceGetComputeRunningProcesses", fail)
    with pytest.raises(RuntimeError, match="injected driver query error"):
        gpu_process_used_bytes()
    assert nvml_probe["closed"] == 1
