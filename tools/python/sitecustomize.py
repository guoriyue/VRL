"""Interpreter-startup hook for every Bazel Python target that depends on vrl.

``site`` imports this module in each process that starts with the runfiles
venv on its path, including subprocesses (torchrun ranks, reward workers), so
the locked NVIDIA libraries are in place before any of them imports torch.
"""

try:
    from tools.python.nvidia_preload import preload

    preload()
except Exception:  # never break interpreter startup over a preload problem
    pass
