"""Verify the locked torch wheel runs on the GPU from Bazel-managed dependencies only."""

import os
import pathlib
import re
import sys
import unittest

import torch

from tools.python.nvidia_preload import preload

# sitecustomize already ran the preload at interpreter startup; this returns
# what it loaded.
PRELOADED = preload()


class TorchCudaTest(unittest.TestCase):
    def test_torch_matches_lock_and_runs_on_gpu(self):
        self.assertEqual(torch.__version__, "2.11.0+cu130")
        self.assertEqual(torch.version.cuda, "13.0")
        self.assertNotIn(".venv", pathlib.Path(torch.__file__).parts)
        self.assertNotIn(".venv", pathlib.Path(sys.executable).parts)
        self.assertIsNone(os.environ.get("CUDA_HOME"))
        self.assertTrue(torch.cuda.is_available(), "GPU test requires a visible NVIDIA device")
        x = torch.randn(256, 256, device="cuda")
        y = (x @ x.T).sum().item()
        self.assertTrue(abs(y) < float("inf"))
        self.assertEqual(torch.cuda.get_device_capability()[0] >= 8, True)

    def test_cuda_libraries_come_from_the_lock_not_the_host(self):
        self.assertTrue(PRELOADED)
        maps = pathlib.Path("/proc/self/maps").read_text()
        cuda_libs = sorted(
            set(re.findall(r"\S+/lib(?:cudart|cublas|cublasLt|cudnn)\.so\S*", maps))
        )
        self.assertTrue(cuda_libs)
        for lib in cuda_libs:
            self.assertNotIn("/usr/local/cuda", lib)
            self.assertNotIn("/usr/lib", lib)


if __name__ == "__main__":
    unittest.main()
