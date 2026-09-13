"""Verify the managed interpreter imports declared VRL sources without a venv."""

import pathlib
import sys
import unittest

from vrl.utils.validation import require_int


class PythonToolchainTest(unittest.TestCase):
    def test_managed_interpreter_and_declared_sources(self):
        self.assertEqual(sys.version_info[:3], (3, 12, 13))
        self.assertNotIn(".venv", pathlib.Path(sys.executable).parts)
        self.assertEqual(require_int(3, path="sample_count", minimum=1), 3)


if __name__ == "__main__":
    unittest.main()
