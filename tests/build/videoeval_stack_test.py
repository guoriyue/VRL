"""The VBench evaluation stack resolves from Bazel: its pinned transformers and
the source-built tokenizers import together, and VBench itself imports."""

import unittest

import tokenizers
import transformers
import vbench  # noqa: F401


class VideoEvalStackTest(unittest.TestCase):
    def test_pinned_versions(self):
        self.assertEqual(transformers.__version__, "4.33.2")
        self.assertEqual(tokenizers.__version__, "0.13.3")
        tokenizer = tokenizers.Tokenizer(
            tokenizers.models.WordLevel({"a": 0, "[UNK]": 1}, unk_token="[UNK]")
        )
        self.assertEqual(tokenizer.encode("a").ids, [0])


if __name__ == "__main__":
    unittest.main()
