"""A tiny real byte-level BPE tokenizer shaped like VideoScore2's for CPU tests.

The soft-score path searches two spellings of each marker ("quality" and
" quality") and accepts multi-token needles. A stand-in that ``strip()``s its
input collapses both spellings to one id and never emits a multi-token marker,
which defines two production branches out of existence. This builder is a real
``PreTrainedTokenizerFast`` over a real byte-level BPE, so the tokenizer, not the
test, decides the ids. Built from a fixed vocabulary and merge list: no seed, no
download, ~1 ms.
"""

from __future__ import annotations

from typing import Any

# Space-prefixed forms ("Ġ" is the byte-level space) come first so the merge
# chain for " consistency" wins over the bare "cons" + "istency" pieces. The
# bare "consistency" deliberately has no whole-word entry: it tokenizes to two
# pieces, mirroring the real Qwen2 tokenizer, which is what exercises the
# multi-token needle path.
_MARKER_WORDS = (
    "Ġquality",
    "Ġalignment",
    "Ġconsistency",
    "quality",
    "alignment",
    "cons",
    "istency",
)


def build_tiny_marker_tokenizer() -> Any:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    vocab = {token: index for index, token in enumerate(bytes_to_unicode().values())}
    merges: list[tuple[str, str]] = []
    for word in _MARKER_WORDS:
        # BPE can only merge through prefixes that exist in the vocabulary.
        for length in range(2, len(word) + 1):
            prefix = word[:length]
            if prefix not in vocab:
                vocab[prefix] = len(vocab)
            merges.append((word[: length - 1], word[length - 1]))
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer)


__all__ = ["build_tiny_marker_tokenizer"]
