# Token adapter installation and separable vocab heads

Reviewed complete token/lora.py, token/vocab_head.py and token/__init__.py,
all five families' adapter installation calls, Janus/GLM split producers,
ARModelBase._head_replay_values, scoped PEFT loader validation and relevant tests.
Previous audit commit: 56ace4e44. No production change required for these modules.

## Retain and why

- install_token_lora_adapter is a genuine shared PEFT adapter used by Janus,
  NextStep, Emu3, GLM and LlamaGen. Family methods still own replacement of the
  correct trunk reference; NextStep additionally updates its pipeline reference.
  A shared object cannot infer those different ownership paths safely.
- Fresh creation passes resolved family config to LoraConfig. Warm start passes
  expected topology to load_trainable_lora_adapter, so checkpoint topology cannot
  silently replace configured intent. CAUSAL_LM is explicitly selected by Janus;
  a default for all families would change PEFT wrapper behavior.
- PEFT imports remain lazy. Import errors gain installation guidance with their
  cause preserved. The broad warm-start catch adds the adapter path and chains
  the exception; it never initializes a fresh adapter after failure. Removing it
  would change the public exception surface already covered by regression tests.
- VocabHeadSplit is the algebraic interface between family-owned head structure
  and fused log-prob evaluation. It holds projection tensors plus an optional
  prefix; it is not another generic validator. from_linear centralizes shared
  eligibility and row slicing rather than requiring each family to copy them.
- The exact plain nn.Linear check is necessary: a subclass/wrapped layer can
  change forward math without changing the apparent weight attribute. Splitting
  a LoRA final layer could omit its delta. Eager logits are the declared fallback
  for an unsplittable head, not an exception-swallowing retry after a broken fused
  calculation. Prefix failures propagate normally.
- Weight/bias row slices are views; they retain gradient and live-parameter
  relationships. GLM restricts to the codebook prefix, Janus can first apply its
  MLP prefix, and Emu3's structural masking keeps the eager path.
- The token package initializer only identifies the package. No facade class or
  eager re-export is needed. __all__ lists are API names; there is no ALL_CAPS
  workflow vocabulary or backend table in these modules to extract.

Non-goals: moving every family installation method into a shared class, unifying
PEFT task types, removing legitimate eager evaluation, adding row-count checkers,
or moving projection math into the checkpoint loader.

## Evidence and limits

45 tests passed on CPU: token adapter loading, vocab-head contract, Janus and GLM
replay, and LlamaGen model construction. The LlamaGen roundtrip creates a real
PEFT adapter, modifies and saves its parameters, then verifies trainable restored
values. Janus/GLM tests compare fused-result log-probs with eager logits on tiny
models. This does not prove every pretrained family or CUDA kernel combination.
No Python edits were made, so no new test or formatting-only changes were needed.

VocabHeadSplit.from_linear's exact type check is deliberately conservative but
does not inspect forward hooks or arbitrary monkey-patched forward methods on a
plain Linear. Inspected producers use ordinary projection modules; no guarantee
for hook-modified projection semantics is implied. Such support would need a
family declaration and numerical tests, not blanket removal of this guard.
