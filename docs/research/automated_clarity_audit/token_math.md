# Token math and package boundaries

Following 9b349b510, reviewed math/__init__.py, math/token/__init__.py,
math/token/logprob.py and math/token/flow_matching.py. Followed categorical
scoring through ReplayModel and LlamaGen/GLM runners, and continuous-token
sampling/replay through the NextStep runner and model.

Change: qualify the categorical scorer's memory claim. chunk_size bounds each
normalization temporary over token positions, not total autograd graph storage.
A one-off CPU saved_tensors_hooks run with 8 positions, vocabulary 16 and chunk 2
observed vocabulary-shaped tensors saved across all chunks; backward completed.
These are saved references, not a measurement of unique allocations or CUDA peak
memory. Correct the existing test docstring that called this vocabulary-axis
chunking. Numerical implementation and tests themselves are unchanged.

Retain and why:

- top_k_top_p_filtering is an in-place sampling transform; inspected runners
  provide 2D batched logits and clone them before applying sampling filters.
  Replay scores the conditional temperature-scaled logits separately. Do not
  unify the two operations merely because both consume logits.
- gather_categorical_log_probs shares stable FP32 normalization and selected-ID
  scoring across families and the replay interface. Integer device/dtype movement
  is required for gather, including supported int32 recorded IDs; no float-ID
  validation or hypothetical invalid-token test was added.
- require_positive_temperature expresses the categorical policy boundary. It is
  also used by samplers and the fused scorer. A generic numeric checker would
  obscure why greedy zero-temperature decoding is a separate policy contract.
- _flow_terminal_mean, _flow_noise_std and _isotropic_gaussian_logprob remove
  actual duplication between continuous-token generation and replay. They share
  Euler/CFG integration, the final noise scale and dimension-summed density.
  Keep them as pure numerical functions, not a new sampler/checker class.
- NextStep's runner records the initial prior; the model passes that same prior
  into replay. Scoring cannot redraw it. The head's .net call and input_dim are
  the inspected adapter contract; do not replace them with speculative fallbacks.
- The sample helper detaches the sampled target for log-density evaluation;
  replay differentiates the current mean. Keep this existing autograd behavior.
- Both package roots are lightweight boundaries. Token __all__ is intentionally
  empty; submodule exports are public API names. There is no module-level ALL_CAPS
  business vocabulary in the reviewed math implementations.

Non-goals: alter density normalization, terminal-noise convention, sampling
distribution, Euler integration, introduce checkpointed/custom backward math,
re-export every helper or add speculative dtype tests. This review verifies VRL's
implemented continuous-token policy and call sites, not an independent upstream
paper/model-card equivalence claim from the historical module header.

Limits: top-p filtering uses its 2D batch/vocabulary contract; it does not promise
arbitrary leading axes. The full logits tensor is already an input to the
categorical scorer; chunking does not remove that allocation. Production runner
conditioning/prior ownership, not annotations alone, establishes replay parity.

Validation: 46 existing token-flow math, categorical log-prob, NextStep runner and
GLM sampling contract tests passed on CPU. Existing tests compare categorical
values with full log-softmax and continuous sample/replay densities with the same
prior. Ruff passed for changed files. No new tests or gates. Coverage is now 198
reviewed, 301 pending baseline modules.
