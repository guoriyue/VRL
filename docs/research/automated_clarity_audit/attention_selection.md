# Attention backend selection

Reviewed complete nn/modules/ar_attention_backends.py, executor request projection,
sampling schema vocabulary, native builder tests and scoped vLLM import-failure
tests. Previous audit commit: b4124bb33.

## Change

Describe the native builder as explicitly selected rather than a "fallback".
Selection does not catch vLLM errors or switch implementations. Runtime unchanged.

## Retain and why

- _ATTENTION_BACKENDS is a deliberately isolated two-name diagnostic vocabulary
  in the backend selection module. It belongs here; a config asset or registration
  framework would add more machinery than the two explicit branches require.
- build_attention_backend selects between two public builders. The vLLM builder
  projects block/cache options; the native builder projects only family identity.
  Direct builder imports exist in tests/tools, and this uniform shape keeps each
  implementation's constructor inputs visible. No static-method namespace needed.
- _lm_trunk is the shared structural model boundary. It checks callable presence,
  invokes the hook once and propagates failures; it does not guess language-model
  attributes or fall back to the full model. Family-specific trunk ownership stays
  on the model.
- attention_backend_name centralizes the request default. An absent key selects
  vllm_paged; explicit values are not rewritten after a runtime failure. Invalid
  names fail in the selector before backend construction.
- __all__ lists public APIs. Internal vLLM imports occur when its kernel owner is
  constructed, not merely because the native selector imports the Python decoder
  class. The import gate preserves the failing module and original exception.

Non-goals: silently substituting native attention for missing vLLM, adding a
backend registry class, changing the default backend, or normalizing arbitrary
None/empty values into defaults beyond the current request contract.

## Validation and limits

11 selector, native-cache adapter and kernel import/forwarding tests passed on
CPU. Unknown-name/default behavior, native trunk calls and chained vLLM import
errors are covered. Kernel forwarding tests use injected modules; no CUDA/vLLM
ABI or numerical parity was verified. Ruff check and format check passed for
the changed Python file. The larger kernel wrapper remains pending full review.
