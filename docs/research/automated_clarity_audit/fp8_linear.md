# FP8 module and optional kernel boundary

Reviewed complete nn/quantization/fp8.py, config recipe declaration, loader dispatch
and compile guard, shared master lifecycle, and FP8 targeting/sync/import tests.
Previous audit commit: 318ae2c78.

## Change

The lazy blockwise kernel import now catches only ModuleNotFoundError naming the
top-level vllm package. That absence retains the actionable install/rowwise message
and original cause. Missing internal modules, transitive dependencies and removed
exported kernel APIs propagate their original exceptions. Previously all of these
were wrapped as "vLLM is not installed", misdiagnosing broken installations.

The regression calls the real module forward with an isolated import hook; it
does not uninstall shared packages, initialize CUDA or replace the quantization
implementation. Three broken-install cases failed before the change; the absent
package control passed. All four pass after the fix. The hook is restored by the
test fixture.

## Retain and why

- Fp8Linear owns scheme-specific quantization and forward; QuantizedLinear owns
  cache movement and state loading. There is no additional detached lifecycle
  helper to consolidate. Weight caches are non-persistent derived state, keeping
  the original weight/bias checkpoint keys and adapter-only synchronization.
- _amax_scale is shared by weight and activation quantization. Keeping one small
  method expresses the same scaling rule at both call sites without a new class.
- _blockwise_gemm is a real lazy optional-dependency/kernel adapter. Rowwise and
  tensorwise do not need vLLM. Removing this boundary or importing vLLM eagerly
  would change dependency behavior; copying its kernel would add implementation.
- FP8_BLOCK is an actual block-layout dimension used for shape eligibility,
  weight reshaping, activation quantization and kernel arguments. ALL_CAPS is
  appropriate; a separate one-constant module provides no additional ownership.
- Non-128-aligned blockwise inputs deliberately use rowwise, exercised by an
  existing test. This is explicit per-module recipe compatibility, not exception
  retry after a failed kernel. Preserve it rather than silently dropping eligible
  modules or changing production quantization scope during clarity cleanup.
- Recipe vocabulary is checked by QuantizationPolicy before loader dispatch.
  Repeating string validation in every forward is unnecessary. Direct constructor
  callers bypass that policy; unsupported recipe strings currently enter the
  tensorwise branch. That direct-API limitation is recorded, not broadened into
  an unrelated validation framework.

Non-goals: change scales, clamping, intermediate/output dtypes, recipe selection,
accuracy tolerances, bias promotion, compile compatibility or model targeting.

## Evidence and limits

45 quantization tests passed, 18 skipped with CUDA hidden. Ruff check and format
check passed for the two changed Python files. Existing CPU tests exercise real
swaps, cache state and adapter-only loading. GPU kernel accuracy, device movement
and compiled behavior were not rerun, and CPU import tests do not validate the
installed vLLM kernel API. Historical performance claims in the module comments
were not remeasured by this audit.

Compatibility changes only the exception presented for a broken installed vLLM:
the original ImportError/ModuleNotFoundError now reaches the caller rather than
a misleading RuntimeError. The requested blockwise path still fails; it does not
silently select a different recipe on import or execution errors.
