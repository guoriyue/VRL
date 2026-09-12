# vLLM kernel API adapter

Reviewed complete kernels/attention/vllm_paged.py and its package initializer,
decoder call sites, injected-module forwarding/import tests and the separate
real-CUDA KV write test. Previous audit commit: 3427c66a3.

## Change

Correct obsolete class documentation: the shared decoder executes trunk
projections, residuals, normalization and MLPs. Family runners do not patch
attention layers as the old explanation claimed. Runtime behavior unchanged.

## Retain and why

- VllmPagedAttentionKernels owns imported modules and backend config. Its short
  methods isolate unstable upstream calls from decoder math; replacing them with
  scattered direct imports would duplicate keyword and output-layout knowledge.
- _REQUIRED_MODULES is an isolated dependency/protocol list, not workflow business
  vocabulary. Named module properties keep internal paths out of every method.
  __all__ is the public API declaration; the initializer adds no eager re-exports.
- Import failure is wrapped with the failing module, exception type, message and
  cause. It does not switch implementations or discard ABI/dependency failures.
  Kernel constructor/forward errors propagate without signature guessing/retry.
- Block-table creation forwards resource limits and explicit kernel block size;
  only None chooses the configured default. The existing zero-value forwarding
  test verifies it is not silently replaced by a truthiness fallback; it does not
  establish zero as an accepted real-vLLM block size.
- compute_slot_mapping returns the active prefix of the block table's GPU tensor.
  KV update and FlashAttention forward are separate operations with different
  mutation/output contracts. Keep them separate, as used by the decoder and the
  real kernel test.
- run_flash_attention allocates an output only when none was supplied and flattens
  the returned head dimensions for the decoder's output projection. It does not
  swallow forward errors or substitute an uninitialized supplied output.

Non-goals: version-probing fallback, generic kwargs-only wrappers, removing typed
API arguments just for shorter code, changing cache dtype/attention math, or
claiming broader vLLM compatibility than the concrete signatures support.

## Evidence and limitations

The five import/forwarding tests passed in the immediately preceding selector
validation (11 tests total); this turn changes documentation only. Ruff check and
format check passed. A fresh CPU Python process confirmed importing this module
leaves vllm and its submodules absent from sys.modules. Constructor injection
tests use fake upstream modules; they are not ABI verification.

The real-CUDA test inspected here verifies actual block-table slot mapping and KV
cache writes, but does not itself call run_flash_attention. Separate Janus/
NextStep GPU backend parity tests are the forward evidence. None were run in this
CPU audit, preserving the parallel manual process's GPU resources. Upstream API
compatibility and CUDA numerical parity therefore remain unverified here.
