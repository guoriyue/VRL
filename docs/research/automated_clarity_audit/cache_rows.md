# Cache-row owner and NN package boundaries

Reviewed complete nn/layers/attention/cache_rows.py and five package initializers
(nn, kernels, layers, layers/attention, modules), cache tests, native-backend
consumption and token envelope/GLM callers. Previous audit commit: ec3289b54.
Disposition: retain implementation, with the optional-import follow-up below.

## Retain and why

- ARCacheRows already groups row storage, selection and update operations. Class
  methods share recursive tensor/container and HF-cache conversion with callers
  needing conversion without persistent row storage. Splitting those operations
  into more standalone files would add navigation without clearer ownership.
- Tensor batch/rank checks apply recursively to both HF key/value layers, not only
  the first key. Concatenation refuses dtype promotion and mismatched container
  keys/lengths; torch.cat alone cannot enforce the whole nested cache contract.
- _cache_kv_pairs/_cache_from_kv_pairs isolate Transformers cache API differences.
  They are shared across split/concatenate and preserve supported DynamicCache
  objects, rather than pretending all caches are legacy tuples.
- scatter prepares row values before publishing them. scatter_rows preserves a
  public already-split API with separate count validation; even though its current
  direct coverage is tests, deleting it would be an API change unrelated to this
  audit. Selection returns row references, not deep copies.
- Per-call index validation protects the mutable owner independently of token
  protocol construction. Duplicate indices are allowed here; TokenStepBatch has
  its own unique scheduled-row requirement. Do not conflate those contracts.
- Package initializers deliberately avoid eager class exports. The one observed
  `from vrl.nn.modules import ar_attention_backends` import addresses a submodule,
  not an exported class. Empty __all__ lists express facade policy; no ALL_CAPS
  business table needs extraction.

Non-goals: universal mapping-subclass support, cloning every row to avoid aliasing,
unchecked row-selection objects, or restricting this general cache owner to a
single family's scheduler invariant.

## Validation and limitations

38 cache-row and native-backend tests passed on CPU, covering nested reorder/update,
actual DynamicCache contents, every-layer batch mismatch, scalar rejection and
dtype-promotion rejection. Native tests use trunk doubles. No CUDA memory or
pretrained parity claims follow. No Python change or new redundant test was needed.

The optional-import issue identified in this review is closed by the follow-up
below. Missing Transformers and broken installed imports now have distinct outcomes.

DynamicCache reconstruction preserves key/value tensors through a default cache
constructor; arbitrary extra cache metadata, specialized subclasses or sliding
cache policies are not proven preserved. Unknown non-container values are shared
by reference on split and must compare equal on concatenate. These are existing
supported-shape limits, not reasons to add generic introspection fallbacks.

## Follow-up: optional dependency failure handling

Following 88a85474e, catch only ModuleNotFoundError naming the top-level
transformers package. Missing cache_utils, missing transitive dependencies and
ImportError for an incompatible exported API propagate unchanged. The previous
broad catch falsely disabled HF cache support for all of those failures.

The regression executes the actual module in an isolated module namespace with
controlled import errors. Three broken-installation cases failed on the old code;
the absent-package control successfully gathers real tensor rows without HF cache
support. No shared environment packages were changed or uninstalled. The test
preserves the original module used by other tests and restores the import hook.

Compatibility: environments with an installed but broken Transformers now fail
at import instead of silently using the plain-value path. Environments without
Transformers retain plain tensor/container operations. No cache math, shape/dtype
validation or production public API changed. This narrows an exception boundary;
it introduces no new checker class or per-call validation.

60 cache-row, native-backend and token-loop tests passed on CPU after this change.
Ruff check and format check passed for both changed Python files.
