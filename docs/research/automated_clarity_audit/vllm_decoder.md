# vLLM decoder packing and cache ownership

Reviewed complete nn/modules/ar_decoder.py, attention request contracts,
executor runner construction, scoped kernel initialization and CPU packing tests.
Previous audit commit: 4339b1a81. Kernel implementation and executor-wide lifetime
review remain separate pending work.

## Changes

- Remove _validate_runtime_tensor and its two calls: both typed input constructors
  already require rank-three embeddings. Remove the repeated step state-count
  check already owned by ARAttentionStepInput. Internal packing now trusts those
  constructor invariants rather than introducing another assertion layer.
- Remove the comparison of nonzero count with length computed from the same
  boolean mask's sum. Nonempty and contiguous-span checks remain; they establish
  different facts needed to compact prompts without losing their positions.
- Correct _pack_prefill's return annotation to six tensors plus states. The
  implementation and consumer already unpacked seven values; runtime unchanged.

## Retain and why

- _pack_prefill compacts each contiguous valid span, uses compact cache positions
  but retains original rotary position offsets, and reserves prompt plus token
  budget blocks. _pack_step enforces one new token and checks remaining physical
  capacity. These are distinct packing algorithms, not trivial wrappers to merge.
- _typed_states rejects states from another backend before reading physical block
  ownership. Input dataclasses only know the number of opaque states, not their
  concrete backend type. Keep this boundary and the CUDA/half-precision kernel
  restrictions; type annotations cannot express either hardware capability.
- The decoder owns residual/norm/MLP and QKV/output projections; the kernel owner
  handles block tables, slots and FlashAttention. _VllmAttentionScaleShim is a
  framework adapter supplying required scale buffers, not a business object.
- _ensure_kv_caches allocates enlarged per-layer caches into a separate list,
  copies old cache content, then publishes it. Retain this preparation-before-
  assignment shape; mutating each live cache as allocation progresses weakens
  failure behavior. Growth may temporarily retain old plus new allocations.
- Attention dimensions come from installed attention attributes when available,
  otherwise from trunk config. Existing tests prove an available attribute does
  not eagerly access absent fallback config. This is an upstream version adapter;
  deleting the fallback without checking supported model versions is not cleanup.
- Rotary helpers express the same tensor operation for query/key; retain their
  mathematical separation. __all__ is the module's API list, not domain vocabulary.

Non-goals: changing rotary math, introducing a free-list allocator, weakening
kernel dtype checks, or claiming mutable tensors cannot be resized after request
construction. Such external mutation is outside the trusted packing path.

## Validation and limits

21 CPU request-contract, decoder packing and scheduled Janus loop tests passed.
Ruff check and formatting passed for the changed module. The tests cover existing
input rejection, left/right contiguous prompt compaction, position offsets and
recording-backend scheduling. No new tests merely asserting removed implementation
details were added. CUDA/vLLM forward parity was not executed.

Remaining boundaries: prefill allocates logical block IDs while packing; later
failure does not roll counters back. Cache capacity grows monotonically on a
backend instance with no row-release API. The executor constructs backends when
building runners, so this alone is not evidence of a process-lifetime leak, but
long-lived/reused runner capacity requires its lifecycle review. Device/dtype and
layer-layout stability are assumed once caches/attention implementations exist.

Output request/result count validation remains the deferred issue recorded in
attention_backend_contract.md. This cleanup removes duplicated input checks only;
it does not claim to solve output validation or partial KV writes on kernel failure.
