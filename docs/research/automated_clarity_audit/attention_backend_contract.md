# Attention request contract and native decoder adapter

Reviewed complete nn/layers/attention/paged.py and nn/modules/torch_attention.py,
their request/result callers, native builder, scoped vLLM result construction,
and complete input/native-backend tests. Previous audit commit: c9d5160fa.
The larger ar_decoder.py and backend selector remain pending full review.

## Retain and why

- ARAttentionBackend separates family token loops from cache implementations.
  Input/result dataclasses carry different prefill/step facts; they are protocol
  boundaries, not redundant wrappers around one model call. The layer imports no
  vLLM internals. ARAttentionUnavailable classifies backend initialization failure.
- _require_embed_mask_batch is shared input geometry validation. It intentionally
  checks rank and batch, not equal sequence length: decode embeds can be [B,1,H]
  while the attention mask covers all past tokens plus the current token.
- Positive block_size and max_new_tokens validation protects physical cache
  reservation sizes. Step state count must equal its input batch. These are
  external constructor constraints, not arbitrary checks repeated in inner math.
- The native backend owns its trunk, label and forward protocol. Prefill splits
  the batched cache into row states; step concatenates selected row states in
  scheduler order and splits the updated cache again. It already shares the
  repository's ARCacheRows implementation; another cache adapter would duplicate it.
- _forward is shared between prefill/step and verifies that use_cache actually
  produced a cache. _last_token_hidden handles the two supported HF output
  surfaces: last_hidden_state or the final hidden_states entry. This is an
  upstream output adapter, not catching a failed tensor operation and guessing.
- output_hidden_states=True supplies the latter output surface. Removing it for
  all trunks merely because some expose last_hidden_state would break the other
  supported surface. Avoid family-name dispatch to optimize this assumption.
- __all__ lists public names; no mixed-in ALL_CAPS vocabulary needs extraction.
  The config dataclasses and backend subclass retain consistent protocol shapes.

Non-goals: adding a generic validation class, combining input and output types,
changing cache layout, making GPU claims from CPU tests, or automatically falling
back to the native backend after a vLLM runtime failure.

## Output validation decision and remaining scope

An output dataclass alone cannot prove its batch matches the request. Checking
len(sequence_states) against last_hidden.shape[0] would still accept two mutually
consistent but incorrect lengths. Do not add that partial check and claim the
runner issue fixed. A complete consolidation must validate returned rows against
the original request before runners publish state, with the production backend
and custom/test backend contract handled consistently. The findings in
paged_cfg_runner.md remain deferred to that backend-call ownership work.

Native hidden extraction takes the last physical sequence position. Its intended
prompt layout must therefore leave the final position valid (e.g. left padding);
this module does not provide general right-padded last-valid-token selection.
No new padding inference is added. Arbitrary custom trunks/outputs remain limited
to the documented HF shape; their hooks can retain cache or hidden storage beyond
this adapter's lifetime.

## Validation

16 input-contract and native-backend tests passed on CPU. They use real tensors and
row-identity/row-independent trunk doubles to verify reordered subset caches,
batched versus separate row stepping, missing-cache failure and builder forwarding.
They do not execute a pretrained HF trunk or vLLM kernels. No Python changes were
needed for the reviewed modules; no formatting or speculative checks were added.
