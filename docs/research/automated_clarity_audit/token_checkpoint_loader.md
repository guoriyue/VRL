# Token checkpoint loading and shard lifetime

Reviewed complete steps/token/loader.py, ARReplayCore.from_pretrained, NextStep
pipeline/replay directory callers, checkpoint adapter tests and NextStep loading
tests. Previous audit commit: 12ed74d0e.

## Change

Explicitly delete core_state and shard_state after installing each shard. Python
evaluates the next load_state_dict call before assigning its result to shard_state;
previously both local dictionaries retained the previous shard during that call.
This could overlap two shards' source tensors, despite loading one shard per loop.
The model's copied parameters remain intact. No shard ordering, format preference,
key selection, strictness or tensor values change.

## Retain and why

- resolve_hf_checkpoint_dir is a shared optional-Hub adapter for replay cores and
  NextStep's upstream pipeline, which requires directory paths rather than modules.
  Its local-directory versus Hub choice is an input-source distinction. Keep the
  free function; a stateful resolver would duplicate upstream cache ownership.
- load_ar_replay_checkpoint adapts full-generation checkpoint formats to a minimal
  replay core. The index selects relevant shards, then module-owned keys select
  tensors within each shard. Those are different filters, not redundant checks.
- Index key coverage is checked before shard I/O; loaded key coverage is checked
  afterward because an index can name a tensor absent from the actual shard.
  Per-shard strict=False permits partial module updates; overall coverage prevents
  silently retaining random missing parameters. Single-file loading can check all
  keys before using strict=True.
- _StateDictModule is a structural protocol boundary that avoids importing torch
  eagerly. Its short methods are interface declarations, not utility wrappers.
- SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME and WEIGHTS_NAME
  come from Transformers' file-format contract. __all__ is an API export list.
  Do not move these names into another local table or invent format guessing.

Non-goals: changing checkpoint formats, materializing all shards for a transactional
load, adding a loader class, clearing CUDA allocators, or promising reduced process
RSS solely because Python references are released.

## Validation and remaining boundaries

13 checkpoint adapter and NextStep loading/decode tests passed on CPU; Ruff check
and format check passed for both changed Python files. The new regression checks
weak references to owned and ignored shard tensors at the beginning of the next
load, and verifies both copied model weights afterward. It failed on the original
implementation. It uses real tensors and nn.Module copying, with disk reads
substituted to isolate lifetime; it does not measure multi-GB checkpoint RSS.
Custom load_state_dict implementations that retain source tensors can still hold
memory themselves; deleting loader references cannot override that ownership.

Sharded loading is not transactional. A missing actual tensor, shape mismatch or
later I/O error can leave earlier shards installed; the caller normally discards
the failed construction. Current filtering also accepts any owned key present in
a selected shard, not only keys mapped to that exact shard by the index. Malformed
duplicate/misassigned index contents need a deliberate format-contract decision.

Deferred path review: index-provided shard names are joined directly to the source
directory. They do not yet use require_checkpoint_source_member, the existing
shared POSIX-member validator in models/checkpoint_identity.py. Its lexical
absolute/parent-segment contract and Hub cache symlink compatibility should be
checked together before consolidating this boundary; do not add a second ad-hoc
safe-path helper. resolve_hf_checkpoint_dir likewise does not expand '~' or
distinguish a nonexistent intended local path from a Hub ID. Current subfolder
arguments come from replay-core declarations, not arbitrary request data.
