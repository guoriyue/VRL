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

resolve_hf_checkpoint_dir does not expand '~' or distinguish a nonexistent intended
local path from a Hub ID. Current subfolder arguments come from replay-core
declarations, not arbitrary request data.

## Follow-up: shared shard member contract

Following 6be0ed27f, selected index shard names now use the existing
require_checkpoint_source_member helper before opening any shard. This closes
the previously deferred lexical path boundary without another validator/class.
An absolute or parent-segment path is not a member of the declared checkpoint
source. Nested relative paths remain accepted. Only selected shards are validated;
irrelevant generation-only entries remain unopened and do not block minimal replay.

Two regression cases failed on the old loader because external shard paths were
opened instead of rejected. A real safetensors file outside the snapshot, linked
through a nested snapshot member, still loads both actual module weights. This
tests the HF-style cache topology without network or a fabricated loader result.
All 12 checkpoint adapter tests passed after the change; Ruff check/format passed.

Compatibility: malformed selected names that previously escaped the source root
now raise ValueError before module mutation. The check is lexical, not realpath
containment: cache symlinks may point outside the snapshot by design. This is not a
filesystem sandbox or protection against symlink retargeting. The existing helper
still owns its accepted path grammar; no speculative dtype or whole-index schema
validation is added. Its lazy import preserves the loader's dependency boundary.
