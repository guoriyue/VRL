# Model checkpoint identity review

Reviewed the complete `vrl/models/checkpoint_identity.py` module across this and
the preceding checkpoint inspection. Read source-mutation, schema coverage,
LoRA, member override, default canonicalization and worker-construction tests;
also inspected actual member-path consumers in Anima/CausVid and the worker's
before/after identity checks. No production change is justified in this group.

## Retain the boundary

The identity is not a comparison of all training arguments. It consists of
declared model sources and construction values: remote sources are pinned,
local sources are content-addressed, and non-identity configuration is excluded
by schema metadata. Moving a local file without changing content preserves its
identity. Changing an effective model construction dimension changes identity.
The worker compares this with the driver's expected identity before construction
and again afterward; these checks catch different sources across nodes or a
source changed during loading. They are not redundant validations of one fixed
in-memory value.

## Constants and small functions

- MODEL_IDENTITY_SCHEMA and CHECKPOINT_IDENTITY_METADATA_KEY are persisted
  protocol/schema identifiers. IdentityKind and its derived set are the isolated
  schema vocabulary, not a family-name table. Commit/SHA regexes describe source
  identifiers; `_MISSING` distinguishes absence from an explicit None/default.
- `checkpoint_identity_metadata` is the schema declaration API. `_field_metadata`
  and `validate_checkpoint_identity_schema` interpret and validate that API,
  including source/member/revision relationships. The allowed-key table is the
  metadata grammar. No generic checker class or family-specific registry copy.
- `LocalCheckpointContent.from_path` already owns construction of content facts.
  File streaming and recursive directory traversal are different operations
  sharing one digest and cycle stack. `_stat_signature` and `_raise_changed`
  keep the mutation comparison and diagnostic consistent across file, alias and
  directory checks. Moving them into a stateless hasher class would add an owner
  without simplifying this traversal.
- `require_checkpoint_source_member` is used by actual family loaders as well
  as identity resolution. It validates a POSIX source-member name for both local
  and remote sources; RootedPaths alone cannot replace this remote path grammar.
  Symlinks are deliberately followed as content aliases, so this is not a claim
  that every member resolves physically inside a local root.
- `require_remote_checkpoint_source_pin` is also called by the Wan configuration
  boundary. It distinguishes an existing local source from a pinned Hub source.
  `_resolve_source` additionally hashes local content and tracks aliases. Keep
  those responsibilities distinct, including the local/remote race check.
- `_normalize_identity_value` recursively produces comparable values, while
  `_normalize_identity_field` applies a declared field-specific canonicalizer.
  Sorted-unique target names and ordered general sequences must not be conflated.
  Finite float checks here belong to persisted equality semantics, unlike a
  hypothetical float-token guard inside an already typed tensor operation.
- `_value_for` implements explicit schema/default precedence; `_is_configured`
  shares the optional-source presence rule. Neither derives values from a
  checkpoint filename or guesses training progress. Token LoRA defaults come
  from the same registry projection used by model construction.

## Limits and non-goals

Keep the custom local resolver seam: tests inject mutation and count resolver
calls, so validation of resolver output is an actual external callback boundary.
Repeated source aliases use one content hash per resolution call, avoiding a
second full read of large weights. The cache is not a persistent content cache.

Stat comparisons detect the tested file mutation and symlink retarget cases;
they do not make an arbitrary mutable filesystem an atomic snapshot. In
particular, this review does not prove immunity to changes of already visited
descendants during a later part of a directory walk. Do not advertise stronger
guarantees than the implementation/tests establish. No hashing format, path
acceptance, model identity schema, supported family or construction default was
changed. The larger model loaders remain pending full review.

## Validation

42 existing model-identity, worker-identity and Wan DPO identity tests passed on
CPU. Coverage includes registry schema classification, path independence,
symlinks, special files, mutation, remote pins, family default projection and
construction-time identity mismatch. Worker tests use isolated fakes, not a
multi-node model launch. No new tests for unchanged helper organization.

Previous audit commit: `f21fbe8aa`. This retained-module disposition is committed
with the coverage ledger; source review does not require manufacturing a change.
