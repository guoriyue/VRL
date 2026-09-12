# Shared atomic artifact publication

Follow-up to b7807dc66. Reviewed the complete existing JSON writer helper, its
callers/tests, the shared artifact utility module and SFT shard producer/reader.
These modules already had full coverage; this closes a recorded publication
finding rather than increasing the module count.

## Change

Move the existing temporary-sibling/fsync/replace-or-link mechanism from the
private JSON callback helper into atomic_file in the existing artifacts module.
The context manager supports UTF-8 text or binary handles. JSON and JSONL write
directly inside it, removing two nested emit callbacks and JSONL's nonlocal
counter. The shard writer uses the same binary publication boundary so failure
does not truncate a previously published shard.

SFT resolves/expands its output path before opening the atomic context. This
matches reader tilde handling and preserves writing through a destination
symlink instead of replacing the symlink itself. Generic atomic_file deliberately
does not choose root containment or tilde policy for all callers; JSON retains
its existing path semantics. Schema and loaded tensor contents are unchanged.

## Retain and why

- A context manager owns a real acquire/write/publish/cleanup lifetime shared
  across formats. A new saver class or callback per format is unnecessary.
- JSON serializers still own formatting, newline, sort order and row counting.
  Torch owns binary serialization. atomic_file owns no knowledge of either.
- overwrite=False retains the existing exclusive hard-link publication, which
  cannot overwrite a previous record even when another writer wins a race.
- RootedPaths remains separate: containment is a path-authorization contract,
  while publication is a file-write lifecycle. Neither guarantees the other.
- Existing schema/env/file/taxonomy constants remain in their legitimate owners;
  no backend table or new business vocabulary is introduced.

Non-goals: migrating every writer at once, adding locks or append transactions,
changing checkpoint directory publication, or claiming symlink resolution is a
race-proof filesystem security boundary.

## Compatibility and limits

The same-directory temporary file makes publication atomic for concurrent
readers under the filesystem's rename/link semantics. Content is fsynced, but
the directory is not: power-loss durability is not guaranteed. Normal errors
remove the temporary sibling; hard termination can leave one. Replacement uses
a new inode and temporary-file permissions, so an overwritten SFT shard does
not preserve its former inode, hard-link peers or custom permission bits.
These are explicit consequences of atomic replacement. An open reader may keep
reading the old inode, while later readers see the new complete shard.

The primitive is torch-free and reused by existing modules, not a new
format-specific thin file. No global path validation/checker class was added.

## Validation

Existing JSON tests retain exact formatting, parent creation, replacement and
exclusive-write behavior. New failure tests preserve previous bytes and leave
no temporary sibling after JSONL input failure, Torch serialization failure and
fsync failure. The shard roundtrip checks actual loaded tensors; a symlink case
checks the link remains and its target receives the shard. The affected reward
artifact-store tests exercise JSON publication consumers too. CPU-only tests;
no crash/power-loss or distributed filesystem guarantees are inferred.

All 39 focused tests passed; Ruff passed for the five changed Python modules.
