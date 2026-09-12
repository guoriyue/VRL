# CausVid artifact and backend loading (scoped review)

Following a79d0d9e2, read model.py's artifact owner, backend constructor/prompt
encoding, released-build validation and all loading/source/import helpers from
_load_official_backend through the end of the file. The model's cache, decode and
replay implementation still requires full review; model.py remains pending.

Changes:

- After strict transformer and text-encoder load_state_dict calls, release the
  temporary checkpoint dictionaries. Otherwise those local references survive
  subsequent device moves and remaining model construction. This shortens their
  lifetime; no host-RSS or full-model GPU measurement is claimed.
- _load_generator_state_dict already verifies every key is a string starting
  with the same model. prefix. Remove the redundant str conversion and impossible
  post-strip duplicate-key check; stripping the same prefix is injective over
  those validated keys. Keep the actual external format check and strict model
  load, using removeprefix to express the operation directly.

Retain and why:

- CausVidResolvedArtifacts.from_build is the construction boundary for verified
  paths. Source/import/backend preflight precedes multi-gigabyte weight downloads.
  Source/base/checkpoint resolution remain separate named operations for different
  external artifacts; a generic safe-path loader would hide those differences.
- _load_official_backend owns the heavy import and distinguishes generation's
  encoder/tokenizer/VAE from replay's transformer-only build. Lazy imports are
  necessary because upstream causal-model import compiles FlexAttention.
- _resolve_checkpoint supports explicit files, directory members and Hub
  references. Member validation applies only when joining a source member; a
  directly selected checkpoint file does not consume that member field.
- _verified_source_revision supports Git checkouts and packaged Ray source trees;
  the latter validates an executable-subset digest. _require_pinned_source_import
  separately ensures Python actually resolves to that source, including already
  loaded modules. Path verification alone cannot establish module provenance.
- _git_revision checks tracked modifications as well as revision. This is not a
  generic path existence check or a replacement for archive digest verification.
- Pinned repositories/revisions/checksums, file members and source include/exclude
  patterns are real external integrity boundaries. Keep these constants; there
  is no large workflow vocabulary or prompt template to move to an asset.
- FlashAttention preflight supports two upstream import names. It stays a lazy
  dependency adapter rather than being merged into arbitrary module import.

Non-goals: change license acknowledgement behavior, rewrite source integrity,
add speculative collision tests, alter precision or move all functions into a
loader class. No tests or gates were added.

Limits/follow-up: FlashAttention's ModuleNotFoundError branch currently treats a
missing transitive dependency like an absent backend and omits its detail from
the final error. Distinguish package absence from broken installation if this
adapter receives a focused failure-path cleanup. Git checks ignore untracked
files by design, unlike the packaged runtime digest; neither is a general
hermetic execution guarantee. Full backend load is not exercised by the CPU lane.

Validation: 18 existing CausVid replay/loading tests passed, including real torch
checkpoint prefix parsing, digest checks and packaged-source subprocess import.
Ruff check and format check passed. They do not measure the lifetime improvement
under full model load. Coverage remains 221/499 until the rest of model.py is read.
