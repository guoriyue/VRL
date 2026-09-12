# Checkpoint restoration and completed module review

Following f414c9748, reviewed the remaining restoration, schema/identity,
state export/load, RNG, persisted config, metadata and checkpoint discovery code.
Together with checkpoint_positions.md and checkpoint_publication.md this covers
checkpointing.py. Related RNG/size fixes are recorded in earlier branch commits.

Change: use runtime state_dict keys directly in both state classifiers instead
of converting each module-owned string name with str(). No incoming checkpoint
keys or tensor values are coerced by this change. Correct the full-state loader
comment: it checks all roots/keys before loading, while the restore entry point
also checks tensor shapes. Direct calls to that public loader do not gain a
transactional guarantee from its key validation.

Retain and why:

- Schema v1 full state includes frozen base parameters; schema v2 contains exact
  checkpoint-owned state. The existing full-state test restores a changed frozen
  parameter as well as trainable/registered state. Dropping the full-state branch
  would change which base model a legacy checkpoint restores.
- Legacy compiled-prefix migration is scoped to v1. Its existing regression
  obtains keys from torch.compile(module).state_dict(), not a guessed directory
  name. V2 keeps its declared key contract. No new format heuristics are added.
- Selective v1 restore requires verified identity in strict mode, because omitted
  base tensors come from the runtime model. Full versus selective classification
  concerns serialized model coverage, not inference of a training position.
- Shape validation happens before any root is loaded through restore. DTensor
  global shapes are appropriate here; dtype coercion remains PyTorch's load
  behavior. Do not add dtype gates or hypothetical tensor-type test matrices.
- Model restore is a public shared boundary for training and evaluation.
  Wan HPSv3 evaluation validates sidecars before constructing the model and then
  validates/loads each authoritative payload. These checks have different inputs
  and lifecycle costs; they are not interchangeable duplicate calls.
- Single-device strategy loaders delegate to public checkpoint loaders; FSDP
  loaders scatter through their own framework seam. Keep public loader checks
  and facade functions rather than assuming every caller entered via restore.
- Shallow state mapping copies protect the loaded checkpoint from DCP replacing
  leaves. Strict post-load comparison checks the owned projection, not all frozen
  tensors of an old full-state checkpoint. No broader guarantee is claimed.
- RNG helpers serve process and explicitly named generator state. Config helpers
  serve shared persisted-run readers/writers with lazy OmegaConf/config imports.
  Metadata helpers are a shared sidecar protocol. Their free-function shape does
  not justify adding a checkpoint-manager class.
- Schema versions, filenames and strict default are real protocol/API constants.
  The metadata progress-name tuple is a serialized field list. Keep these near
  their readers/writers; there is no workflow business vocabulary to extract.
- Discovery uses explicit global_step metadata and stable path ordering for ties.
  It does not derive progress from checkpoint directory names. Size checks detect
  truncated copies cheaply; they are not digest or deserialization validation.

Non-goals: retire supported schemas, change strictness, add another validation
layer, add path/RNG wrapper classes, change training math, or guarantee rollback
after framework load hooks fail. Existing publication durability limits and the
offline accumulation-window follow-up remain documented in the linked reviews.

Validation: the existing checkpoint test module passed 122 tests with one skip
and 14 dependency warnings on CPU. Ruff check and format check passed for the
changed Python file. No tests or runtime gates were added; this does not establish
GPU or multi-rank recovery. Coverage is now 190 reviewed and 309 pending modules.
