# Trainer diagnostic selection and scope

Reviewed full trainers/diagnostics.py, online first-step/replay diagnostic callers
and utility/integration tests. Previous audit commit: 04b9d51b7.

## Change

trainable_state_digest now selects named parameters by requires_grad directly.
Previously an empty selection fell back to state_dict, guessed names containing
lora_, then selected all tensors if that guess was empty. A fully frozen model
therefore produced a changing "trainable" digest when frozen weights changed.
Remove this fallback rather than adding another inference rule or name table.

The regression failed before the edit because mutating a frozen Linear changed
the result. It now returns the stable empty selection with tensor_count/numel=0.
Existing real trainable parameter digests and DTensor behavior are unchanged.

Compatibility: this function now requires named_parameters, as all inspected
production callers provide. State-dict-only external objects no longer receive
a guessed digest; callers must supply an explicit parameter-bearing module.
parameter_state_summary's separate absent-API behavior remains unchanged.

## Retain and why

- These diagnostic functions operate on caller-supplied values and hold no
  persistent model state. A diagnostic class would just namespace stateless
  operations without improving their ownership.
- _local_diagnostic_tensor is an actual distributed boundary: it uses only an
  already-owned shard/replica, rejects Partial placements and never gathers full
  parameters. Keep it separate from the digest loop to make communication scope
  explicit. It is not an ordinary redundant tensor assertion.
- parameter_state_summary reports registered/trainable counts and devices;
  trainable_state_digest inspects trainable values. They answer different
  questions, so neither can replace the other with fewer output fields.
- tensor_stats intentionally summarizes values, including empty/singleton
  tensors, for JSON diagnostics. append_jsonl_record's nested traversal handles
  mappings, lists, primitive values and tensor summaries. It is not the same
  contract as a tensor-preserving device-tree conversion; do not force it into
  that abstraction merely because both recurse.
- Diagnostic append is a stream append, not whole-file atomic publication.
  Replacing it with atomic_file would need to rewrite the existing log and does
  not follow from the shared publication cleanup.
- Result keys and __all__ are schema/public boundaries. There is no ALL_CAPS
  business vocabulary to extract; the removed lora_ substring was an implicit
  selection rule, not a necessary schema key.

Non-goals: changing diagnostic numeric formulas, training math or append format,
introducing global diagnostic ownership, or removing DTensor placement checks.

## Validation and limits

22 diagnostic utility and online integration tests passed. This includes real
two-rank CPU Gloo uneven-shard tests that forbid full_tensor gathering, replicated
DTensor comparison, Partial rejection and first-step diagnostic integration.
Ruff passed for the changed Python files. No CUDA training run was performed.

Digest values are converted to FP32, so the digest is not a byte-exact checkpoint
integrity mechanism for FP64 or large integer values; the docstring now says so.
The scope field distinguishes local DTensor values from plain tensors. Generic
tensor_stats does not declare the same local-DTensor guarantee and is consumed
here for replay signal tensors; do not extend the no-collective digest claim to
every diagnostic helper without inspecting those producers.
