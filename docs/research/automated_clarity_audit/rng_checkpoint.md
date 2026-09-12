# RNG checkpoint error ownership

Scoped follow-up to the sampling review, previous commit e0e7a6fb1. Inspected
capture_rng_state, restore_rng_state, checkpoint payload assembly, online and
offline restore call sites, and sampler resume tests. Full checkpointing.py
review remains pending; this does not increment complete-module coverage.

## Change

Remove catch-all exception suppression around Python random getstate/setstate
and NumPy get_state/set_state. A state-operation failure now reaches the caller
instead of publishing an incomplete snapshot or silently resuming with another
random sequence. Python random is standard library, not optional.

Capture still omits NumPy only when importing the numpy package itself raises
ModuleNotFoundError naming numpy. An internal missing dependency or other broken
import propagates. Restore imports NumPy normally when a saved numpy field is
present: a checkpoint requiring that state cannot be restored successfully
without the provider. State values are passed to the provider's own loader;
there is no parallel hand-written RNG schema validator.

Compatibility: absent state/fields remain accepted as before. Existing valid
snapshots retain their values and sampling order. Environments missing NumPy can
still capture other providers, but restoring a snapshot containing NumPy state
now fails there. Malformed provider state and provider implementation errors no
longer disappear, including during non-strict trainer/model resume: these caller
paths restore RNG separately from the trainer's optional-state policy.

## Retain and why

- The two functions coordinate process-global providers plus named external
  generators; they do not own one model or trainer instance. Both online and
  offline recipes use this shared boundary. A stateful wrapper class would not
  remove complexity, and callbacks around each provider would obscure the calls.
- torch, cuda, python_random, numpy and generators are saved schema keys, not
  workflow business constants. Keep their existing spelling and structure.
- Named prompt generators are distinct from process-wide training randomness.
  Preserve named-state restoration rather than deriving prompt progress from
  an optimizer step counter or re-seeding on resume.

Non-goals: making all provider restoration transactional, changing optional
legacy field handling, altering seeds, moving process RNG into a model object,
or changing GPU-state restoration on a host without CUDA. A later-provider
failure may occur after earlier RNG providers were restored; failure is now
visible, but rollback is not claimed.

## Validation

All four injected Python/NumPy capture/restore error regressions failed before
the edit because the expected original exception was swallowed. They pass now.
Additional cases distinguish an absent NumPy package from a broken dependency
and require the provider for a saved NumPy state. A real roundtrip verifies the
next Python, NumPy and named torch generator draws and restores process state
after the test. Existing sampler preview/resume behavior remains covered.

146 checkpoint and prompt-sampler tests passed, one skipped. Ruff check/format
passed on both changed Python files. Tests used CPU-only visibility; no CUDA RNG
or multi-rank determinism claim is made.
