# SPRINT: Remove policy-reorganization residue and guard retired source paths

**Date:** 2026-07-18  **Status:** PLANNED

This is the small residue pass after the policy-axis reorganization and the
executed dead-wrapper and single-caller sweeps. It contains two code actions;
the broader audit found no additional unowned cleanup.

## 1. Delete two orphan batch-stacking helpers

`vrl/rollouts/batch/core.py::_stack_extra_values` and
`_stack_training_views` served the deleted `stack_batches` function.

- `stack_batches` has no remaining definition or caller.
- `_stack_training_views` has no caller.
- `_stack_extra_values` is referenced only by its own recursive branch.
- The module's public surface exports only `RolloutBatch`.

Delete both helpers. This is zero behavior change and completes the earlier
dead-wrapper cleanup rather than creating a new abstraction.

## 2. Guard retired Python source paths

The family-first layout forbids restoring production routing packages under:

```text
vrl/models/ar
vrl/models/diffusion
vrl/generation/ar
vrl/generation/diffusion
```

These locations currently contain no tracked source. Some existing worktrees
retain ignored `__pycache__` directories from older commits; those caches are
local build artifacts, not repository architecture.

Add an architecture test that rejects Python source files (`*.py` and `*.pyi`)
under the four retired paths. Do not assert `Path.exists()`: that would fail on
a correct checkout merely because stale bytecode remains. Prove the guard
red/green by temporarily creating a source file under one retired path, running
the test, and removing the probe.

`tests/quality/protocols` is deliberately excluded. That path was removed with
the separate fake quality-gate lifecycle and is not part of the policy-axis
source taxonomy.

## Audited items that stay unchanged

- Family token-config adapters retain real per-family sampling allowlists.
- Denoise builders remain a cohesive public dispatch boundary.
- VAE decode-memory functions remain a parse/configure/policy pipeline with
  production callers.
- Attention cache-row functions remain one cohesive stateless operation group.
- Family runtime helpers that name real model-loading concepts remain in place.
- Vendored upstream code remains provenance-preserving even when the local tree
  has no caller.
- `FAMILY_REGISTRY` remains the deliberately isolated family wiring table.

These are non-actions already consistent with the executed
`SPRINT_single_caller_inlines` and `SPRINT_dead_code_wrapper_sweep` decisions.

## Non-goals

- Do not delete ignored caches as a committed deliverable. Local cache cleanup
  is a one-shot workspace operation.
- Do not sweep historical output or third-party directories.
- Do not reopen previously executed trainer, FSDP, trajectory, config, or
  resource-boundary decisions.
- Do not flatten thin family/runtime seams that preserve protocol or
  cross-family consistency.

## Verification

- Run the new architecture test red/green with a temporary forbidden source
  file.
- Run the relevant rollout batch and architecture suites.
- Run Ruff only on touched Python files using the repository's required
  fix/format/check sequence.
- Confirm both orphan helper names have no matches after deletion.
