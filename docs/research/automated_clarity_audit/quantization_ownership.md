# Quantization ownership and format boundaries

Reviewed complete quantization/base.py, formats.py, targeting.py and __init__.py,
with loader dispatch/master cleanup, configuration format/recipe declarations,
scheme alignment declarations and shared/per-scheme regression coverage.
Previous audit commit: 10bcc4347. Scheme kernels remain separate pending reviews.

## Changes

Correct the runtime registry comment: registering a class does not also declare
its public config format and recipes. The torch-free config boundary owns that
declaration. Correct the shared test fixture's outdated NVFP4 alignment comment
to K % 32 and N % 16. Neither change modifies execution or numeric behavior.

## Retain and why

- QuantizedLinear owns traversal, master lifetime, cache refresh and device moves
  across both schemes. Its class methods read scheme declarations and construct
  replacements; an additional registry-driven builder class would add indirection.
- drop_quantized_masters is a useful cross-scheme operation over a model tree. It
  has no per-scheme receiver or stored state; turning it into a static helper on
  a new owner would not simplify ownership. The loader calls it only when base
  weights will not be synchronized again.
- _apply_preserving_dtype is shared packed-cache movement logic. Viewing bytes
  shields the stored FP4/FP8 representation from the model's floating dtype cast.
  It is not a speculative tensor converter and should not be inlined repeatedly.
- formats.py ALL_CAPS values are numeric-format and hardware-layout boundaries.
  E2M1 thresholds are derived from its encoding table; tests check exact grid
  reconstruction and tie rounding. These are not arbitrary workflow vocabulary.
- targeting.py deliberately isolates name-based exclusions and MLP path taxonomy.
  Profile matching and the scheme's can_replace shape gate serve different roles:
  numerical scope versus kernel eligibility. Existing attention/MLP and excluded
  module tests exercise real swaps. Replacing these rules with inferred names or
  merging them into each family would duplicate policy.
- QUANTIZATION_SCHEMES maps each implementation's declared identity to its class;
  __init__.py is the public facade. Config vocabulary remains torch-free rather
  than importing runtime classes simply to eliminate a declaration boundary.

Non-goals: alter quantization math, change production target scope, add checker
classes, flatten shared class methods, or claim a newly registered backend is
production-validated solely because it appears in the registry.

## Validation and limitations

41 quantization tests passed and 18 skipped with CUDA hidden. Existing tests cover
master state-dict ownership, cache refresh after loading, rejection of base loads
after master release, invalid profiles before mutation, real target swaps and CPU
reference numeric behavior. GPU moves, scaled matrix multiplication and compiled
production behavior were not exercised. No new test was needed for comment edits.

Open: _apply restores hidden caches if superclass movement raises, but this is
not a transaction over parameters already moved by that superclass. Subsequent
requantization or master-free cache movement can also fail after state changes;
there is no all-stage rollback. Normal supported move tests do not establish
recovery after OOM/kernel failure. Address only with an explicit recovery contract,
not a generic catch-and-continue wrapper.

Open: swap_linears mutates incrementally to avoid retaining a second whole model.
Invalid profiles fail before traversal, but a later constructor failure can leave
earlier modules swapped. Do not generalize the profile test into a guarantee that
all swap failures leave the entire model untouched. The audit preserves this
memory tradeoff and does not introduce whole-model staging.
