# Rollout optimization pass ownership

Reviewed complete nn/optimization/passes.py and its public package facade,
quantization loader contract, denoise/token build callers and pass/order/core
coverage tests. Previous audit commit: 5072d4206.

## Changes

- Correct module/function documentation: denoise device placement runs after
  quantization and before compile; offload hooks are installed within the pass
  sequence after compile. The earlier claim that device moves happen after all
  passes contradicted the implementation and could mislead a new family author.
- Remove QuantizationPass's duplicate count branch. enabled requires a selected
  quantization policy, and apply_rollout_quantization rejects zero matching
  linears before returning their positive count. The pass always validates every
  core and returns applied=True after successful application.
- Log the actual PassResult.applied flag rather than always saying "applied",
  including an offload pass with no requested/supported offload. Human-readable
  log text changes; pass ordering, enabled state and transformations do not.

## Retain and why

- OptimizationPass is a structural build-time interface. Each implementation
  has enabled/apply with the same shape, making cross-family ordering inspectable.
  Small methods here are protocol implementations, not incidental free functions
  to inline or turn into another controller hierarchy.
- ROLLOUT_PASSES is the deliberately isolated dependency-order table. Replacing
  it with per-family branches would duplicate ordering. REQUEST_SCOPED_DRIFT_SOURCES
  is the separate request-time drift taxonomy; TeaCache does not mutate a module
  tree and cannot be made a build pass merely for uniform appearance.
- validate_every_core_quantized checks actual quantized modules in every declared
  root; a positive global match count does not prove both experts were covered.
  CompilePass similarly checks each root's compilation effect. These boundaries
  catch partial application and must not disappear with the redundant count test.
- OffloadPass.enabled remains true even for mode none because the family installer
  checks construction-mode consistency. Its apply handles unsupported families
  and the reported hook health before publishing a reusable model.
- before_compile is the device-placement seam and must fire once even if compile
  is disabled. VAE decode memory touches a disjoint subtree. Their ordering cannot
  be inferred from a generic enabled-pass count.
- PassResult carries applied/detail for diagnostics; the package facade preserves
  supported imports. Existing constants are protocol ordering/taxonomy, not
  algorithm-name facts misplaced in workflow code.

Non-goals: changing quantization targets, kernel recipes, compile modes, offload
behavior or drift math; replacing protocol classes with ad-hoc functions; or
adding a second conflict matrix after config validation.

## Validation and limits

50 pass, family-core declaration and denoise LoRA/FP8 build tests passed. Existing
tests cover zero matches, partial expert coverage, quantization before compile,
the device seam with compile disabled and request-scoped drift. Ruff passed for
the changed module. Tests use CPU module transformations and compile wrapping;
they do not establish CUDA kernel performance or end-to-end numerical parity.

The quantization coverage check establishes at least one matching quantized
module per policy core, not exhaustive quantization of every eligible linear.
The compile check uses the existing _orig_mod marker. No stronger validation or
arbitrary custom-wrapper support is claimed by this cleanup.
