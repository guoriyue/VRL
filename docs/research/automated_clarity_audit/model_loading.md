# Shared model loading boundaries

Reviewed the complete models/loader.py, the denoise build caller, registry
dispatch, ModelBuild's role-specific compile projection, SANA scheduler conversion,
FLUX replay preparation, loader tests and LoRA/quantized construction tests.
Previous audit commit: f961035c0. The larger runtime interface and denoise builder
remain pending their own complete caller/contract review.

## Change

Correct the loader module's ownership description: it loads components and swaps
quantized linears; family builders and optimization passes own LoRA, fine-tuning,
device placement and compile ordering. Replace the quantization guard's abstract
free-function justification with its two actual call boundaries. No runtime
behavior or public signatures change.

## Retain and why

- Component loader functions are shared Diffusers adapters and lazy dependency
  boundaries. They consistently pass revision/local-file settings; a stateful
  loader class would merely duplicate ModelBuild's state.
- load_flow_match_scheduler translates the real SANA checkpoint flow_shift field
  to FlowMatch's shift, matching the rollout conversion. This is source-schema
  translation, not inference from a model name or checkpoint filename.
- Dynamic schedulers defer timestep construction until family code knows the
  resolution-dependent mu. FLUX prepare_replay owns that operation. Eagerly
  treating every scheduler alike would break this contract.
- validate_rollout_quantization_support runs before expensive model loading in
  the denoise builder and before mutation in the directly callable swap API.
  Removing either call would leave a different entry point unprotected.
- The blockwise/compile check consumes ModelBuild.torch_compile, whose disabled
  or out-of-scope value is None. A raw {enable: False} dictionary is not the
  production property's contract; no speculative second normalization is added.
- QUANTIZATION_SCHEMES is an imported, isolated backend registry. The loader
  contains no new per-family list, recipe defaults or backend table. __all__ is
  the existing public export list. Scheme classes own swap targets and kernels;
  model policy_cores and exclusions define traversal boundaries.
- Missing quantization returns zero; selected quantization with zero matches
  raises. Dropping masters is gated by base-weight synchronization ownership and
  happens before device placement. Neither condition is redundant type checking.

Non-goals: turning adapters into classes, changing SANA/FLUX schedule math,
altering quantization profiles, deleting externally callable guards, or claiming
all denoise-builder validations have been reviewed through every custom family.

## Validation and limits

24 existing loader and LoRA/FP8/NVFP4 construction tests passed with CUDA hidden;
Ruff check and format check passed for loader.py. The tests include real tiny
Diffusers checkpoint serialization, precision preservation and real CPU module
swaps; they mock NVFP4 availability where needed. No CUDA execution, kernel speed
or whole-training parity is established. Documentation-only changes need no new
tests mirroring their wording.

The flow_shift reconstruction eagerly sets timesteps and does not repeat the
generic loader's dynamic-shifting condition. Inspected SANA and FLUX paths do not
establish a supported checkpoint combining flow_shift with dynamic shifting;
leave that combination unsupported/unproven rather than adding a speculative
fallback. If such a producer is introduced, its scheduler contract needs an
explicit resolution-aware initialization path and behavioral coverage.
