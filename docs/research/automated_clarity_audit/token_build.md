# Token build projection and entry boundaries

Reviewed complete steps/token/build.py, all five family config projections,
registry descriptor dispatch, scoped replay constructors, the run-level compile
capability gate, LoRA projection tests and minimal-bundle wiring tests.
Previous audit commit: 00c532e33. Family model/runtime modules remain pending their
full review; reading the constructor/projection does not cover their generation
or replay math.

## Disposition: retain implementation

No demonstrated in-scope cleanup is needed in this shared builder. In particular:

- token_model_config_base supplies identity and explicit LoRA overrides to all
  family config dataclasses. It deliberately does not use ModelBuild.lora, which
  requires complete rank/alpha/targets; token family defaults support a partial
  override such as rank alone. Replacing this projection with the denoise view
  would change defaults or raise on valid token configurations.
- _validate_token_lora_path protects two distinct public entry paths: direct
  config projection and bundle construction before importing a family. Existing
  tests check both rejection and that import_from_path was never invoked. Its
  small size is not grounds to duplicate or remove this shared rule.
- Presence checks for init/init_lora_weights preserve False as a valid value and
  reject simultaneous aliases. Replacing them with truthy fallback would silently
  alter initialization. Disabled LoRA ignores other nested overrides but refuses
  a warm-start path that could otherwise be silently ignored.
- The shared builder selects the replay or generation constructor from the
  descriptor, validates the build role, then lets the model constructor attach
  LoRA. Applying the denoise builder's separate apply_lora sequence would attach
  adapters twice. Full-parameter training support is an explicit descriptor fact;
  frozen replay constructors for Janus/Emu3/GLM/LlamaGen do not justify removing it.
- Import paths are lazy dependency boundaries. ModelFamilyEntry's type-only
  import avoids a registry cycle. No stateful builder class would add ownership
  beyond the existing ModelBuild/config/model objects.
- __all__ is a public API list. This module has no ALL_CAPS business table to
  extract. Family field-name loops are bounded config projections owned by those
  families, not an algorithm-name taxonomy hidden in shared workflow code.

Non-goals: forcing token and denoise construction into identical steps, adding
new per-field integer checks, changing accepted casts, or wrapping each free
function in a static-method namespace.

## Evidence and limits

52 LoRA config and minimal replay wiring tests passed with CUDA hidden. Coverage
includes partial overrides, initializer aliases including False, early path
rejection, nested precision reconstruction and registry rollout/replay assembly.
No Python source changed, so no new tests or formatting churn were introduced.
Wiring tests substitute model checkpoint loading; they do not establish pretrained
weights, CUDA behavior or whole-family numerical parity.

Replay constructors inspected here exclude generation-only decoders: Janus and
Emu3 reject retained vision/VQ modules, GLM excludes its decode pipeline, NextStep
loads the replay core, and LlamaGen omits VQ while retaining T5 needed for replay.
The latter is necessary replay state, not proof that replay memory is small.
The shared loads_full_generation_modules=False flag is not a byte-level memory
measurement and remains subject to actual loader review per family.

Token replay does not apply compile; token entries currently declare compile
unsupported and run.py rejects raw enable=True for either scope. Do not infer a
missing supported replay optimization from the denoise builder's compile step.
Direct registry/build calls bypassing run.py do not have that capability gate:
replay can ignore a requested compile, while rollout reaches the optimization
pass without a token compile method. Defer unifying that direct-API behavior to
the run/registry entry-boundary review; do not enable unsupported compilation or
add another per-family support list here.
