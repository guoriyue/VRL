# Launch configuration ownership

Reviewed complete `generation/launch_contract.py`, `generation/ray/launch_inputs.py`,
`generation/ray/config.py` and `rewards/launch_contract.py`. Followed generation
payload projection in `run.py`, driver validation before launch in the online
entrypoint, and reward contract consumption in the runtime and HTTP service.

## Change

Remove the repeated positive engine-count condition from the colocated replay
memory guard. RayGenerationConfig construction already rejects fewer than one
engine, and its resolved resource snapshot is frozen. The guard now asks only
whether trainer and rollout share devices and the bundle loads full generation
modules. Retain the constructor rejection. No new validation helper or tests
are needed for removing this internal repetition.

This preserves behavior for constructed valid configurations. RayGenerationConfig
itself is mutable; replacing its resources afterward with an invalid zero-engine
snapshot no longer bypasses the colocated memory warning/error. Such replacement
does not rerun constructor validation and is not the supported configuration path.

## Retain and why

- GenerationRuntimeLaunchContract rejects live objects in configuration and
  validates serialization. Its recursive classmethod shares key/path diagnostics
  across four mappings; the normalization helper also copies the outer mapping.
  These methods belong to this contract, not a new universal validator class.
- RayGenerationLaunchInputs adds the registry-owned gatherer and optional rank
  spec. Its pickle check validates that larger object graph: a gatherer can own
  an unpickleable lock even when the primitive launch contract is valid. Keep the
  thin module as the shared launch/actor boundary. Do not merge it into an actor
  module and introduce eager Ray dependencies in composition code.
- Neither frozen dataclass deep-freezes nested dictionaries. Construction checks
  establish validity at construction, not after arbitrary later mutation. Do not
  cite frozen=True as proof that every payload leaf can be trusted forever.
- RewardRuntimeLaunchContract intentionally owns only runtime keys; component
  plugin options remain an open mapping. Generation's primitive-only traversal
  is therefore not interchangeable with reward parsing. Sharing a base class
  because both names end in LaunchContract would obscure their different input
  contracts. Keep the reward factory's existing conversion behavior in this pass.
- RolloutWorkerConfig.from_public_section uses public schema defaults, then
  freezes the projection. Direct dataclass construction also exists, so its
  numeric guards remain useful. RayGenerationConfig.from_root copies profiler
  settings before changing output_dir rather than mutating the root's profiler.
- Driver device checks observe the loaded model and training roots, not just
  declared resource settings. They can catch actual placement that differs from
  config. `_get_device` preserves errors from declared properties; the recursive
  parameter walk handles mappings/iterables with cycle protection. Keep these
  related helpers together under the driver configuration owner.
- `_cuda_device_index` interprets unindexed CUDA through the current device;
  defaulting it to ordinal zero would be incorrect. Cross-node checks deliberately
  avoid comparing unrelated local ordinal spaces and defer to launcher placement.
- VRL_STRICT_REPLAY_MEMORY_GUARD is an environment interface; device strings
  are device vocabulary. Export lists are API declarations.
  No misplaced ALL_CAPS algorithm lists or prompt templates occur in these files.

Non-goals: combine generation and reward schemas, freeze arbitrary plugin bags,
remove checks on actual driver state, generalize device discovery into another
utility namespace, or change deployment topology/precision/offload behavior.

## Validation

- Runtime configuration and host-memory guards: 59 passed, 1 GPU case skipped;
  upstream Ray/SWIG warnings. Includes real CPU Ray launch/probe paths, public
  projection, explicit overlap, property failures and current-device behavior.
- Reward runtime factory: 19 passed, including plugin factory flow and runtime
  parking-key rejection. No production reward model or GPU inference was loaded.
- Launch pickle coverage: 28 passed, including every registry gatherer round-trip
  and rejection of a gatherer containing an unpickleable lock; 16 other input
  composition tests deselected for this focused check.
- Ruff check and format check passed for the changed config module.

Previous isolated audit commit: `e54285d00`.
