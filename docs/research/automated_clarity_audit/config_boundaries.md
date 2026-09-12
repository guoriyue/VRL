# Configuration boundaries review

Reviewed modules: `vrl/config/{__init__,algorithm,base,loading,rules,validation}.py`
and `vrl/algorithms/config_contract.py`. The full `schema.py`, builders, precision
and algorithm implementations remain pending; selected ownership/call-site
sections were inspected here without treating that as whole-file coverage.

## Change

Remove the second single-key check on a defaults dictionary in `_load_one`.
Every entry reaches `_apply_default_override` first; that helper either rejects
a dictionary with other than one key, returns the validated dictionary, or
replaces it with a string. The downstream check therefore repeated an invariant
on an unchanged internal value. Invalid input still fails at the existing entry
boundary with the same message. No loader grammar or overlay-order changes.

## Retained responsibilities

- `config.__init__.build_configs`: keep the lazy public facade. Resource discovery
  must not import the runtime builders. The subprocess resource test exercises
  exactly this import boundary.
- `algorithm_config_class`: keep explicit lazy dispatch. The branches select
  classes; they no longer duplicate per-algorithm SDE/KL/SFT facts. Replacing the
  imports with a string registry would introduce another import-path vocabulary
  without removing the algorithm-kind/schema boundary.
- `AlgorithmConfigContract`: keep algorithm-owned immutable declarations.
  `config_contract` is a ClassVar, excluded from serialized hyperparameters and
  rejected as a YAML override. Tests change declared facts without changing kind
  and demonstrate that rules follow those declarations.
- `rules.py`: the original five hardcoded lists are absent. Keep cross-section
  checks and the explicit Janus-R1/NextStep pairing checks. No reintroduction of
  the previously rejected `FamilyTrainingContract` abstraction. Remaining numeric
  SFT validation ownership is tracked below, not hidden by the review status.
- `ConfigBase.revalidate` and `_extract_error_message`: keep the Pydantic adapter
  and pure error formatting. The nested `location` function shares section
  anchoring; `_UNKNOWN_KEY_ERRORS` names Pydantic error protocol codes. Tests
  exercise nested unknown keys and section-qualified errors.
- `loading.py`: keep stateless YAML/resource composition functions. `Traversable`
  resources need not be filesystem paths, so they cannot be replaced wholesale
  with `RootedPaths`. `_BUNDLED_CONFIGS` is the package-resource boundary and
  `_SELF_` the YAML composition token. Recursive visiting/loading, resource
  joining and override selection each implement part of that grammar.
- `validation.py`: keep ordered `TRAINING_GATES` as the launch protocol, including
  uniform signatures. `gate_compile_compatible` now aggregates conflicts itself;
  it is not the old forwarding-only wrapper. `compile_conflicts` is shared with
  runtime checkpointing, which selects its own feature. Drift and production
  checks need precision/runtime/filesystem facts; do not move them into pure
  parsing or introduce one class per gate.

## Evidence and unresolved ownership

Read the algorithm selection and root validator in `schema.py`; declarations in
GRPO, Flash-GRPO, Flow-DPPO, GRPO-Guard, Token-GRPO, V-GRPO, DiffusionNFT and DPO;
compile-conflict consumption in `trainers/activation_checkpointing.py`; and
flow-matching, continuous-token and multisegment recipe composition. These were
read alongside algorithm contract, loading, validation-tier, unknown-key and
drift/compile tests.

`rules.py` still checks that `sft_weight` is finite and nonnegative, which is a
single-field property rather than a cross-section relationship. GRPO and DPO
configs currently declare the field without their own validation. Moving that
check must account for direct dataclass construction, GRPO subclass inheritance,
and the prebuilt-hyperparameters path. Keep it until the algorithm-owner review
can move it coherently; dropping it here would remove the only numeric guard.
The requirement for a latents dataset remains a cross-section rule regardless.

No claims that compile restrictions are universal facts about every current
torch backend/version: this review retains the repository's compatibility policy.
No training mathematics or supported family pairing changed.

## Validation

70 tests passed across loading composition/resources, algorithm contracts,
validation tiers, rollout drift/compile compatibility and unknown keys. Includes
the bundled experiment structural parse test; this is not a full training-launch
or GPU compilation check. Ruff passed on the touched loader.

Previous implementation commit: `9d3b55df3` (shared profiler manifest writer).
