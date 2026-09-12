# Model build and checkpoint ownership contract

Reviewed the complete models/interfaces/runtime.py, registry build projection,
worker payload reconstruction, shared denoise builders, LoRA frozen-adapter
registration and checkpoint/FSDP ownership consumers. Previous audit commit:
d6bbe26e6. This completes the interface module, not those larger consumers.

## Change

Correct use_lora documentation: absent/null disables adapters in production too.
ModelSection defaults it to None, and registry projection uses exclude_unset=True;
the absence is not an accommodation solely for fake builds. Runtime unchanged.

## Retain and why

- ModelBuild normalizes primitive wire payloads into RolePrecision, nested
  QuantizationPolicy, RolloutBuildOptions and GenerationMemoryPolicy. The worker
  constructs it from launch_contract.model_build; these are real deserialization
  checks, not repeated assertions on an already constructed internal object.
- require_rollout/require_replay define callable builder role boundaries. Replay
  cannot inherit generation memory or deferred-device-move rollout settings.
  Full generation modules and selected precision role are separate: offline DPO
  deliberately requests full generation with training precision.
- Curated lora, num_steps and revision properties serve multiple family adapters.
  config_revision_kwargs keeps tokenizer/encoder repository revisions separate
  from the model's revision; substituting the model revision would be incorrect.
- torch_compile_for_role serves typed config guards before ModelBuild exists as
  well as its property afterward. Keep the shared function and enum vocabulary;
  do not duplicate role logic in loaders, strategy or activation checkpointing.
  TORCH_COMPILE_MODEL_KEY is a schema key, not a misplaced business table.
- register_checkpoint_owned_state declares frozen mutable state that gradients
  cannot identify. The actual producer freezes the previous adapter, registers
  its parameter names and rolls back requires_grad if registration fails.
  Removing registration would omit algorithm state from selective checkpoints.
- checkpoint_owned_state_names derives current trainable state and verifies the
  registered names still exist. Registration-time validation does not prove this
  after a module tree changes. _module_state_names is shared discovery of both
  parameter and buffer FQNs, including aliases via remove_duplicate=False.
  _CHECKPOINT_OWNED_STATE_NAMES_ATTR is an internal module protocol key.
  These functions share module-owned state across plain, DDP and FSDP consumers;
  moving them onto RuntimeBundle would not fit callers that have only a module.
- RuntimeBundle publishes roots, adapters, memory ownership and precision. Its
  post-init stamps model.precision for replay consumers. Keep this construction
  responsibility rather than creating another wrapper or generic validator.

Non-goals: freezing all build objects, changing LoRA coercion/default semantics,
adding dtype checks throughout callers, importing torch eagerly into this
interface, or flattening protocol helpers just to reduce their count.

## Validation and deferred limits

42 existing minimal-replay wiring, checkpoint ownership and frozen-adapter tests
passed with CUDA hidden. They cover nested payload normalization, compile scopes,
registry construction of minimal denoise/token bundles and registration rollback.
Checkpoint loaders are mocked in the wiring tests; this does not establish real
model loading or distributed-training parity. Ruff check/format check passed.

ModelBuild is mutable, and model_config remains a mapping of family fields.
Properties still coerce LoRA rank/alpha and local_files_only; this is not a fully
strict typed boundary for arbitrary hand-built mappings. No new checker was
added, and no broad guarantee of post-construction validity is claimed. Likewise,
RuntimeBundle stamping is not an ownership lock against reusing the same model
in another bundle with a different precision.

A CPU probe confirmed registration accepts a nonpersistent buffer because it
exists in named_buffers; export_checkpoint_state then raises that state_dict is
missing the owned name. Current production registration targets adapter parameters,
so no observed training path loses state silently. Defer early rejection of
nonpersistent/custom-state-dict entries until such a producer is supported; calling
state_dict at registration merely for validation could materialize unwanted state.
The existing export check remains necessary and identifies the missing name.
