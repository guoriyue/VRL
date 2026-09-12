# FSDP framework operations

Reviewed complete trainers/fsdp.py, strategy prepare/export/restore call sites,
and existing FSDP unit/integration test coverage. Previous commit: 5f6ca9bfa.

Change: correct export documentation. Only rank0_only exports discard full copies
on non-primary ranks. Ordinary optimizer checkpoint export uses DCP CPU offload;
FP32-master export uses recursive materialization with per-rank retention. Model
checkpoint-owned gathering currently retains results on all ranks. No runtime
behavior was changed in this review.

Retain and why:

- Mesh/policy construction and apply_fsdp adapt Torch framework operations for
  the existing strategy owner. They need no additional class holding the same
  mesh/model state. The supported mesh and precision names are protocol choices.
- unwrap_module recognizes actual PEFT/compile wrappers, rather than following
  arbitrary child names. iter_blocks reads the model's declared block classes;
  no new per-family hardcoded taxonomy is necessary.
- Trainable rollout export and checkpoint-owned export select different state.
  The latter includes registered frozen mutable state. Combining their key sets
  would either omit checkpoint data or send frozen state on each rollout update.
- _full_cpu_tensor clones CPU values because detach/cpu alone can alias live Adam
  counters. _gather_named_full_cpu sorts selected names to preserve collective
  order. These helpers share real snapshot/communication rules.
- _materialize_full_cpu supports nested optimizer state and still participates
  in collectives when it does not retain results. A generic tensor-tree map lacks
  that retention contract; replacing it is not a harmless code deduplication.
- FP32 masters have different object identities from model parameters, so DCP's
  model-FQN optimizer mapping cannot serve them. The explicit positional restore
  adapter preserves the live masters' DTensor mesh and placements. Keep scalar
  step state local and full parameter-shaped state distributed.
- Parameter dtype normalization addresses actual mixed FP32 normalization versus
  low-precision model storage before fully_shard validates a group. A native
  precision policy must not silently cast. Keep those checks and the source dtype
  summary; do not remove them based solely on their number.

Non-goals: change collective ordering, checkpoint ownership, supported meshes,
dtype policy, optimizer math, or introduce another FSDP wrapper class. No ALL_CAPS
business vocabulary or unnecessary one-function module was found here.

63 existing tests in test_fsdp.py passed with CPU process groups and dependency
warnings. Coverage includes real sharding/forward/backward, exports/restores,
registered frozen state, wrapper discovery and preparation policy. Ruff checks
passed. No new tests or hypothetical input gates were added. Multi-rank CUDA,
Wan checkpoints and FP32-master distributed round trips were not rerun in this
documentation-only turn; CPU results do not prove those paths.

Limit: CPU model-state snapshots are retained on every rank by
gather_checkpoint_state_dict. Reducing that memory requires accounting for all
strategy consumers; the corrected module description no longer implies that
optimization already exists. Restore uses Torch's private _init_optim_state for
ordinary optimizers, an existing framework-version coupling kept explicit.
