# Rank launch and distributed identity boundaries

Reviewed complete scripts/common/launch_environment.py and trainers/distributed.py,
train.py's mask-to-config override, online reward-placement comparison, strategy
process-group/coordination consumers, and existing entrypoint/context tests.
Previous audit commit: 7eb44e1a9.

## Changes and retained boundaries

Correct documentation rather than change device semantics without a complete
mapping contract. Resource ordinal conversion assumes launcher alignment when
counts match; matching counts does not prove alignment. The distributed context
validates rank/world identity, while Torch's rendezvous owns MASTER_ADDR and
MASTER_PORT validation. The old documentation overstated both guarantees.

- narrow_rank_local_cuda_visibility must run before importing training runtimes.
  Its separate torch-free module is an actual import/lifecycle boundary. The
  function validates the mask, selects this rank's physical ID, then mutates the
  environment once. An object with a single method would add no owner value.
- train.main feeds that physical ID back into visible_devices via configuration
  loading. On this supported narrowed path torch uses cuda:0 and Ray reports
  the physical ID. Existing tests cover masks such as 2,4,6,7 and one rank per
  node. Keep LOCAL_RANK distinct from the process-local CUDA index.
- DistributedTrainingContext is a description, not the process-group owner.
  from_root accepts an explicit environment for framework/test integration.
  Its short primary/distributed properties and shared _require_env_int method
  serve actual consumers. Environment validation is an external boundary.
- _TORCHRUN_ENV_KEYS is the real env protocol and belongs here. The module's
  _CPU_COORDINATION_GROUP is mutable process-global state aligned with Torch's
  default process-group lifecycle, not an ALL_CAPS business taxonomy. Turning
  it into a namespace class would not remove that lifecycle constraint.
- init/shutdown and cpu_coordination_group are shared DDP/FSDP framework
  boundaries. Gloo coordination avoids GPU kernels during memory parking.
  Keep the transport distinction; do not merge coordination into arbitrary
  model or snapshot classes merely because these functions are short.
- run_on_primary_rank broadcasts the primary operation outcome so peers cannot
  blindly continue after rank-0 IO failure. This is distributed control flow,
  not a redundant write wrapper. No new wrapper hierarchy is introduced.

Non-goals: choosing a different collective backend, removing rank checks,
changing supported GPU masks, or renaming public process-group helpers without
benefit. This pass changes comments/docstrings only, not runtime behavior.

## Remaining ownership questions

The device mapping issue in resource_resolution.md remains open for generic
callers. The normal launch path supplies aligned mask/plan state, but the resource
object itself does not record whether its ordinals are physical or logical.
A complete solution must make that origin explicit across auto discovery,
pre-existing masks and rank-local overrides; checking another count is not enough.

init_training_process_group returns early if a default group already exists.
For an externally initialized NCCL group, no Gloo coordination group is created
here. The strategy's _distributed_all_ranks_succeeded then falls back to NCCL,
and _cpu_coordination_barrier becomes a no-op. That conflicts with the parking
path's stated CPU-only coordination requirement if such a group is adopted.
Also, generic strategy shutdown destroys the default group without recording
whether this module created it. Resolve creation/adoption/teardown together in
the strategy audit; do not add a lone flag that fixes only one side of ownership.

## Validation

27 distributed-context and online-entrypoint tests passed, including rank/world
validation, non-contiguous masks, no mutation for rejected masks, and physical
ID propagation into resource resolution. This does not validate actual NCCL
parking or externally initialized group adoption. Ruff check and format check
passed on the two Python files with documentation changes.
