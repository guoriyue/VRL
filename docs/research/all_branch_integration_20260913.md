# Experiment branch integration on main

## Scope

Review branch: `review/all-mgpu-main-6b723075`.
Exact upstream base: `6b723075e994255539b7604e249cf7efdb1f5547`.

Sources preserved and replayed in dependency order:

- `feat/multi-gpu-acceptance` through `0f4b6793`, including the latest local execution-ledger snapshot.
- `feat/cosmos-cp-runtime` through `9145b2af`: 51 additional commits.
- `feat/h3-four-l40s` through `17b6f460`: 11 additional unique commits.

The previous `review/multi-gpu-main-6b723075` branch is unchanged. Original
worktrees and the dirty `third_party/videophy` submodule are not modified.
No independent old spike/integration branch is added to the review scope.

## Main API adaptations

- Preserve upstream `ModelParking` and `TrainingStateParking` ownership while
  retaining mixed-device restoration and local FSDP shard parking.
- Preserve `TrainingCollectives` and transactional checkpoint saving while
  adding rank-local RNG capture and strict named-generator admission.
- Migrate online accumulation to `prompts_per_collection` and replay width to
  `training_microbatch_size`; retain update-wide global advantage normalization.
- Use current reward launch configuration and parking-pool ownership; preserve
  driver RNG during lazy reward construction and bind CUDA device scopes.
- Update explicit imports, EMA naming, runtime bundles and collector test
  interfaces without restoring removed upstream APIs.
- Accommodate the lock's FSDP parameter-group layout and contiguous-gradient
  requirement at the context-parallel differentiable gather boundary.
- Prevent H3 loader test mocks from leaking through first-time module imports.

## Validation boundary

Tests use `/mnt/nvme/venvs/vrl-review-all`, synced with the unchanged lock via
`uv sync --frozen --group test --group lint --extra cosmos --extra reward`.
Final results:

- All affected non-GPU test files: **791 passed, 14 skipped, 35 deselected**
  in 162.18 seconds. Includes real CPU multiprocess CP composition, checkpoint
  resume/EMA, and four-rank streaming advantage/gradient/Adam equivalence.
- Explicit tiny GPU gates: **12 passed, 24 deselected** in 33.01 seconds:
  `tests/trainers/test_fsdp_cuda_parking.py`,
  `tests/trainers/test_checkpoint_rng_distributed.py`, and
  `tests/models/families/minimax_h3/test_device_dispatch.py`, with `-m gpu`,
  `VRL_H3_DISPATCH_CUDA=1`, and `VRL_H3_FOUR_GPU=1`.
  Covers four-GPU parking and subsequent optimizer updates, two/four-rank RNG
  restore, and tiny random-weight H3 cross-device execution, replay and decoding.
- Expanded trainer regression before the final test-only RNG API migration:
  752 passed; its three stale barrier-call failures were fixed and covered by
  the final CPU/GPU runs above. This is not reported as a separate all-green run.
- Scoped Ruff check/format and `git diff --check` passed.

GPU markers were added to the migrated hardware tests so ordinary CPU runs do
not inadvertently launch CUDA workers. H3 hardware evidence uses random tiny
weights only and does not establish released-model quality or capacity.

Existing GPU experiment timings and official-model outcomes in the execution
ledger predate this rebase. They are not new acceptance evidence for this branch.
No continuous experiment queue or official-weight run is restarted by integration.
