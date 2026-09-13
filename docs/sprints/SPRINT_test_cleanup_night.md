# Test cleanup sprint: every kept test names the real error it prevents

Status: plan (2026-09-13). The `/goal` text points here; this file carries the
full brief so the goal itself stays short.

## Objective

Clean up the whole test suite of `~/Desktop/VRL` so that every retained test
answers "what real error does this prevent", and replace wiring tests built
from stacked fakes with tests that run real code wherever a real path is
affordable on CPU. Verify before changing; commit per batch; never push.

## Hard constraints

- Work only in `~/Desktop/VRL`. No worktrees, no backup branches, no stash.
  One commit per batch; the message says what was deleted or changed and
  which test now guards the real error. Do not push.
- `git status` must be clean at start; `git fetch` and base on `origin/main`.
  Re-check `git status` before every batch; if a change you did not make
  appears, stop and report -- never overwrite it.
- Environment: `uv sync --frozen --group test --group lint --extra cosmos
  --extra reward --extra reward-service --extra data --extra detection
  --extra ocr`; confirm with `uv sync ... --dry-run` that nothing drifts.
  (Upstream `78d2edc6` retired the editable `third_party` wrapper; vendored
  code is reached through each family's vendor loader and the checked-out
  submodules.) Never `pip install` into `.venv` by hand. `ar-vllm` conflicts with `cosmos`; the vLLM real-ops
  test is out of scope here.
- The GPU is shared: `nvidia-smi --query-compute-apps` before any GPU test;
  never kill a process you did not start; never launch training.
- Ruff only on files you touched; no repository-wide formatting.
- Eight tests are red on upstream itself. Do not touch them and do not
  "fix" them green; record them:
  `tests/architecture/test_generation_rollout_boundaries.py::test_generation_model_imports_stay_on_public_floor`,
  `tests/generation/bindings/chunk_autoregressive_denoise/test_binding.py::test_serialized_replay_records_preserve_axes_values_and_sample_order`,
  `tests/generation/bindings/full_sequence_denoise/test_layout.py::test_unseeded_window_survives_serialized_batch_split_retry`,
  and the five in `tests/scripts/test_train_signals.py`.

## Judgement (CLAUDE.md; do not relax)

- Delete: tests that assert a mock's own return value; parametrized matrices
  whose rows all take one code path; structure tests that only assert where
  a method is defined or MRO order; inputs no real call path can produce
  (unless production explicitly rejects them).
- Keep: tests running real tensors, real backward, real gloo collectives, a
  real local Ray cluster; config-takes-effect, checkpoint restore,
  rollout/replay parity, gradient direction; rejection of real input
  boundaries or explicit contracts.
- Never change production behavior to pass a test, never split production
  for test convenience. Sole exception: a public injection point for an
  external boundary (subprocess, network) replacing a private-name patch.
- A shared builder/fixture only when 3+ test files reuse the same
  construction; constructor bypasses (`object.__new__`) only in one named
  package `_helpers.py`.
- Never hide a regression by dropping an assertion, loosening a tolerance,
  or updating an expected value. Run the touched files before and after;
  show red-then-green where a fix is claimed.

## Batches (each: analyse -> change -> run touched dirs -> commit -> ledger)

0. Inventory, no code changes. `pytest --co -q --real-cover-report` (111
   labelled doubles, 39 NO REAL COUNTERPART). AST-profile every test:
   assertion kind (numeric / raises / state / none), whether monkeypatch
   targets are `vrl.*` or external, branch count of fake classes. Record the
   baseline as a new batch section in `docs/sprints/SPRINT_test_credibility.md`.
1. Replace the weakest wiring tests with real code. Build a `tiny_test`
   model family (CPU, KB-scale weights, reusing
   `tests/models/steps/denoise/fixtures.py::build_tiny_*`) registered
   through a test-only entry in `FAMILY_REGISTRY`, so eval / DPO / lifecycle
   script tests run the real `resolve_model_build -> build_rollout ->
   generate -> restore` chain. Targets: the 17
   `monkeypatch.setattr(..., "get_model_family_entry", ...)`, 23
   `build = SimpleNamespace(...)`, 11 `entry = SimpleNamespace(...)`.
   `tests/scripts/test_online_lifecycle.py` (55 setattr, `_install_common_fakes`)
   keeps only shutdown-order and error-propagation theorems; the rest fold
   into the real-weights lane in `tests/e2e/test_real_checkpoint_rl.py`.
2. Ray. `_FakeRay` appears in 11 files. Anything whose assertion depends on
   scheduling semantics (object refs, timeouts, dead workers) moves to
   `tests/conftest.py::real_local_ray` (the slow_test lane already has 21
   real-Ray counterparts as models); fakes stay only for pure argument
   assembly.
3. The 39 doubles with no real counterpart: read each `why=`. Keep the label
   where the reason holds (live multi-node clusters, multi-GB weights, nvtx
   unobservable in-process, HTTP to external sites). Where it says the real
   path would work (`test_jrdb_import`,
   `test_setup::test_video_world_targets_rows...`; imageio is declared),
   make it real.
4. Private-name patches (`_run_command`, `_cumem_allocator`,
   `_source_head_revision`, `_embed`): give the external boundary a public
   injection point; tests stop patching underscore names.
5. Remaining same-shape tests (34 groups by normalized AST last time, 3
   merged): merge only same-theorem ones; split thousand-line files that mix
   subjects by theorem, as `test_checkpointing.py` was split.

After each batch: `pytest <touched dirs> -q`. Every second batch: the full
suite (~4.5 min CPU). Ruff check + format --check on touched files only. The
full suite may show only the eight upstream reds.

## Finish and report

Add a section to `SPRINT_test_credibility.md`: what was deleted (one line
each: which test now guards it), what was simplified, what was kept and why,
what remains unverified (separate "not run" from "cannot run here"), and the
upstream-red list. Do not report coverage or test counts; do not claim "all
verified" unless the full suite ran. When two approaches both work, pick one.
Stop and write it down -- do not guess -- only for credentials, access you
lack, or a decision that changes user-visible behavior.
