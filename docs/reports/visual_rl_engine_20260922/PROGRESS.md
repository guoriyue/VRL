# Visual RL engine: implementation and evidence ledger

## Goal and time boundary

- Started: 2026-09-22 07:36:41 UTC.
- User requested at least 24 hours of work: do not claim this duration satisfied before
  2026-09-23 07:36:41 UTC. Elapsed time alone does not establish completion.
- Target: independent reward reliability tooling, rollout admission, real Qwen-Image-2.1
  editing, and bounded agentic visual RL with explicit credit assignment.
- Distinguish CPU/fake tests, real inference, actual policy training, and independently
  measured capability improvement. No deployment, push, paid API/cloud allocation, or
  large model download is authorized.

## Initial authoritative state

- Workspace: `/home/mingfeiguo/Desktop/vrl2/VRL`.
- Existing modifications: reward and admission sprint documents; untracked NGU sprint
  and `third_party/PhyMotion/`. Preserve these; do not sweep into implementation commits.
- Local GPU: RTX 5090, 32 GB. At startup another process owns approximately 16.8 GB and
  is actively evaluating a Qwen-Image-2.1 checkpoint. Do not interrupt it.
- Observed process: PID 4175992, working directory `/home/mingfeiguo/Desktop/VRL`,
  command `vrl.scripts.eval.reference_image_checkpoint_eval`, checkpoint
  `outputs/qwen_image_21_edit_rl/main/checkpoint-final`, label `rl20`.
- That checkout has `qwen_image_21` family/editing documentation and commit
  `90852130` (reference-conditioned EditReward training). Treat it as a read-only
  integration reference; inspect dependencies and actual diffs before porting.
- Model config there names `Qwen/Qwen-Image-2.1`; resolve the cached artifact path before
  any model launch. Existing documentation does not prove this workspace supports it.

## Work completed / underway

1. Synchronized current workspace environment from its frozen lock with test/lint,
   cosmos, reward and reward-service extras using `--inexact` to retain existing
   optional installations. Did not modify the active external Qwen environment.
2. Implemented propagation of nested reward observations using `parent/axis` keys,
   preserving weighted training totals and zero-weight audit components. Reject
   namespace collisions instead of overwriting measurements. Ruff checks passed;
   `tests/rewards/functions/test_multi.py`: 42 passed (CPU, 0.37 seconds).

## Next implementation steps

1. Finish component propagation checks and make an isolated code commit.
2. Implement independent media-manifest scoring using existing reward runtime/service;
   persist raw axes and immutable scoring provenance before report/calibration layers.
3. Reuse existing score-report statistics; validate cached rescoring and comparisons.
4. Inspect Qwen family integration and dependency compatibility, then port bounded
   reference editing support without copying the other checkout's unrelated changes.
5. Add bounded episode/tool execution and training ownership/credit assignment with
   explicit controller/generator semantics; validate with real model when available.
6. Continue rollout admission work after preserving distributed advantage semantics.

## Evidence not yet obtained

- No real model inference or RL has run in this workspace for this goal.
- No preference labels or capability improvement have been manufactured or claimed.
- No 24-hour completion, production readiness, or working agent policy RL is established.
