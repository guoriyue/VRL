# Automated clarity audit

Status: active; repository review is not complete.

## Isolation and recovery

- Baseline: `b1e219dd4e182dcc4465026641abd02651d673e0`.
- Worktree: `/home/mingfeiguo/Desktop/vrl2/VRL-auto-audit`.
- Branch: `codex/automated-clarity-audit`.
- The original `VRL` worktree belongs to concurrent manual review. Never edit,
  reset, switch, merge into, or push its branch as part of this audit.
- Commit small reviewable groups here. Integration is separate work.
- Goal metadata is not the recovery record. On a new thread, read this file,
  inspect this worktree's status and log, and continue pending coverage.

## Scope and completion

`coverage.tsv` lists every tracked baseline production Python module, with its
Git blob ID. Inventory is not review. Pending entries require source inspection,
owner/caller evidence, and relevant config/test inspection before a disposition.
Related tests and configuration are reviewed alongside production modules.
Historical findings in `../repository_clarity_audit.md` are context, not proof
that a current module has been reviewed.

Finish only when baseline modules have an explicit reviewed or justified deferred
disposition, actionable cleanups are implemented and verified, and remaining
limitations and commits are documented. Token consumption is not a completion
criterion. No token budget was requested.

## Decisions

Change redundant internal checks, speculative loading fallback, unclear names,
misplaced ownership, and genuinely duplicated mechanisms when callers establish
that the change is safe. Prefer existing owners over additional wrapper classes.

Retain external/protocol validation, optional dependency boundaries, framework
adapters, numerical helpers and useful cross-family consistency. Thin functions
are justified by these responsibilities, not by their line counts. Retain real
schema keys, environment names, file names, architecture dimensions and isolated
taxonomy tables; inspect workflow-local business vocabularies for better owners.

Non-goals: changing training mathematics, removing supported backends, blanket
class conversion, signature uniformity for its own sake, or deleting behavioral
tests merely because a guard looks defensive.

## First inspection: reference forward lifetime

Read `vrl/rollouts/evaluators/token/ref_pass.py`, its three evaluator call sites,
and `tests/rollouts/replay/test_ref_pass.py`. The helper owns a real shared lifetime:
no gradients and temporary adapter disabling, restored even on failure. Keep that
shared boundary. A context manager could make the three callback/lambda call sites
more direct without creating a class. This is a candidate, not an implemented
change; inspect full evaluator coverage before changing it. Other portions of the
three evaluator modules remain pending. No tests have run in this audit yet.
