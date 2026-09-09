# GOAL: repo hygiene sweep — execute the outstanding audit items, one reviewable commit each

Work in `~/Desktop/VRL` on branch `main`. Autonomous run, expect ~10 hours. The
operator reviews your work commit by commit afterwards, so **the commit history is
the deliverable**, not the diff size.

## Scope

This repository has already been audited end to end. Do **not** start a fresh
whole-repo sweep — 175 completed sprints under `docs/sprints/done/` include eight
dead-code passes, an ALL_CAPS audit, a function-organization audit, a design-smell
audit and a duplicate-consolidation pass, each with a KEEP list recording what was
examined and deliberately left alone.

Your queue is, in this order:

1. **`docs/sprints/planned/SPRINT_full_repo_bloat_audit.md` §8.2** — the remaining
   items of the full-repo audit, each already carrying a path, evidence, and the
   reason it was deferred. This is the bulk of the work.
2. **§1 A1** — the one undecided item (wan's `autocast_adapter_dtype=False` opt-out).
   Decide it statically if you can; if it needs a numerical run, mark it blocked.
3. **A fresh pass on the two packages that audit rated worst**: `vrl/trainers/`
   (heaviest horizontal duplication) and `vrl/scripts/` (most dead knobs). Only
   after the queue above is empty.

## Read before touching anything (do not skip)

- `AGENTS.md` — the binding rules: dead-code five forms, placement four rules,
  the thin-function and ALL_CAPS keep-lists, formatting and diff discipline.
- `docs/sprints/planned/SPRINT_full_repo_bloat_audit.md` §5 (KEEP summary — already
  examined, do not re-audit), §7 (non-goals), §8.1 (what already landed and the
  three verdicts that were *corrected* during execution).
- `docs/sprints/done/SPRINT_resolver_function_placement.md` — the most recent sweep
  of exactly this kind, including its naming rules and the list of functions
  deliberately kept free with reasons.

## Ledger — create it first, update it before every commit

Create `docs/sprints/SPRINT_repo_hygiene_sweep.md` with one row per item:

| item | source | verdict | evidence | commit |
|---|---|---|---|---|

`verdict` is one of `changed`, `kept`, `blocked`. **`kept` is a first-class
outcome** — an item you examined and decided not to touch is finished work, and the
reason is the deliverable. `blocked` means it needs something you must not do
(a GPU numerical gate, a public config key migration); say which.

The ledger is what the operator reads to decide where to look. Keep it current;
never batch up ledger updates at the end.

## Work loop, per item

1. **Read the function body, its call sites, and its tests.** Never judge from the
   caller, the name, or the file it lives in. Two wrong verdicts in the last sweep
   both came from judging by the caller.
2. **Decide change or keep**, and write the one-sentence reason either way.
3. If changing, make the smallest change that removes the root cause. No
   opportunistic cleanup of nearby code.
4. `ruff check --fix <touched files>` then `ruff format <touched files>`, then
   `ruff check` and `ruff format --check` on the same files. **Never repo-wide.**
5. Run the affected test packages, then the full suite (`pytest -q`, ~3 min).
6. **Commit.** One concern per commit. The message states what changed and the
   evidence that justified it (a path:line, a grep result, a measurement). No
   `Co-Authored-By` or generated-with trailers.
7. Update the ledger row with the commit sha.

## Commit discipline — this is what makes the run reviewable

- **One concern per commit.** A single reason touching six files is one commit; two
  unrelated fixes in one file are two commits.
- **Every commit must leave the suite green.** If you split an accumulated working
  diff into commits after the fact, check out each intermediate in a detached
  worktree and run the affected tests *there*. A hunk-level split that drops a
  `def` line produces a commit that does not even import, and the tip still looks
  fine.
- **Never mix formatting-only changes with functional ones.**
- Do not amend, squash or rebase. The operator wants the sequence you actually took.

## Hard boundaries

- **Do not push.**
- **Do not run GPU training or evaluation**, and never kill a GPU process — the
  operator runs their own jobs on the same card.
- **Do not delete or rewrite sprint documents.** They are measurement archives.
  Append a dated correction instead.
- **Do not change public config keys, model presets, or dataset files.** A key
  rename is a user-visible migration and is out of scope.
- **Do not "simplify" the things AGENTS.md protects**: registry dotted-string
  dispatch, protocol boundaries, lazy-import facades, framework adapters, test
  fakes, and per-family uniform shapes whose value is grepability. When a thin
  function or an ALL_CAPS constant is one of these, record it as `kept`.
- **Do not extract a helper that would have a single caller.** Inline it instead.
- **Do not create a new module for one class.** Sink shared code into an existing
  file that already owns the concept.

## Traps that have already cost time in this repository

- **Moving a free function onto a type breaks callers that pass structural stubs.**
  An annotation is a hint; a method is a runtime contract. One such move broke 146
  tests because the call sites pass `SimpleNamespace` and ad-hoc fakes. Check every
  call site, tests included, before moving anything onto a class.
- `tests/config/test_load_all_experiments.py::test_config_parsing_stays_torch_free`
  forbids torch on the config path. Cross-section validators must stay free
  functions in `vrl/config/validation.py`.
- After renaming or deleting a symbol, **grep for it as a string too**. Registries,
  launch contracts and e2e harnesses dispatch by dotted path
  (`"module:function"`), and a plain-symbol grep misses them.
- A test that monkeypatches a module attribute breaks when that function becomes a
  classmethod. Repoint the patch target, and watch for bare-string targets that a
  symbol rename will not reach.
- Deleting a name from a package requires updating its lazy-export table
  (`vrl/trainers/data/__init__.py` and siblings), or imports fail at runtime only.
- A test that only exercises a fake it also defines is mock self-certification, not
  coverage. Either give the fake the real object's method surface or delete the
  test — do not leave it as evidence.
- `vrl/scripts/lint/dead_flags.py` currently reports **350 CLI flags, all consumed**.
  That gate must stay at zero; if a change orphans a flag, delete the flag too.
- One `pytest` failure is expected on this machine and is not yours:
  the vLLM paged-attention test fails by upstream design here.

## Stop conditions

Stop, write the ledger, and leave the rest for the operator if:

- the full suite goes red and two attempts do not fix it (leave the tree clean —
  reset the offending change rather than committing it red);
- an item needs a numerical regression on GPU;
- an item would require changing a public config key;
- you find a correctness bug that is bigger than the cleanup it was hiding in —
  record it, fix it in its own commit ahead of the cleanup, and say so.

## Pace

Aim for one commit every 20–40 minutes. Prefer finishing one item completely
(code, tests, ledger) to starting three. At the end, the ledger must account for
every item you looked at — changed, kept, or blocked — and the working tree must
be clean.
