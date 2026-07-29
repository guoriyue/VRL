# Sprint docs layout

```text
docs/sprints/      active — in-flight work or living logbooks (appended often)
  planned/         decided to do, not started yet (waiting on TIME)
  parked/          deliberately deferred (waiting on an EVENT; each doc states
                   its trigger — ask "what is it waiting for?" to file)
  done/            completed, kept as records (delete when no longer referenced)
  reading/         source-code studies and direction discussions, not action items
  info/            profiling / measurement result archives — kept as reference and
                   revisited often, not action items (e.g. perf + cross-model profile)
```

`done/` vs `info/`: `done/` is *finished work*; `info/` is *measurement data* that
outlives the work that produced it. `reading/` and `info/` are categorized by KIND,
the other three by STATUS.

File new sprints at the level matching their status; move them as status changes
(`git mv` keeps history).

Lifecycle invariants:

- A file in `planned/` must contain an executable current-state plan. A `DONE`,
  `completed`, or “all items landed” execution banner means the file belongs in
  `done/`, not `planned/`.
- Move a completed sprint in the same commit that lands its final action. Do not
  leave cleanup bookkeeping for a later sweep.
- When a sprint is partially complete, split the remaining current-state action
  into a short `planned/` file and archive the pre-execution audit in `done/`.
- A sprint blocked on an owner decision or external capability belongs in
  `parked/` and must name the event that moves it back to `planned/`.
- Paths and line numbers in `done/` are historical evidence, not instructions
  against current HEAD. Every new cleanup sprint must re-read the definition,
  producers, non-test consumers, tests, and dotted-string/config references.
