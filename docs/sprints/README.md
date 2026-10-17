# Sprint docs layout

```text
docs/sprints/planned/   decided to do, not started or in flight; each file is an
                        executable current-state plan
```

Only plans live on `main`. Finished sprints, measurement archives, source-code
studies, run records, research notes and papers are not kept in the tree: the
last full snapshot is branch `archive/main-20261016` (`git show
archive/main-20261016:docs/sprints/done/<file>`), and references in code or docs
use that `ref:path` form.

Lifecycle invariants:

- A file in `planned/` must contain an executable current-state plan. When a
  sprint lands its final action, delete the file in the same commit; a `DONE`
  banner in `planned/` is a bug.
- A sprint blocked on an owner decision or external capability states the event
  that unblocks it at the top of the file.
- Paths and line numbers in a plan are evidence at the time of writing, not
  instructions against current HEAD. Re-read the definition, producers,
  non-test consumers, tests, and dotted-string/config references before acting.
- Measurements that matter to a current decision go into the plan that needs
  them; a measurement that matters to no plan is not committed.
