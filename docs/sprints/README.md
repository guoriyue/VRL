# Sprint docs layout

```text
docs/sprints/      active — in-flight work or living logbooks (appended often)
  planned/         decided to do, not started yet (waiting on TIME)
  parked/          deliberately deferred (waiting on an EVENT; each doc states
                   its trigger — ask "what is it waiting for?" to file)
  done/            completed, kept as records (delete when no longer referenced)
  reading/         source-code studies and direction discussions, not action items
```

File new sprints at the level matching their status; move them as status changes
(`git mv` keeps history).
