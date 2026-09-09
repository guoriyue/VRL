# SPRINT: repo hygiene sweep

Executing the outstanding items of `planned/SPRINT_full_repo_bloat_audit.md` §8.2,
one reviewable commit each, under `GOAL_repo_hygiene_sweep.md`.

`kept` is a finished outcome: the item was examined and deliberately left alone,
and the reason is the deliverable. `blocked` means it needs something this run
must not do (a GPU numerical gate, a public config-key migration).

## Ledger

| # | item | source | verdict | evidence | commit |
|---|---|---|---|---|---|
| 1 | three lazy-export facades -> shared installer | §8.2 | changed | All 46 table entries had the value duplicate its own key, and the `__getattr__`/`__dir__` bodies were identical apart from a docstring. Public surface verified unchanged before/after: `__all__`, the `AttributeError` text, and torch-free import all identical; `dir()` differs only by module-private names that were never in `__all__`. | (this commit) |
| 2 | `vrl/algorithms/grpo/__init__.py` as a fourth facade | §8.2 neighbour | kept | Not the same machinery: it deliberately exports nothing (`__all__ = []`, 10 lines) because `vrl.config.algorithm` imports each objective by submodule. Nothing to share. | — |

## Notes for the reviewer

- The whole-repo audit's conclusion stands: there is no structural rot. These
  items are the ones its execution batches deliberately deferred, each with a
  recorded reason.
- One test failure is pre-existing on this machine and unrelated:
  `tests/nn/kernels/test_vllm_paged_attention_real_ops.py` fails by upstream
  design here.
