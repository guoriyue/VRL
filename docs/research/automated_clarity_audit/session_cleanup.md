# Session cleanup ownership

Reviewed complete `generation/ray/session.py` and `ray/dependencies.py`, runtime
shutdown callers, actor-group shared cleanup, placement topology consumption and
existing failure/retry tests. Keep the larger runtime, placement and actor-group
modules pending until their complete reviews.

## Change

Session kill collected the surviving rank handles correctly, then retained every
rank of an engine if any rank failed. Consequently the session continued to
report already-killed ranks as owned and submitted kills for them again on retry.
The existing retry test used one rank and could not expose this grouping error.

The session now retains failed rank handles directly and clears its engine list
on teardown. It does not construct partial engines or mutate the original engine
objects held by the executor/health monitor. A subsequent close checks the owned
rank handles, so it retries failed kills even though the engine list is empty.

A new two-rank runtime shutdown regression failed before the fix: after one kill
succeeded and one failed, the owned-rank list still contained both. It now proves
only the failed rank remains, original topology is unchanged, and the second
shutdown retries precisely that actor before reaching TERMINATED.

## Retain and why

- Session owns concrete actor resources; runtime owns admission, stable failure
  and shared shutdown. Keep these responsibilities distinct rather than merging
  a small lifecycle method into a generic utility.
- `close` attempts graceful release then kills in finally. `force_close` upgrades
  an existing wait and future closes; cancelling the asyncio waiter does not
  preempt the background ray.get thread. Its bounded get and actor kill remain
  necessary. `kill_engines` is also the synchronous teardown path.
- `sleep_engines` returns validated evidence; `wake_engines` returns no report.
  Their shared deadline shape is understandable duplication across two lifecycle
  operations. A callback-based barrier wrapper would hide their result semantics.
  Preserve distinction between a rank-raised TimeoutError and barrier expiry.
- The two adjacent timeout constants are the session's isolated lifecycle limit
  table, with different reasons for parking versus graceful release. Do not
  create a separate two-value file or expose new YAML knobs just to move them.
- `kill_actors` performs the shared complete sweep and reports failures. Each
  owner directly retains its failed handles. The `kill_and_retain` callback
  adapter was removed; mapping through a generic callback added an unnecessary
  layer between the owner and this sweep.
- `require_ray`, current-node and GPU-ID helpers are lazy dependency/framework
  adapters. GPU string ordinals normalize at the Ray API boundary, with errors
  preserved rather than silently dropping devices. ClusterTopology.from_ray
  groups live-node resources relative to the current process for placement
  preflight. Keep fractional resource quantities as floats; they are not token
  IDs or gratuitous dtype conversions.
- API export lists are declarations. Ray node record keys and GPU resource names
  are protocol vocabulary, not misplaced model/algorithm taxonomies.

Non-goals: retry generation on surviving ranks, mutate original fleet topology,
promise actor termination merely because ray.kill returned, add another cleanup
manager, hide failed kills, or alter sleep/wake and remote-call deadlines.

## Validation

114 tests passed across runtime lifecycle, lease parking, dependency adapters and
cleanup logging. One upstream Ray environment warning. These include the new
controlled two-rank kill failure and existing real CPU Ray lifecycle cases;
they do not simulate a multi-node outage or prove physical GPU reclamation.
Ruff check passed for changed Python files; Ruff formatted the session module.
Previous isolated audit commit: `58a2c8cfd`.
