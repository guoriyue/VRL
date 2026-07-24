# SPRINT: Ray cluster ownership and shared-host isolation

Status: **done (2026-07-10)**. Implemented and verified with fake-Ray and CPU-only
tests; no live Ray cluster or GPU experiment was used. It was split from
[`SPRINT_dino_reward_rl_trainability.md`](../info/SPRINT_dino_reward_rl_trainability.md)
because cluster ownership is independent of reward trainability.

## Outcome

Every online recipe must make one of two cluster choices explicitly:

1. A single-node, DDP, or FSDP rank starts and owns a fresh local cluster with
   `ray.init(address="local")`.
2. A cross-node driver attaches only to the operator-selected cluster named by
   a concrete `RAY_ADDRESS`.

No online-recipe startup path may rely on bare `ray.init()`, `address="auto"`,
`/tmp/ray/ray_current_cluster`, or whichever cluster happened to start most
recently. Recipe cleanup uses `ray.shutdown()` for the connection it opened;
repository production code never shells out to `ray stop`, and runbooks forbid
it on shared hosts.

This prevents accidental attachment and makes ownership observable. It does
**not** make a process immune to another user who can terminate it.

The same ownership rule applies to standalone `RayGenerationLauncher` use: its
default init kwargs select `address="local"`; an attaching caller must override
the address explicitly.

## Field evidence and limits

The available evidence proves external termination, not its source:

- `outputs/janus_smoke/aesthetic.log:591` records `SIGTERM` in a Ray generation
  worker.
- `outputs/janus_smoke/aesthetic_rbs24.log:1708` records another worker
  `SIGTERM`.
- `outputs/janus_smoke/baseline.log:3038-3040` records `Termination signal` and
  says PID `174794` received `SIGTERM` from PID `235998`.

These are 2026-06-21 Janus smoke logs, not direct evidence from the July DINO
continuation. They do not preserve the sender command, command line, owner, or
process lifetime. Therefore they cannot establish that `ray stop`, a test, OOM
handling, or any named neighboring job sent the signal.

The environment audit used Ray 2.55.1. Its `ray stop` implementation iterates
machine-local processes, matches Ray process names/command lines, and terminates
the matches. It does not accept a cluster address, namespace, port, or session
directory as a kill scope. This supports the statement that `ray stop` is unsafe
as per-run teardown on a shared host. It does **not** prove that the command was
used in the incidents above.

Ray is not a machine-wide singleton: separately addressed clusters can coexist.
The precise risk is that the `ray stop` CLI has a broad machine-local process
scan within the caller's PID visibility and permissions. Distinct ports,
namespaces, and temp directories isolate cluster startup and normal cluster
operations, but not that external scan. A dedicated host, Unix user permission
boundary, or container PID namespace is the operational protection.

At audit time, `gh issue list --state all` returned no repository issues, and no
local issue record identified the sender. The logs and the DINO sprint were the
only incident records found.

## Scope

### Cluster selection

- Keep the topology decision in one place in
  `vrl/scripts/common/online.py`.
- For `distributed.resources.cross_node=false`, call
  `ray.init(address="local")`, even when `RAY_ADDRESS` or a current-cluster
  pointer exists.
- For `distributed.resources.cross_node=true`, require a concrete
  `RAY_ADDRESS`; reject missing, empty, `auto`, and `local` values before actor
  or placement-group creation.
- Remove implicit cross-node auto-detection and implicit attachment.
- Preserve a pre-initialized Ray connection supplied by an embedding caller;
  the recipe did not create it and must not disconnect it.

### Ownership and cleanup

- Record whether the recipe owns a local cluster, attached to an external
  cluster, or received a pre-initialized connection.
- Log driver PID, resolved GCS address, session directory when available, and
  Ray version. These values are diagnostic provenance; they must not select
  business behavior after initialization.
- Tear down handle-based rollout and placement resources before disconnecting
  Ray.
- Use `ray.shutdown()` only for a connection opened by the recipe. For an owned
  local cluster it stops the processes created by that `ray.init`; for an
  external cluster it disconnects the driver without stopping the cluster.
- Attempt every cleanup step. Recipe-component cleanup errors must be visible,
  must raise when the run itself succeeded, and must not replace an earlier
  training error. Handle-level best-effort actor/placement cleanup remains
  non-raising, but logs failures with the owned handle instead of silently
  swallowing them.

### Graceful termination

- The CLI turns `SIGINT`/`SIGTERM` into asyncio task cancellation so the recipe's
  existing `finally` path runs before exit; the signal handler itself never calls
  Ray APIs.
- Stop and join a continuous rollout producer before closing its collector,
  runtime, or placement group.
- Terminal shutdown of a sleep-eligible generation lease must terminate its
  actor-owning worker fleet; `offload()` remains the reversible sleep/wake operation.
- DDP/FSDP strategies destroy their training process group after Ray resources
  are released.

### Documentation

- Keep the cross-node runbook explicit about `RAY_ADDRESS` and forbid
  unconditional `ray stop` on shared hosts.
- Describe DDP/FSDP local ownership as a result of
  `ray.init(address="local")`, not as a property of bare `ray.init()`.
- Keep the DINO sprint's claim limited to source-unknown external `SIGTERM` and
  link here for Ray work.

## Acceptance criteria

The implementation is complete when CPU-only tests prove all of the following:

- A non-cross-node run passes exactly `address="local"`, ignoring a stale
  `RAY_ADDRESS` value.
- A cross-node run passes the concrete operator address exactly.
- Missing, `auto`, and `local` cross-node addresses fail before placement or
  actor creation.
- An already initialized connection is neither reinitialized nor shut down by
  the recipe.
- A recipe-opened local or attached connection is shut down exactly once, after
  rollout/placement cleanup.
- `SIGINT` and `SIGTERM` unwind async cleanup and exit as 130 and 143; synchronous
  trainer entrypoints retain their existing signal behavior.
- Continuous schedule shutdown joins its producer, and sleep-eligible terminal
  shutdown cleans its worker fleet exactly once; the fleet has no public terminal
  state.
- DDP/FSDP strategy shutdown reaches the process-group teardown boundary.
- Cleanup continues after one component fails; the first cleanup error raises
  only when there is no earlier run error.
- Tests use a fake Ray boundary and do not launch Ray processes or require a
  GPU.
- Repository searches find no production shell invocation of `ray stop` and no
  bare `ray.init()` or `address="auto"` in the online-recipe startup path.
- The DINO, cross-node, DDP, and FSDP documents contain no claim that Ray is a
  machine-wide singleton or that the observed `SIGTERM` was confirmed as
  `ray stop`.

## Architecture hygiene

What should change:

- Cluster selection and connection ownership become one explicit decision
  instead of being spread across auto-detection and later initialization.
- Derived ownership state must drive shutdown behavior; it is not a log-only
  field.
- Runbooks must distinguish cluster-selection safety from process-isolation
  safety.

What should stay unchanged:

- `Runtime -> Executor -> Ray actor adapter -> WorkerCore` remains intact. These
  are real process and protocol boundaries, not thin files to flatten.
- Placement-group ownership, cross-node placement preflight, actor GPU
  validation, OOM splitting, and trainer-handoff sleep/wake remain intact.
- `vrl/ray/lifecycle.py` remains a thin framework adapter. Its separation keeps
  Ray imports and handle cleanup behind one protocol boundary; error semantics
  may be made truthful without flattening it into recipe code.
- Ray's upstream `RAY_PROCESSES` table remains a legitimate ALL_CAPS protocol
  table. This sprint does not copy or maintain a repository-local version.
- `_RAY_ADDRESS_ENV` is a justified ALL_CAPS constant because it names an
  external environment-variable boundary. It must not become a duplicated
  vocabulary or config table.
- A small cluster-session object is justified: it carries behavior-consumed
  ownership state and makes idempotent shutdown a lifecycle boundary. It is not
  a decorative manager or a log-only resolved struct.

## Non-goals

- Identifying PID `235998` after the fact or attributing the incidents to a
  person, test, or command without new evidence.
- Actor restart, retry policy, Ray fault tolerance, or checkpoint cadence.
- Building container, service-manager, Unix-user, or multi-tenant orchestration.
- Making distinct ports, namespaces, or temp directories a claimed defense
  against external `ray stop`.
- Changing rollout/reward placement, restoring a remote reward actor pool, or
  resolving inline reward device semantics.
- Running GPU experiments or a live Ray cluster as part of this sprint's unit
  verification.

## Verification

- Focused ownership/lifecycle/signal/strategy tests: `120 passed`.
- Wider scripts, generation-Ray, Ray substrate, rollout-orchestration, and
  strategy suite with `slow_test` and `e2e` excluded: `326 passed, 2 skipped,
  12 deselected`.
- Full repository Ruff check and `git diff --check`: passed.

## References

- Recipe ownership boundary: `vrl/scripts/common/online.py`
- Fake-Ray ownership tests: `tests/scripts/test_online_ray_cluster.py`
- Cleanup adapter: `vrl/ray/lifecycle.py`
- Cooperative signal boundary: `vrl/scripts/train.py`
- Rollout schedule shutdown: `vrl/rollouts/orchestration/{schedule.py,continuous/schedule.py}`
- Distributed strategy shutdown: `vrl/trainers/strategy.py`
- Cross-node runbook:
  `docs/training_examples/online_nft_kling_video_reward_cross_node/README.md`
- DDP runbook:
  `docs/training_examples/online_nft_kling_video_reward_ddp_2x1/README.md`
- FSDP runbook:
  `docs/training_examples/online_nft_kling_video_reward_fsdp_2x1/README.md`
- Incident logs: `outputs/janus_smoke/aesthetic.log`,
  `outputs/janus_smoke/aesthetic_rbs24.log`,
  `outputs/janus_smoke/baseline.log`
- Ray 2.55.1 `ray stop` source:
  https://github.com/ray-project/ray/blob/ray-2.55.1/python/ray/scripts/scripts.py
- Ray `ray.init` / `ray.shutdown` lifecycle documentation:
  https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html and
  https://docs.ray.io/en/latest/ray-core/api/doc/ray.shutdown.html
