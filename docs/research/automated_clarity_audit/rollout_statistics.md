# Rollout statistics and iteration payload

Reviewed all of stats.py and orchestration/types.py, collector timing producers,
trainer sink construction, and statistics/orchestration/online-metrics tests.
Previous audit commit: d9ef3b713.

## Change

LoggingStatsSink previously returned immediately when its percentage denominator
was nonpositive. collect.* phases are intentionally excluded from that
denominator, so valid collection-only timings, counters and gauges vanished from
the log. Keep the full metric output and omit percentages when there is no
positive base. Empty stats remain a no-op; positive-base formatting and JSONL
output are unchanged. Nonpositive totals retain the historical displayed zero.

The regression constructs collection wall time, sample count and an inflight
gauge without training phases. Before the fix only total=0.000s was emitted;
afterward each metric appears without a percentage. This improves visibility,
not training performance or the interpretation of overlapping timings.

## Retain and why

- Phase durations, counters and gauges have different reductions: sum, sum and
  peak respectively. Keep these maps separate. Names are intentionally dynamic
  across families; introducing a fixed business-name table would restrict them.
- Reward latency samples enable merged percentiles; replacing them with summed
  p50/p95 values would be mathematically wrong. Optional timing totals preserve
  absent versus observed-zero semantics. _sum_optional is shared by merging and
  reward ingestion and has no persistent state, so a helper class adds nothing.
- fold_reward_timing validates all supplied reward timing fields before mutation.
  Its local timing normalization shares one finite/nonnegative boundary across
  plugin/service values. The reserved timing names protect the flattened metric
  schema; this is a real schema list, not algorithm vocabulary in workflow code.
  Do not add another checker object or remove that boundary indiscriminately.
- add_collection_timing computes overlap between generation and scoring timelines
  supplied by the collector. It assumes each individual timeline is disjoint;
  it is not a general interval-union utility. Preserve that stated scope.
- The phase context manager records failed phases in finally. Plural accumulation
  helpers reuse the corresponding reduction methods. They provide shared
  accumulation behavior, not arbitrary wrappers to flatten into callers.
- StatsSink is a consumer interface; LoggingStatsSink, JsonlStatsSink and
  MultiStatsSink own formatting, append IO and fanout respectively. The trainer
  installs the latter two outputs together. Keep the distinct owners and thin
  protocol boundary rather than move IO into RolloutStats.
- RolloutIteration attaches batches and stats to one handoff. Its default_factory
  avoids sharing mutable accumulators. RolloutScheduleMode's str+Enum shape
  preserves existing string representation; do not switch to StrEnum for style.
  Enum members and __all__ are API declarations, not misplaced ALL_CAPS data.

Non-goals: reducing every metric into one map, changing timing units/percentiles,
adding blanket numeric validation, or changing the JSONL append/resume policy.
Stats are per-request objects, not generally thread-safe shared accumulators.

## Validation and limits

The new logging regression failed before the production change. Then 70 stats,
orchestration and online-metrics tests passed. Existing tests cover summed
counters, peak gauges, reward percentiles, extra-name collisions, empty outputs,
JSONL rows and ordered sink fanout. Ruff check and format check passed on both
changed Python files. No GPU execution was needed.

Dynamic metric maps remain publicly mutable and flattened with update semantics;
cross-category name collisions are not globally rejected. Sink total is a
percentage base, not end-to-end elapsed time, since nested phases can overlap.
This pass does not claim otherwise or add runtime guards for every internal key.
