# Metrics schema and checkpoint alignment

Reviewed full trainers/metrics_io.py, online initialization and offline DPO CSV
construction, row/CSV tests and paired online output tests. Previous audit
commit: 6f73ccc95.

## Change

MetricsCSV resume alignment now uses shared atomic_file instead of a fixed
metrics.csv.tmp and its own flush/fsync/replace implementation. The old path
overwrote and removed any unrelated file at that name. The existing alignment
test now keeps such a file beside the CSV: it failed before the change and
passes with unique temporary siblings. Header and row-selection rules stay.

## Retain and why

- MetricsCSV already owns initialization, append and checkpoint alignment;
  prepare_metrics_csv is no longer a free function needing integration.
- OnlineMetricRow owns schema and formatting. _csv_field puts format and phase
  source on the field itself, avoiding a second mapping. _fixed_fields and
  _component_columns keep headers and values in one class-derived order.
  OnlineMetricsCSV initializes headers directly for run callers. These methods
  serve actual schema ownership rather than arbitrary function grouping.
- Dynamic names are checked before tuple conversion/lookup and during direct
  row construction. A string must not become individual character columns;
  malformed names should not fail as dictionary lookup errors. Do not delete
  these boundaries just because public construction paths overlap.
- Missing components are NaN; full-precision output preserves aggregated
  binary64 values independently of display rounding. Nonfinite measurements
  remain evidence rather than being suppressed by a generic finite checker.
- Resume retains positions strictly before the supplied next checkpoint
  position, drops an incomplete final line, and rejects incompatible headers
  or malformed complete rows. The caller supplies epoch/step explicitly; no
  checkpoint progress is inferred from CSV contents.
- Offline DPO supplies its own field-derived schema to MetricsCSV. Online uses
  paired standard/full-precision files. Keep one file owner and separate metric
  schemas rather than forcing all trainers into one row class.
- Persisted column/phase keys, legacy lookahead spelling and __all__ are real
  schema/API boundaries. No ALL_CAPS business vocabulary needs relocation.

Non-goals: changing column names, decimal formats, metrics math, append validation
policy, or adding another writer class. Shared publication owns filesystem work;
MetricsCSV retains resume semantics.

## Validation and limits

48 metric/CSV and JSON tests plus eight online metrics tests passed. The regression
checks retained rows, preservation of the unrelated .tmp file and absence of
temporary leftovers. Online tests verify both output files and full-precision
resume. Ruff passed for both changed Python files.

Only resume replacement uses the atomic context. Initial headers and append
retain ordinary writes. The two online files are updated sequentially, not in
one transaction; multiple writers remain unsupported. Shared publication's
directory durability and inode/permission limitations apply. Unique temporary
names prevent temporary-file collisions, not concurrent destination overwrite.
