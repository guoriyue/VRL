# Shared utilities review

All twelve baseline `vrl/utils` modules were read. Review also followed
representative actual consumers: profiler scopes in the generation worker and
online trainer; `profile_smoke`; reward server root validation; manifest data-root
normalization; family registry mapping conversion; policy publication in the Ray
runtime; and trajectory tree movement/storage. Call-site searches locate the
remaining subsystem reviews; they do not mark those consumer modules reviewed.

## Change

`profiling._write_manifest` now publishes through `utils.json_files.write_json`.
The existing shared writer owns same-directory temporary files, fsync, replacement
and cleanup. The profiler retains manifest contents and destination ownership.
Parsed JSON and replacement behavior remain the same; output now has the shared
writer's trailing newline and atomic publication. Temporary file permissions
follow the shared writer rather than `Path.write_text` creation permissions.
The trace and summary remain separate files, not an atomic three-file transaction.

## Decisions

| Module | Retained responsibility and rationale |
| --- | --- |
| `__init__.py` | Empty namespace/documentation leaf; no eager dependency exports. |
| `validation.py` | `require_int` is the dependency-free shared primitive. A checker class with one static method would add a namespace hop. Repeated use at internal sites needs case-by-case review, not deletion of this boundary primitive. |
| `config.py` | Keep three free adapters. `plain_mapping` preserves presence and only shallow-copies ordinary mappings; `to_builtin_deep` recursively normalizes containers. They are different contracts. `import_from_path` shares explicit `module:attribute` grammar. |
| `artifacts.py` | Keep `RootedPaths` for normalized root state; keep manifest-specific syntax/error conversion outside it. `DATA_ROOT_ENV` is an environment boundary, `IMAGE_SUFFIXES` an isolated extension taxonomy. `repo_root`, default/override normalization, and hashing are shared public operations with actual consumers. A universal safe saver would conflate read permissions with write publication. |
| `json_files.py` | Keep public JSON/JSONL readers/writers and one private atomic writer. Nested emit callbacks allow streaming JSONL into the same publication mechanism. No extra file-owner class is needed for stateless IO. |
| `memory.py` | Keep snapshot as values and monitor as acquisition/logger owner. `_read_fields_mb` reads both requested system fields in one pass. Proc absence remains unknown, not zero. |
| `cuda_memory.py` | Keep CuMem's stateful owner and optional allocator import seam. Best-effort cleanup and strict parking cleanup intentionally propagate errors differently. Peak counters and physical per-process ownership measure different quantities. `CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT` is an environment-configurable backend handoff limit, not an algorithm list. Do not infer correctness of its calibration from CPU tests. |
| `deadline.py` | Keep absolute monotonic budget, stable timeout exception and shared timeout normalization. Constructor validation must also support direct runtime callers. Do not merge timeout conversion with integer validation: timeout accepts numeric conversion while integer counts deliberately do not. |
| `lifecycle.py` | Keep lock-protected phase and first failure ownership. `publication_guard` linearizes assignments against health-thread failure; it is more than an idempotent shutdown flag. `_require_running_locked` avoids recursively acquiring the non-reentrant lock in both public entry paths. Enum members are protocol states. |
| `logging.py` | Keep process namespace initialization, live stdout handler and `kv` formatting facade. `_ROOT_NAME` names the logger namespace; `_FORMAT` defines its text format. The stdout property is a stdlib adapter needed for Ray/test redirection. Initialization helper isolates that module-owned operation. |
| `media.py` | Keep stateless media IO/conversion adapters. Unit-range reward conversion differs from uncertain-range external-image conversion. No generic converter class; explicit layout/range contracts need a later caller migration before heuristics can be removed. |
| `profiling.py` | Keep distinct stage annotation and trace-capture contexts. Activity selection derives backend names from torch. Summary rendering, handler filename discovery and worker-label sanitization have concrete framework/output boundaries. The manifest writer now shares publication mechanics. No new generic trace base or callback wrapper class. |

## Findings for subsequent consumer review

- Media converters currently infer range/layout. An HWC image with a small first
  dimension can be ambiguous against CHW. Existing tests establish ordinary
  layouts, not every ambiguous shape. Trace producers and declare layouts before
  changing this public conversion behavior; do not silently reinterpret images.
- CuMem availability probing catches all initialization exceptions and returns
  `None`; the required path then loses the original cause. Review that distinction
  with parking consumers and their optional-dependency tests before changing
  availability semantics. No allocator code changed here.
- `TorchProfilerConfig` remains mutable and normalizes on construction and read.
  Removing repeated conversions requires checking post-construction assignments;
  constructor annotations alone do not establish immutability.
- Utility source review does not establish that every caller's `require_int` or
  timeout check is needed. Those remain in the pending module coverage.

## Validation

78 tests passed, 1 GPU test deselected, 1 torch profiler warning about cycle event
retention. Command scope: `tests/utils`, `tests/test_logging.py`,
`tests/ray/test_operation_deadline.py`, with `-m 'not gpu'` and
`CUDA_VISIBLE_DEVICES=''`. Includes real CPU trace export plus manifest/summary
readback, atomic JSON writer behavior, path escape cases, host-memory acquisition,
NVML fakes, conversion and timeout tests. No GPU/CuMem or real distributed run.

Previous groups: `3e9c1250c` reference context; `d09493da4` trajectory simplification.
