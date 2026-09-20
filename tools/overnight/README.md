# 2026-09-18 four-L40S execution

Production code is isolated at origin/main 3ab166bb. Experiment scripts and
receipts do not modify the user's original worktree or shared Python environment.

Live status: `outputs/overnight_20260918/STATUS.md` (refreshes every30seconds).
Manifest: `outputs/overnight_20260918/queue.json`.
Research notes: `docs/research/primary_fill_four_l40s_20260918.md`.

`queue_runner.py` owns an exclusive runner lock, launches one GPU job at a time,
checks rank receipts and numeric gates, and blocks dependent jobs after failure.
The scheduling window is12hours from supervisor launch; per-job timeouts apply.
A ready job with unmet prerequisites has NOT started. Pending entries without
commands are research blockers, not executable jobs. Historical diagnostic
failures remain failures even when a later deterministic control succeeds.

To add a reviewed job, create a JSON with id, command argv, gate, output,
depends_on, optional env/timeout_seconds, and either training updates/parity_limit
or probe ranks. Run `python tools/overnight/enqueue.py /path/to/job.json` using
the existing VRL Python. This locks edits and inserts the job ahead of the
conditional Wan14B fallback. Do not edit a running job or reuse an output directory.

Models staged in /dev/shm are volatile. The host-headroom guard can release only
the night's generated H3 random fixture while Wan runs, then rebuild it. No raw
NVMe initialization is performed. Checkpoint/receipt archiving for large Wan2.2
runs is explicit in their wrapper commands; the original HF cache is untouched.

The three miles arms are diagnostics with20updates each. C intentionally uses
the historical0.05 rollout/replay allowance; it cannot count as strict parity.
The SP comparison keeps the original0.02 pixel limit plus a single-rank repeat.
H3 is a random-weight trainer-only synthetic replay test, not end-to-end generation.

Storage update23:46PDT: user authorized one NVMe initialization. Future models
and large runs now use `/mnt/nvme/vrl-night-20260918`; original tmpfs paths for
migrated assets are compatibility symlinks. Active Wan paths move only after their
writers/auditors finish. `nvme_migration.jsonl` records checksum verification.
The user explicitly retained the original Wan parity=0 gate; H3 must remain
blocked on any nonzero result even though fixed-input loader controls passed.
