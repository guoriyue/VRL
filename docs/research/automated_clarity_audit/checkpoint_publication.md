# Checkpoint publication (scoped review)

Following 0a1b93821, inspected save_training_checkpoint, adapter export resolution,
rank agreement helpers, EMA parameter selection, artifact/metadata writing and
directory publication. Checkpointing remains pending its full module review.

Change: correct publication comments that claimed every artifact was fsynced and
an existing checkpoint was replaced atomically. The implementation fsyncs
checkpoint.pt, writes adapter/metadata files, then publishes the staging directory
and fsyncs its parent. Existing final directories are removed first. Runtime
behavior is unchanged; documentation now describes that actual sequence.

Retain and why:

- Save owns rank-stage ordering; only final file publication is primary-only.
  State export can enter FSDP collectives, so rank0-only gating the whole save
  would be wrong. Local error capture lets peers agree before the next stage.
- _checkpoint_ranks_agree_bool distinguishes unanimously enabled/disabled EMA
  branches. _checkpoint_stage_agreement also serves local-only callers. These
  helpers encode shared control-flow semantics, not generic defensive wrapping.
- _checkpoint_trainable_parameters preserves the model's EMA parameter order
  while checking checkpoint roots describe the same parameters. Sorting tensor
  names here would not preserve that order.
- Adapter export resolution maps actual module identity to checkpoint root and
  namespace. Named PEFT adapters add a directory level, so effective output paths
  differ from the requested root paths. Existing overlap/path checks protect real
  artifact publication; no additional safe-path class is needed.
- Raw checkpoint state is gathered before temporary EMA swapping. EMA artifact
  state is separately gathered and raw weights restored before file publication.
  These snapshots serve distinct resume versus export purposes.
- _write_checkpoint_artifacts_and_publish separates primary I/O from all-rank
  orchestration. _publish_checkpoint_dir owns rename and parent fsync. Keeping
  these operations explicit aids review more than another writer wrapper would.
- Checkpoint schema and artifact filenames are protocol constants and stay where
  shared readers/writers can find them. No workflow taxonomy needs extraction.

Non-goals: add backup generations, locks, timeouts, restore services or failure
tests. This review does not change adapter serialization or collective recovery.

50 existing selected checkpoint tests passed on CPU (publication, EMA and adapter
export), with dependency warnings. These cover state restoration and error
propagation using strategy doubles and actual temporary files; they do not prove
multi-rank crash recovery or power-loss durability. Ruff checks passed. No tests
or runtime gates were added.

Known limits: removing an existing final directory creates a gap before rename;
a crash there can lose that checkpoint name. Metadata and adapter content are not
individually fsynced by this module. Stage agreement cannot recover a rank already
stuck or dead inside a collective. None of those stronger guarantees should be
inferred from staging, nor added as speculative infrastructure in this cleanup.
