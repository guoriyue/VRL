# Preference pairs and clean-latent storage

Reviewed full data/preferences.py and data/sft_latents.py, offline DataLoader
and DPO batch consumers, target encoding and online regularizer shard loading,
plus preference layout/loading and shard/recipe tests. Previous audit commit:
12fea9542. Larger trainer and producer modules were inspected only at these
contract call sites and remain pending full review.

## Change

Clarify PickAPicPreferenceDataset's actual loading and augmentation behavior.
Both hub-loading branches return an indexable dataset; streaming is used only to
read the bounded prefix before materialization. max_samples applies before tie
filtering, so it is not a promise of that many surviving pairs. Item access uses
the configured crop and flip, not always a center crop. Runtime behavior and
accepted dataset schemas are unchanged.

## Retain and why

- PreferenceBatch.collate is already the batch-owned DataLoader adapter. It
  creates [B,6,H,W] winner/loser pairs; stacked_winner_then_loser supplies the
  [2B,3,H,W] order used for VAE encoding and duplicated captions. Its shape check
  protects that convention for manually supplied batches too. Do not flatten
  these methods or introduce a second collator object.
- __getitem__ decodes both source images and applies transforms before channel
  concatenation. Bytes/PIL/tensors are different stages of this image dataset,
  not alternative trainer batch formats. The two hub branches implement bounded
  input consumption versus full indexable loading; forcing streaming=True for
  both would require an iterable/shuffle/restart redesign.
- CleanTargetRef.from_source unifies typed examples and rollout metadata across
  producer/consumer boundaries. Exactly one target identity avoids encoding an
  image under one key and later trying to read a video under another. This is
  the existing owner, not a reason for a generic artifact-identity utility class.
- save_sft_latents/load_sft_latents are paired file APIs without persistent
  loader state. SFT_LATENTS_SCHEMA_VERSION is a real serialization version;
  family/model/revision fields reject incompatible latent spaces. Their own
  module separates tensor persistence from prompt artifact path validation.
  A class containing the same two static methods would add no useful lifetime.
- Lazy torch imports in the shard module and optional datasets/torchvision
  loading remain at their actual use boundaries. Target/schema keys and public
  exports are genuine file/protocol contracts, not business vocabulary mixed
  into orchestration.

Non-goals: changing DPO pair order, image augmentation randomness, rewriting the
dataset as an iterable stream, changing latent normalization or schema, or
merging unrelated persistence and prompt-manifest code to reduce file count.

## Follow-up and limits

The shard loader checks container/schema/provenance/target names, while online
regularization compares each latent shape against rollout geometry before
transfer. weights_only=True is not a guarantee every stored value is a tensor;
no blanket tensor-validation claim is made here. Producer normally supplies
encoded tensors; any stricter external-value contract belongs at shard loading.

The shard writer uses Path(path), while its reader expands a leading tilde.
Writer publication is a direct torch.save rather than atomic replacement.
Review shared artifact publication/path policy before integrating these with
the broader requested path cleanup; this review does not claim crash-safe
publication or matching tilde behavior. Family/path/revision strings are
provenance comparisons, not a content digest of mutable model files.

## Validation

31 preference, shard and recipe-loading tests passed; Ruff passed for the changed
Python file. Preference tests verify bounded materialization/cache forwarding
with a fake hub provider and actual torchvision construction. Shard tests perform
real CPU save/load, provenance mismatch rejection and clean-target identity
checks. No external dataset download, GPU VAE encoding or new documentation-only
test was performed.
