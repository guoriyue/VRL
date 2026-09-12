# Collector batch builder

Reviewed complete collector/batch_builder.py, collector construction/scoring call
sites and tests for reward samples, latent Gaussian KL, nonlatent Gaussian tokens,
multisegment R1 packing and gatherer storage-policy adoption. Previous audit
commit: bccdffd96. Disposition: retain implementation, record policy asymmetries.

## Retain and why

- One builder owns an output and storage context across reward sample extraction
  and final batch packing. This is actual shared state, not a class created solely
  to hold static functions. Storage policy intentionally mutates the gatherer's
  trajectory; the adoption regression asserts object identity as well as dtype.
- reward_outputs implements the declared scoring view and normalized numeric
  range. Exactly-one-view/reference checks guard an explicit scoring boundary;
  selecting an arbitrary first view would silently change the reward target.
- _batch_size accepts tensor-shaped and sequence-valued reward outputs, validates
  a batch dimension and translates unsupported values into a useful error. It
  should remain local to this adapter rather than become a general tensor checker.
- Reward sample count must match sample rows. REWARD_GROUP_ID_METADATA_KEY is a
  shared schema key, not a business-name table. Reserved metadata rejection keeps
  caller data from overriding collector-owned group identities.
- build routes latent Gaussian and flow-matching segments to diffusion packing;
  nonlatent Gaussian and categorical policies use AR packing. Distribution alone
  is insufficient: the latent path applies the recorded per-sample KL penalty.
  Existing tests verify both Gaussian interpretations and sum chunk/transition KL.
- Explicit multisegment layout preserves all segment tensors and primary segment
  identity. The generic all-categorical compatibility path stays unchanged; it is
  an existing public-context default, not a reason to invent another contract class.
- _primary_trainable_segment checks the declared primary selection rather than
  guessing from insertion order. _group_ids is the common sample-row projection
  onto each packed batch's device. Both provide useful consistency across packers.

Non-goals: merge AR and diffusion reward policy, remove primary/view validation,
change reward normalization, detach or clone trajectories, or add a distribution
dispatch registry simply to remove three branches.

## Policy differences and follow-up evidence

Diffusion packing follows the stored observation device; AR packing first honors
context.device. The collector supplies CPU as the context device, but diffusion's
actual placement depends on storage policy. Do not infer that the context setting
alone guarantees all completed batches are on CPU. The production-path follow-up
below explains why this is not currently evidence of unwanted GPU placement.

Diffusion copies reward_metadata and runtime_debug into batch context, while AR
copies only trajectory context. RewardSample metadata reaches scoring in both
paths. Deciding whether trainer-side context should also match requires checking
its consumers; moving this code into a shared constructor without that decision
would change observable batch metadata. The SFT consumer is identified below;
AR debug propagation remains open.

Diffusion currently reads the kl tensor even when kl_reward_coef is zero. Current
canonical denoise builders provide it. Removing this requirement is a contract
change rather than evidence that all distribution-specific packing is redundant.

These differences remain explicit review findings. No runtime edit or redundant
new test was introduced for this module review.

Validation: 79 CPU tests passed across collector runtime, R1 wiring, full-sequence
storage adoption and chunk binding tests. They exercise actual trajectory/batch
construction with model doubles and tensors, not pretrained GPU execution.

## Follow-up: trace actual placement and metadata consumers

Following 07b8ae798, inspect GenerationWorkerCore.forward_batch's successful
result: output passes through _to_cpu before GenerationBatchResult publication.
The recursive copy detaches tensor leaves, queues pinned CUDA-to-host copies and
synchronizes before returning. The pipelined path separately joins copy events
before returning host payloads (previously reviewed in pipeline.md). Ray executor
passes those payloads to its gatherer. Thus the current worker path establishes
CPU residency upstream; a preserve storage policy does not put these tensors back
on the trainer GPU. OnlineTrainer's replay sweep later invokes
move_training_batch_to_device with the trainer device and its deferred-replay
option.

Disposition: retain the builder's existing placement semantics. No production
residency bug has been established from the differing device expressions alone.
Custom GenerationRuntime implementations and direct builder callers can supply
other devices; the protocol does not declare an enforced host-only result. A
future runtime must explicitly decide its placement contract rather than infer it
from RolloutBatchBuildContext.device. Do not add another full tensor walk solely
to normalize already-host-owned production results.

The located trainer reward_metadata consumer is _compute_sft_loss, which uses
CleanTargetRef and the diffusion evaluator scheduler to find/noise clean latents.
That explains the diffusion-specific context copy. No AR consumer requiring this
SFT metadata was found in the trainer search. Retain this behavior unless an AR
consumer supplies a concrete need.

runtime_debug has a separate consumer: the trainer's parity diagnostic writes
batch.context['runtime_debug'] into its training_debug.jsonl evidence. The AR
packer does not copy GenerationOutput.runtime_debug, so it can omit that evidence
even though scoring metadata is intact. This remains a distinct follow-up to
trace AR diagnostic production and mode gating; it should not be silently folded
into an unrelated device-policy unification.

This follow-up is source/caller evidence, not a new CUDA experiment. No runtime
change or repeated test run was needed. Worker, Ray executor and trainer modules
remain pending their complete module reviews; these scoped reads do not mark
them covered.
