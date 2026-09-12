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

## Open policy differences

Diffusion packing follows the stored observation device; AR packing first honors
context.device. The collector supplies CPU as the context device, but diffusion's
actual placement depends on storage policy. Do not infer that the context setting
alone guarantees all completed batches are on CPU. Unifying placement requires
examining supported storage policies and the cost/order of KL computation.

Diffusion copies reward_metadata and runtime_debug into batch context, while AR
copies only trajectory context. RewardSample metadata reaches scoring in both
paths. Deciding whether trainer-side context should also match requires checking
its consumers; moving this code into a shared constructor without that decision
would change observable batch metadata.

Diffusion currently reads the kl tensor even when kl_reward_coef is zero. Current
canonical denoise builders provide it. Removing this requirement is a contract
change rather than evidence that all distribution-specific packing is redundant.

These differences remain explicit review findings. No runtime edit or redundant
new test was introduced for this module review.

Validation: 79 CPU tests passed across collector runtime, R1 wiring, full-sequence
storage adoption and chunk binding tests. They exercise actual trajectory/batch
construction with model doubles and tensors, not pretrained GPU execution.
