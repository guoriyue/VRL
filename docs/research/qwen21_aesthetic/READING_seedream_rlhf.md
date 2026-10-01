# Reading note: how Seedream does RLHF (2.0, 3.0, 4.0, RewardDance)

Read 2026-09-30 from the arXiv HTML of each report; quotes are verbatim. Written because our
preference-reward runs (HPSv3 + SigLIP, PickScore) raised the reward while the images drifted
toward one warm cinematic look and fine detail (hands, cables) did not improve.

Sources: Seedream 2.0 [arXiv 2503.07703](https://arxiv.org/html/2503.07703v1), Seedream 3.0
[arXiv 2504.11346](https://arxiv.org/html/2504.11346v1), Seedream 4.0
[arXiv 2509.20427](https://arxiv.org/html/2509.20427v3), RewardDance
[arXiv 2509.08826](https://arxiv.org/html/2509.08826v1).

## Where the details actually are

- **Seedream 2.0 §4.3** holds the full RLHF recipe (data, three reward models, ReFL-style
  optimization, iterative reward refresh, refiner RLHF).
- **Seedream 3.0 §2.3.3** is one paragraph: the CLIP reward models are replaced by a generative
  VLM judge, scaled from 1B to >20B.
- **Seedream 4.0** only lists RLHF as a post-training stage; no reward details.
- **RewardDance** is the paper behind the 3.0 judge.

## Seedream 2.0 §4.3 (the recipe)

Preference data. A 1M-prompt "Prompt System" from training captions and user input, filtered for
vague prompts. Images for annotation come from "various trained models", through a "cross-version
and cross-model annotation pipeline" so the reward models generalize across policies. Annotation is
"multi-dimensional fusion annotation (such as image and text matching, text rendering, aesthetic,
etc.)", meant to "forestall deficiencies in the RLHF stage, and foster the achievement of Pareto
optimality across all dimensions".

Reward models. Bilingual CLIP, "opting to utilize the output of CLIP model as the reward itself",
ranking loss. "Three distinct RMs: a image-text alignment RM, an aesthetic RM, and a text-rendering
RM"; the text-rendering RM "is selectively engaged when a prompt tag relates to text rendering".

Optimization. "Direct optimization of output scores computed from multiple RMs, akin to REFL" --
not policy gradient; they tried DPO and DDPO and kept ReFL for multi-reward optimization. Stability
came from "carefully adjusting learning rates, choosing an appropriate denoising time step, and
implementing weight exponential moving average". The DiT and the LLM text encoder are fine-tuned
jointly, which "significantly amplifies the model's capacity in image-text alignment and aesthetic
improvement".

Iterative refinement (the reward is refreshed offline, not online):

> i) We begin by utilizing the existing reward model to optimize the diffusion model. ii) Next, we
> conduct preference annotation on the refined diffusion model and train a bad-case-aware reward
> model. iii) We then leverage this updated reward model to further optimize the diffusion model.

They prefer this to "dynamically updating the RMs" for "stability and control".

Refiner. Texture is handled by a separate stage: a "score-based texture reward model" trained on
hand-collected high-texture images with random degradation, used to RLHF the 1024 px refiner.

## Seedream 3.0 §2.3.3

> we now utilize Vision-Language Models (VLMs) as the reward modeling framework ... we explicitly
> formulate instructions as queries and derive rewards from the normalized probability of the "Yes"
> response token ... We systematically scale the reward model from 1B to >20B parameters.

## RewardDance (the judge)

- Reward = `P("yes" | image1, image2, prompt, task instruction)`: a **pairwise** comparison against
  a reference image, with optional chain-of-thought (distilled from SEED-VL 1.5). A pointwise
  variant (prompt + one image + quality instruction) exists and is used as the inference-time
  verifier.
- RL uses **ReFL with Best-of-N references**: generate N samples with the SFT model, let the judge
  rank them, use the top one as the reference. Reference quality matters (Best-of-16 Top-2 83.6 >
  Best-of-2 Top-2 82.7).
- Judge sizes 1B–26B (InternVL). Larger judges keep **high reward variance late in RL**, which they
  read as resistance to hacking; small judges collapse to uniform high-reward outputs. The paper
  covers alignment; "aesthetic generation" is named as future work.

## What transfers to VRL

| Seedream | Ours |
|---|---|
| Large generative VLM judge, token-probability reward | `vlm_preference` (Qwen3-VL-32B FP8, batched per-criterion questions; graded expectation for aesthetics/defects, yes/no guards for structure/adherence) |
| Separate reward dimensions, selectively engaged | Per-criterion keys; weights per stage |
| Detail/texture handled in its own stage | Planned stage 2: mid/late SDE window with structure/defect criteria only |
| Bad-case-aware reward refresh | New failure modes go into the judge's criteria and the reward card's shortcut gate |
| Reward variance as a collapse signal | Watch `reward_std` per round; stop when it collapses |

Not transferable: ReFL (needs a differentiable reward; GRPO keeps the judge a black box), joint
text-encoder training, 1M prompts and a human annotation loop, 20B+ in-house reward models.
