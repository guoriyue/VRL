# Qwen-Image-2.1 aesthetic reward qualification

Objective: determine whether SigLIP Aesthetic Predictor V2.5 can improve independently judged visual appeal without reducing prompt adherence or diversity. Separate from editing experiments.

Frozen before generation: 64 authored qualification prompts, 8 categories with 8 prompts each. Generate eight seeds per prompt at the intended pilot sampler (512 square, 10 steps, Flow-GRPO SDE noise 0.7, guidance 1.0, RGB). Use the repository's native generation executor, Qwen revision b3179ad355be050328e483a9dfdd9e60cd62adfa. These are qualification data, not final test data.

Run real scorer inference twice and require finite, repeatable values (absolute tolerance 1e-6 for identical runtime/precision). Pin encoder revision and packaged head hash. Report within-prompt spread and stress transforms separately from preference evidence.

For each prompt, choose a uniform random candidate independently of scores, then compare it with the highest scoring candidate. Random selection includes the best candidate: identical selections count as ties. Randomize A/B assignment with a fixed seed; blind filenames and exclude scores/selection labels from review. Independent visual judge rates aesthetic preference (A/B/tie) and prompt adherence (0=wrong, 1=major misses, 2=minor misses, 3=fully meets) separately, with a reason. State whether labels are human or model judgments. This is not human preference evidence if judged by an AI. Keep mapping out of the judge input until labels are locked.

Training gate: prompt-level paired bootstrap, 10,000 resamples, aesthetic win utility (win=1, tie=0.5, loss=0) lower 95% bound >0.5. Prompt-adherence noninferiority: lower 95% bound of paired score difference >=-0.15 on the 0–3 scale (a predeclared practical margin, not exact equivalence). Show category breakdowns. A pass requires both. Report sample size uncertainty and style preference limitations.

Before any training, also inspect reward repeatability, shortcut sensitivity and actual training-sampler spread per the repository reward skill. Do not infer failure from mere lack of binary threshold crossings in a continuous aesthetic score; report meaningful variation and independent preference ordering. Do not train on an unqualified reward.

If qualified: separate LoRA, frozen-base KL regularization, short pilot with checkpoints and two training seeds if resources permit. Freeze 120 unseen test prompts before training; evaluate base/checkpoints with matched seeds at deployment settings, independent blind appeal/adherence and diversity metrics. Training prompts must be disjoint from qualification/test prompts. Early selection uses validation; reserve final test until checkpoint selection. No claim of broad improvement from aesthetic-score gains alone.

If gate fails: do not launch the conditional pilot; deliver negative/inconclusive qualification evidence, reproducible commands, blind review artifacts, and explicit unexecuted downstream stages. Keep original objective and report what can and cannot be concluded.
