# Qwen-Image-2.1 aesthetic qualification: do not train this recipe yet

Completed September 25, 2026, from main `5ee89674`. The predeclared independent-preference gate did **not** pass. SigLIP V2.5 is functioning and repeatable, but this experiment does not establish that optimizing it will improve Qwen's outputs for users. No RL was launched, consistent with the user's conditional training instruction.

| Measure | Result |
|---|---|
| Generated qualification images | 64 prompts × 8 seeds = 512; all hashes verified |
| Real pretrained scoring | All 512 finite; two complete scoring passes identical, maximum difference 0 |
| Independent blind appeal comparison | Selected wins 32, loses 23, ties 9 |
| Preference utility, ties counted as 0.5 | **57.0%, paired 95% bootstrap CI 46.1–68.0%** |
| Prompt-adherence difference, 0–3 rubric | +0.031, 95% CI −0.125 to +0.188 |
| Predeclared adherence noninferiority margin | −0.15 points; overall gate passed |
| Aesthetic score, random → selected | 4.504 → 4.903; +0.398, 95% CI +0.326 to +0.477 |
| User's training gate | **Not passed: appeal confidence interval includes chance** |

Preference utility is `(wins + 0.5 × ties) / 64`, not a raw win rate. All nine ties were identical selections: the uniform-random baseline was allowed to select the same image as the best-of-eight arm. The confidence intervals resample prompts, not individual images. A positive aesthetic-score difference is expected by construction and is not independent quality evidence.

## What was run

The scorer is the repository's actual `AestheticRewardModel`, using SigLIP encoder revision `9fdffc58afc957d1a03a25b10dba0329ab15c2a3` and packaged V2.5 head SHA-256 `a6caf256b3dc434273c98dcb12f6bf17c5d4fc647d8df11f038386923dd06220`. Float32 inference on L40S. The model runtime was Torch 2.11.0+cu130, Transformers 5.17.0, Diffusers 0.41.0.dev0. Full environment provenance is in the artifacts.

Generation used Qwen revision `b3179ad355be050328e483a9dfdd9e60cd62adfa`, through the repository's native executor, with no LoRA, 512×512 RGB, 10 denoising steps, Flow-GRPO SDE noise 0.7, guidance 1.0, and individually seeded samples. Four independent GPU shards completed. This is the intended short pilot's training sampler; it is **not** the deployment-quality setting.

The 64 authored prompts cover people, nature, architecture, still life, illustration, stylized art, layouts/text, and composition/counting. Random comparison selection and A/B assignment used separate fixed seeds. Scores and arm identities were withheld from the visual reviewer until all 64 labels were saved and hashed. Review was by the GPT-6 assistant, independently of SigLIP, **not by human raters**. A reason and separate A/B adherence scores were recorded for every pair. One q00 image was seen during the generation smoke check before blinding; excluding that entire prompt still fails the preference gate: 57.9%, CI 46.8–69.0%.

The initial offline-scoring attempt passed file paths to an adapter requiring tensors. That attempt failed, was retained under `*_file_mode_failed`, and contributed no scores. The corrected canonical Evaluation uses decoded tensors and completed both passes. No model or editing training code was changed.

## Category results are diagnostic, not new training gates

Each category has only eight prompts. These exploratory differences should not be treated as independently established effects or used to retroactively select a passing task.

| Category | Wins / losses / ties | Preference utility | Adherence delta |
|---|---:|---:|---:|
| People | 6 / 1 / 1 | 81.3% | +0.250 |
| Nature | 3 / 4 / 1 | 43.8% | +0.250 |
| Architecture | 5 / 3 / 0 | 62.5% | 0 |
| Still life | 5 / 1 / 2 | 75.0% | +0.500 |
| Illustration | 5 / 3 / 0 | 62.5% | −0.125 |
| Stylized art | 5 / 3 / 0 | 62.5% | +0.125 |
| Layouts and text | 2 / 2 / 4 | 50.0% | 0 |
| Composition/counting | 1 / 6 / 1 | 18.8% | −0.750 |

The composition subset gives concrete failure examples. For q58, the predictor selected an illustration with **six fish** (5.159) over the candidate with the requested five (4.933). For q63 it preferred the muddled room layout (5.315) over the clearer arrangement (4.732). These are expected blind spots for an image-only scorer; they matter if it is used as the sole objective for a broadly capable model.

## Signal and shortcut checks

All 64 prompt groups have nonzero score spread. Mean within-group range is 0.837 (minimum 0.313, maximum 1.615). Thus this particular failure is not caused by every group having identical scores. The full continuous standard deviations/ranges are saved; no unvalidated absolute aesthetic-success threshold was invented to manufacture a binary spread pass.

On one fixed candidate per prompt, scoring the image after perturbation gave:

| Perturbation | Scores higher than original |
|---|---:|
| Gaussian blur, radius 3 | 24 / 64 (37.5%) |
| Additive pixel noise | 8 / 64 (12.5%) |
| Replace with black image | 7 / 64 (10.9%) |
| Replace with checkerboard | 4 / 64 (6.3%) |

Blur can sometimes improve an already damaged image, so these counts alone are not proof of an exploit. Black/checkerboard replacements nevertheless demonstrate that a scalar aesthetic score is not a prompt-fidelity safeguard. Opaque-alpha transforms are identity controls for these RGB images: all 64 tie exactly, as expected. Alpha diagnostics are retained separately and excluded from the RGB damage card.

The repository's supplementary pooled preferred/rejected AUC is 0.523 (95% CI 0.410–0.628; 55 preferred and 55 rejected samples). It is not a substitute for the primary within-prompt paired comparison because between-prompt score offsets can confound pooled AUC. The reward card correctly says **not ready for a training key**.

## Sampling limitation

A post-gate diagnostic generated four fixed prompts (q00, q16, q40, q56) with native 512/10 and native 1024/40 sampling, using the same seed labels. All eight images and their provenance were saved. Native samples are visibly much cleaner, especially at 1024/40. At the same 512/10 geometry, removing SDE noise also improves the watchmaker example markedly.

This means the qualification result applies to the **tested short noisy sampler**. It is not proof that SigLIP cannot select better normal-quality Qwen outputs, nor proof that Qwen itself has the visible defects in the qualification page under standard inference. Across resolutions, the seed labels are shared but the latent tensors differ in shape; this is a sampler diagnostic, not a tightly paired efficacy test. The diagnostic does not replace or rescue the failed 64-prompt gate.

## Conditional stages and conclusion

The 512 training prompts, 64 validation prompts and 120 unseen final-test prompts were frozen before any training, excluding exact prompt overlap. Sources and hashes are recorded in `splits.json`. No pretraining-decontamination claim is made.

Because the predeclared preference gate failed:

- The separate LoRA RL pilot was **not launched**.
- There are **no trained checkpoints** and no 120-prompt base-versus-checkpoint verdict.
- Post-RL adherence and diversity preservation are **unmeasured**; no improvement or non-regression claim is made.
- The 120 final-test prompts remain unused for checkpoint evaluation.

The delivered outcome is a negative qualification / inconclusive preference result, not a failed RL experiment. A next experiment could qualify a less destructive stochastic sampler and repeat the blind gate on fresh prompts, or test selection at native deployment settings. This report does not automatically authorize treating the current score gains as evidence to train.

## Artifacts and reproduction

- [Protocol](PROTOCOL.md), [commands and environment](README.md), [machine-readable results](results.json), [reward card](reward_card.md), [locked blind labels](blind_labels.jsonl).
- [All 64 comparisons, unblinded after labeling](../../../outputs/qwen21_aesthetic_qualification/index.html).
- [Original blind comparison page](../../../outputs/qwen21_aesthetic_qualification/blind/index.html).
- [Three-sampler diagnostic](../../../outputs/qwen21_aesthetic_qualification/sampler_diagnostic.html).
- Raw images, generation settings, content-bound scoring runs, repeatability, stress and bootstrap reports: `outputs/qwen21_aesthetic_qualification/`.

The experiment follows the repository [reward skill](../../../reward/skills/reward/SKILL.md). The stop decision comes from the user's predeclared preference gate; it does not require another approval request.
