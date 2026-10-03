# Reward card: off-the-shelf IQA / aesthetic predictors (pyiqa), Qwen-Image-2.1 text-to-image

Screened 2026-10-02 as candidates for a "clarity / aesthetics" term. All run through `pyiqa` 0.1.16 on CPU at 768 px
(`/mnt/nvme/aes2/iqa_sweep.py`, `analyze_iqa_sweep.py`). Arbiters: the user's 60 blind pairs (A = base vs stage-1
ck30, B = within-group) and GPT 5.6 Luna's rating protocol (pairs with |gap| ≥ 2, n = 27). Shortcut rate = share of
the 64 drift transforms of a genuine image scoring ≥ the original.

| model | trained on | user A | user B | GPT | noise25 | blur3 | **bright** | **sat** | smooth | **bloom** |
|---|---|---|---|---|---|---|---|---|---|---|
| musiq | KonIQ (photo quality) | 0.47 | 0.43 | 0.41 | 0.00 | 0.00 | 0.44 | 0.55 | 0.00 | 0.36 |
| maniqa | KonIQ | 0.43 | 0.40 | 0.37 | 0.00 | 0.00 | 0.23 | 0.66 | 0.00 | 0.19 |
| hyperiqa | KonIQ | 0.40 | 0.40 | 0.30 | 0.00 | 0.00 | 0.27 | 0.34 | 0.00 | 0.17 |
| arniqa | KonIQ | 0.43 | 0.40 | 0.26 | 0.00 | 0.00 | 0.20 | 0.45 | 0.12 | 0.12 |
| nima-koniq | KonIQ | 0.50 | 0.47 | 0.33 | 0.02 | 0.00 | 0.39 | 0.56 | 0.00 | 0.25 |
| topiq_nr | KonIQ | 0.43 | 0.47 | 0.37 | 0.00 | 0.00 | 0.33 | 0.80 | 0.00 | 0.23 |
| clipiqa+ | CLIP + KonIQ | 0.50 | 0.57 | 0.37 | 0.00 | 0.02 | 0.53 | 0.77 | 0.00 | 0.33 |
| liqe | CLIP multitask | 0.47 | 0.37 | 0.33 | 0.00 | 0.00 | 0.38 | 0.69 | 0.00 | 0.25 |
| musiq-ava | AVA (photo aesthetics) | 0.53 | 0.27 | 0.44 | 0.06 | 0.00 | 0.31 | 0.41 | 0.00 | 0.61 |
| nima-vgg16-ava | AVA | 0.53 | 0.57 | 0.48 | 0.02 | 0.03 | 0.50 | 0.45 | 0.09 | 0.55 |
| topiq_iaa | AVA | 0.40 | 0.27 | 0.44 | 0.03 | 0.00 | 0.55 | 0.56 | 0.00 | 0.64 |
| laion_aes | LAION aesthetics | 0.53 | 0.40 | 0.59 | 0.33 | 0.00 | 0.38 | 0.39 | 0.14 | 0.53 |

**Verdict: none qualifies.** Every predictor is at or below chance against both arbiters, and all of them prefer a
brightened or more saturated copy of a genuine image 20-80% of the time (the AVA aesthetics models also like bloom
50-64%). They were trained on real-photo distortions and photographer ratings; the "generic AI render" look that the
arbiters penalize (plastic smoothness, glow, over-saturation) scores as *good* for them. Noise and blur they do catch,
but the local judge already does. Not to be used as a training term; `musiq`/`topiq_nr` can stay as blur/noise
diagnostics on held-out evals.

What is used instead: a reward head distilled from GPT 5.6 Luna's pairwise ratings on held-out images
(`/mnt/nvme/aes2/distill_rm.py`, card to follow).
