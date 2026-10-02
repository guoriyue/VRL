# Reward card: GPT 5.6 Luna through the Codex CLI as an image judge, Qwen-Image-2.1 text-to-image

- **Kind**: closed API model, no token probabilities; scores come from its text output.
- **Access**: `codex exec -m gpt-5.6-luna -c model_reasoning_effort="high" -i <img1> <img2> -` with the question on
  stdin; ~10-20 s per call, 6 in parallel. Scripts: `/mnt/nvme/aes2/codex_judge/run_codex_{judge,rating,pointwise,rating_arms}.py`.
- **Ground truth**: the user's own 60 blind labels (2026-10-02, `/mnt/nvme/aes2/human_labels`): 30 pairs base vs
  stage-1 ck30 on held-out prompts (set A), 30 within-group shared-noise pairs from the qualification set (set B).
  The user preferred ck30 in 14/30 (46.7%) and flagged no image as fake.

## Protocols on the within-group pairs (set B, the pairs GRPO actually ranks)

| protocol | calls/pair | score gap between the two images | repeatability | agreement with user |
|---|---|---|---|---|
| forced choice, both orders | 2 | -- | same answer in both orders 19/30 | 0.483 |
| pairwise 0-100 per criterion, both orders × 3 | 6 | 1.9 pts (per-call sd 2.1) | split-half sign 20/30 | 0.467 |
| pointwise 0-100 per criterion × 3 per image | 6 | 4.0 pts (per-image sd 2.5) | split-half sign 16/30 | 0.552 |

Shared-noise siblings differ by less than the model can resolve: the gap it reports is at the level of its own
repeat noise, and the sign is a coin flip. Averaging more samples removes noise, not the absence of signal.
**Not usable as a GRPO group reward under the current sampler.** For reference, the local rewards on the same 30
pairs: HPSv3 0.27, PickScore 0.27, Qwen3-VL pairwise 0.30, natural criteria 0.43, SigLIP 0.47, three Sonnet
judges 0.39 -- every automatic ranker is at or below chance against this user.

## Base vs checkpoint pairs (set A, differences a judge can see)

Pairwise rating, both orders × 3: split-half sign 20/30, gap 3.2 pts (sd 3.0); **agreement with user 0.700**
(21/30; 0.75 on the 4 pairs with gap ≥ 5). It rates stage-1 ck30 *below* base overall (composite −1.9 pts, ck30
ahead in 11/30): genuine −8.6, fidelity +7.7, structure +2.1. Forced choice had said the opposite (ck30 66.7%).
The rubric with an explicit "genuine" criterion is what makes it agree with the user; the generic "which is better"
question agrees with the other model judges instead.

## Reading

- Usable now as a **held-out evaluation judge** for checkpoint-level differences, with the rating rubric, both
  orders and ≥3 samples; its "genuine" score is the closest thing we have to the user's "looks fake".
- Not usable as a training reward until within-group samples differ enough to be ranked (sampler change:
  more or noisier stochastic steps), after which this card's set-B numbers must be re-measured.
- No logprobs: the reward would be a sampled mean, not a probability; budget ~100 calls per 16-rollout round.
