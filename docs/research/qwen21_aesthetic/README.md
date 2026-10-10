# Qwen-Image-2.1 / SigLIP aesthetic qualification

Protocol: [PROTOCOL.md](PROTOCOL.md). The attached user objective is the authority; RL is conditional on independent blind preference and prompt-adherence gates. All data are separate from editing experiments.

Environment used: `/mnt/nvme/venv-main/bin/python`, `HF_HOME=/mnt/nvme/hf`, `PYTHONPATH=/home/ubuntu/VRL`. Install `aesthetic-predictor-v2-5==2024.12.18.1` without changing the model runtime's pinned Transformers/Diffusers. Four L40S GPUs, one generation shard each. No paid external judge APIs. Labels are independent assistant visual judgments, not human ratings.

From the repository root:

```bash
export HF_HOME=/mnt/nvme/hf
export PYTHONPATH=/home/ubuntu/VRL
CUDA_VISIBLE_DEVICES=3 /mnt/nvme/venv-main/bin/python docs/research/qwen21_aesthetic/scorer_smoke.py --image /absolute/path/to/a/real/image.jpg
# Run each SHARD in {0,1,2,3}, with CUDA_VISIBLE_DEVICES set to that GPU.
CUDA_VISIBLE_DEVICES=0 /mnt/nvme/venv-main/bin/python docs/research/qwen21_aesthetic/generate.py --manifest docs/research/qwen21_aesthetic/qualification_prompts.jsonl --out outputs/qwen21_aesthetic_qualification/generation --shard 0
/mnt/nvme/venv-main/bin/python docs/research/qwen21_aesthetic/assemble.py
CUDA_VISIBLE_DEVICES=0 /mnt/nvme/venv-main/bin/python -m vrl.scripts.rewards.rescore_media --manifest outputs/qwen21_aesthetic_qualification/media.jsonl --config docs/research/qwen21_aesthetic/scorer.yaml --output-dir outputs/qwen21_aesthetic_qualification/scores
CUDA_VISIBLE_DEVICES=1 /mnt/nvme/venv-main/bin/python -m vrl.scripts.rewards.rescore_media --manifest outputs/qwen21_aesthetic_qualification/media.jsonl --config docs/research/qwen21_aesthetic/scorer.yaml --output-dir outputs/qwen21_aesthetic_qualification/scores_repeat
/mnt/nvme/venv-main/bin/python -m reward repeat outputs/qwen21_aesthetic_qualification/scores outputs/qwen21_aesthetic_qualification/scores_repeat --output outputs/qwen21_aesthetic_qualification/repeat.json
/mnt/nvme/venv-main/bin/python -m reward health --evaluation outputs/qwen21_aesthetic_qualification/scores --output outputs/qwen21_aesthetic_qualification/health.json
/mnt/nvme/venv-main/bin/python docs/research/qwen21_aesthetic/prepare_blind.py
# Judge only blind/cases.jsonl and blind/*_pair.jpg; save independent labels to blind/labels.jsonl.
# Never inspect selection_key.jsonl before judgments are locked.
/mnt/nvme/venv-main/bin/python docs/research/qwen21_aesthetic/analyze_blind.py
```

`generate.py` resumes only hash-verified existing images. Scoring resumes with `--resume`, using the canonical content-bound reward Evaluation. Original raw predictor scores are not probabilities. The baseline in the blind test is sampled uniformly from all eight candidates, including the selected best candidate; identical pairs must be ties.

Frozen possible downstream splits: `train_prompts.jsonl` (512), `validation_prompts.jsonl` (64), `test_prompts.jsonl` (120). `splits.json` records source hashes, selection seed, and exact-overlap exclusions. These do not establish pretraining decontamination. Final-test prompts are not used in qualification.

Additional completed checks:

```bash
/mnt/nvme/venv-main/bin/python -m reward stress-manifest --manifest outputs/qwen21_aesthetic_qualification/stress_source.jsonl --output-dir outputs/qwen21_aesthetic_qualification/stress_media
CUDA_VISIBLE_DEVICES=2 /mnt/nvme/venv-main/bin/python -m vrl.scripts.rewards.rescore_media --manifest outputs/qwen21_aesthetic_qualification/stress_media/media.jsonl --config docs/research/qwen21_aesthetic/scorer.yaml --output-dir outputs/qwen21_aesthetic_qualification/stress_scores
/mnt/nvme/venv-main/bin/python -m reward stress --evaluation outputs/qwen21_aesthetic_qualification/stress_scores --output outputs/qwen21_aesthetic_qualification/stress.json
CUDA_VISIBLE_DEVICES=0 /mnt/nvme/venv-main/bin/python docs/research/qwen21_aesthetic/generate.py --manifest docs/research/qwen21_aesthetic/sampler_diagnostic_prompts.jsonl --out outputs/qwen21_aesthetic_qualification/diagnostic_512_10 --shard 0 --shards 1 --samples 1 --resolution 512 --steps 10 --denoise-mode native
CUDA_VISIBLE_DEVICES=1 /mnt/nvme/venv-main/bin/python docs/research/qwen21_aesthetic/generate.py --manifest docs/research/qwen21_aesthetic/sampler_diagnostic_prompts.jsonl --out outputs/qwen21_aesthetic_qualification/diagnostic_1024_40 --shard 0 --shards 1 --samples 1 --resolution 1024 --steps 40 --denoise-mode native
/mnt/nvme/venv-main/bin/python docs/research/qwen21_aesthetic/summarize.py
```

See [REPORT.md](REPORT.md) for the completed negative qualification verdict. A native-sampler diagnostic does not change the frozen noisy-sampler gate. No RL or checkpoint evaluation was run after that gate failed.

To regenerate the supplementary evidence before running `summarize.py`:

```bash
/mnt/nvme/venv-main/bin/python docs/research/qwen21_aesthetic/prepare_evidence.py
/mnt/nvme/venv-main/bin/python -m reward agreement --evaluation outputs/qwen21_aesthetic_qualification/scores --labels outputs/qwen21_aesthetic_qualification/outcome_labels.jsonl --contrasts outputs/qwen21_aesthetic_qualification/contrasts.json --axis aesthetic --gate 0.85 --output outputs/qwen21_aesthetic_qualification/agreement.json
/mnt/nvme/venv-main/bin/python -m reward card --name siglip-v2_5-qwen21-t2i-512-sde --kind learned --axis aesthetic --repeat outputs/qwen21_aesthetic_qualification/repeat.json --agreement outputs/qwen21_aesthetic_qualification/agreement.json --stress outputs/qwen21_aesthetic_qualification/stress_rgb.json --blind-spot 'Primary paired preference gate failed; see blind_verdict.json. Pooled AUC is supplementary.' --blind-spot 'Continuous spread measured in continuous_spread.json; no calibrated binary success threshold.' --output docs/research/qwen21_aesthetic/reward_card.json
```
