# Preference calibration and the deployment receipt

Use this when a training reward should be a *fitted combination* of several
scored axes, or when a training run must be bound to exactly the scoring that
was validated offline. It does not decide whether a single judge is good --
that is the qualification card.

## 1. Collect human preferences blind

Declare pairs of scored samples (same `Evaluation`), grouped by source and split
before looking at any result:
```json
{"pair_id":"a-01","left":"p0012-s0","right":"p0012-s3","source_group":"p0012","split":"calibration","dimension":"overall","tags":["removal"]}
```
Calibration and holdout must not share prompts, source groups, media or assets
(masks included). Export a blinded page, judge, import:
```bash
python -m reward review-export --evaluation DIR --pairs review-pairs.jsonl --seed 42 --output outputs/preference-review
# open outputs/preference-review/index.html, judge A/B/tie/cannot judge, download review-answers.json
python -m reward review-import --review outputs/preference-review/audit.json --answers review-answers.json --output preferences.jsonl
```
The page hides scores, sample names and splits; it is presentation blinding only.
Unanswered pairs stay unlabelled. Several annotators use distinct pair ids.

## 2. Fit and evaluate on holdout

```bash
python -m reward fit --evaluation DIR --preferences preferences.jsonl --axes alignment quality --l2 0.1 --tie-margin 0.1 --output reports/frozen-combination.json
python -m reward evaluate --evaluation HOLDOUT_DIR --preferences preferences.jsonl --combination reports/frozen-combination.json --output reports/preference-holdout.json
```
Independent scorers join without rescoring: `--component semantic=DIR_A
--component locality=DIR_B --axes semantic/editreward locality/locality`; the
holdout uses the same aliases. Fix hyperparameters before reading holdout.
`apply` scores a new snapshot with the frozen weights, no labels needed.

## 3. Qualify for training (the receipt)

`qualify` re-scores the saved images through training's HTTP adapters, checks
every mapped raw axis matches within tolerance, and writes a deployment receipt:
```bash
python -m reward qualify --component semantic=DIR_A --component locality=DIR_B \
  --combination reports/frozen-combination.json --reward-config deployment/reward.yaml \
  --axis-mapping deployment/axis-mapping.json --atol 1e-6 --rtol 0 --output reports/reward-deployment.json
```
`deployment/reward.yaml` is a reward section (no `reward:` key) with unit component
weights and the intended `inference`; the mapping is `{"semantic/editreward":
"editreward/editreward", ...}`. Then in the training config:
```yaml
reward:
  calibration:
    deployment_path: outputs/reports/reward-deployment.json
```
`RewardDeployment.load` refuses a receipt whose recipe differs from the resolved
reward config. Limits: HTTP services and single float32 RGB/RGBA images only. The
receipt binds *what* is scored, not whether it is right: passing it supplies no
labels and no evidence that optimising the objective improves outputs.
