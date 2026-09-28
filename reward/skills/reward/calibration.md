# Preference calibration

Use this when a reward should be a *fitted combination* of several scored axes.
It does not decide whether a single judge is good -- that is the qualification
card.

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
