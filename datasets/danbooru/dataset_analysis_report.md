# Danbooru Anatomy Dataset Analysis Report

## Scope

This report describes the current Anima anatomy prompt dataset generated from
`nyanko7/danbooru2023` metadata. The prompt stage uses `metadata/posts.tar.gz`
only; it does not download Danbooru image tarballs.

Current generated manifests:

```text
datasets/danbooru/anatomy/train_prompts.jsonl: 20000 rows
datasets/danbooru/anatomy/eval_prompts.jsonl:   1000 rows
```

## Source Metadata

The local Danbooru2023 metadata scan found:

```text
metadata rows:       6,862,469
first post id:       1
last post id:        6,899,125
metadata compressed: ~2.8 GB
metadata expanded:   ~20.7 GB
image archive size:  ~8 TB
```

The post id range is larger than the metadata row count because Danbooru ids are
not a dense contiguous set.

## Filter Funnel

The current prompt candidate filter is intentionally simple and metadata-only.
It filters for active solo anime character posts with anatomy-relevant tags.

```text
total posts:                         6,862,469
active, not deleted/flagged/banned:  6,397,702
active + no excluded tags:           4,019,885
+ solo:                              3,008,574
+ 1girl/1boy:                        2,969,386
+ full_body:                           401,148
+ anatomy tags:                        335,066
```

Raw status counts from the metadata:

```text
deleted: 382,438
flagged:      54
banned:   88,982
```

These raw status counts should not be treated as mutually exclusive buckets.

## Candidate Availability By Score

After the anatomy candidate filter:

```text
score >= 0:   334,918
score >= 5:   276,875
score >= 10:  203,809
score >= 20:  110,030
score >= 50:   31,431
```

`score >= 20` is large enough for high-quality positive image calibration, but
it is not balanced enough to be the only prompt source for RL.

## Bucket Availability

Natural Danbooru tag distribution is heavily skewed toward `feet_visible` and
`sitting_full_body`. This is why prompt generation uses quota balancing instead
of natural sampling.

| bucket | score>=5 | score>=10 | score>=20 | score>=50 |
| --- | ---: | ---: | ---: | ---: |
| feet_visible | 149,297 | 107,210 | 54,816 | 14,818 |
| sitting_full_body | 58,359 | 46,563 | 28,442 | 9,056 |
| hands_visible | 31,729 | 22,615 | 11,389 | 2,950 |
| standing_front | 19,863 | 13,742 | 7,055 | 1,916 |
| kneeling | 11,305 | 9,466 | 6,428 | 2,332 |
| walking | 2,947 | 1,990 | 880 | 173 |
| action_pose | 1,963 | 1,333 | 676 | 126 |
| running | 1,216 | 751 | 260 | 31 |
| standing_side | 196 | 139 | 84 | 29 |

## Current Prompt Build Policy

The current generated dataset uses:

```text
min_score: 5
preferred_min_score: 20
bucket_balance: quota
prompt_style: mixed
```

The builder prefers `score >= 20` rows, then falls back to `score >= 5` for
rare buckets. This keeps quality high without starving action, hand, and motion
buckets.

The prompt renderer mixes Danbooru tag-style prompts with controlled
language-style prompts. This reduces the risk that RL only improves tag prompt
behavior while hurting Anima's language-prompt alignment.

## Current Generated Distribution

Train bucket distribution:

| bucket | count |
| --- | ---: |
| action_pose | 3,318 |
| hands_visible | 2,918 |
| hand_focus | 2,118 |
| kneeling | 2,118 |
| walking | 2,118 |
| arms_visible | 1,717 |
| feet_visible | 1,518 |
| sitting_full_body | 1,518 |
| standing_front | 1,518 |
| running | 1,015 |
| standing_side | 124 |

Eval bucket distribution:

| bucket | count |
| --- | ---: |
| action_pose | 166 |
| hands_visible | 146 |
| hand_focus | 106 |
| kneeling | 106 |
| walking | 106 |
| arms_visible | 86 |
| feet_visible | 76 |
| sitting_full_body | 76 |
| standing_front | 76 |
| running | 50 |
| standing_side | 6 |

Prompt style distribution:

| split | tag | language |
| --- | ---: | ---: |
| train | 9,929 | 10,071 |
| eval | 497 | 503 |

Source score bands:

| split | score>=20 | 10<=score<20 | 5<=score<10 |
| --- | ---: | ---: | ---: |
| train | 16,139 | 2,075 | 1,786 |
| eval | 752 | 142 | 106 |

Uniqueness checks:

```text
train unique prompts:     20000 / 20000
train unique source ids:  20000 / 20000
eval unique prompts:       1000 / 1000
eval unique source ids:    1000 / 1000
```

## Build Command

```bash
python -m vrl.scripts.data.anime_anatomy build-prompts \
  --download-danbooru-metadata \
  --train-output datasets/danbooru/anatomy/train_prompts.jsonl \
  --eval-output datasets/danbooru/anatomy/eval_prompts.jsonl \
  --report-output datasets/danbooru/anatomy/prompt_report.json \
  --train-limit 20000 \
  --eval-limit 1000 \
  --min-score 5 \
  --preferred-min-score 20 \
  --bucket-balance quota \
  --prompt-style mixed
```

## Caveats

This report covers prompt metadata only. It does not prove that every source
image is visually suitable as a positive anatomy reward example. Positive image
manifests, hand crops, hard negatives, and reward calibration should be built
and audited separately.

For positive image classifier data, use stricter image-side checks and consider
`score >= 20` or `score >= 50`. For RL prompts, keep quota balancing because
natural Danbooru sampling overrepresents static full-body and feet-visible
examples.
