# Anatomy Rubric v1 — anime image anatomical plausibility (2-axis)

This rubric is loaded verbatim into a Claude-as-judge reward prompt.
The reward score is computed by code from the per-axis scores you
emit — you do not compute the final aggregate yourself.

You output JSON with **two axes**: `hands` and `body`. The reward
function aggregates via `round((geom_mean(axes) + min(axes)) / 2)`
into a single scalar for RL training.

---

## The two axes

### 1. `hands` — quality of every visible hand and its fingers

This is the hardest axis and the most common failure point. **Do not
score this from a first-glance impression.** Run the enumeration
checklist below for each visible hand. If both hands are fully
occluded (in pockets, behind body, inside gauntlets, gripping a prop
that hides them), set `hands` to `null`.

### 2. `body` — anatomical plausibility of the rest of the body

This is everything except the hands: head, neck, torso, arms (the
shoulder→elbow→wrist chain, not the hand itself), legs, feet, and
overall pose coherence. Score it as a single integer reflecting
whether the body looks like a plausible human pose to a viewer.

These two axes are deliberately the only axes. Hands are isolated
because they fail most often and benefit from focused scoring;
"body" is everything else because limb attachment, joint angle, and
pose coherence are tightly coupled and humans judge them as one
gestalt.

## Core principle: visibility-gated, occlusion never penalizes

You can only score what you can see.

- **Visible and anatomically wrong** → low score on that axis.
- **Not visible** (occluded by clothing, props, body parts, or out
  of frame) → set that axis to `null`. Do not impute, do not
  penalize.

A fist with no visible separated fingers is **not** "hands occluded";
the fist itself is visible and you should score whether the fist is
drawn correctly (round palm, knuckle line plausible, thumb wrapping
correctly). Only set `hands = null` when the hands literally cannot
be seen.

## `hands` enumeration checklist (run this for each visible hand)

For each visible hand in the image, work through these checks in
order before assigning the score. Do not skip steps.

1. **Count fingertips.** Look at the hand carefully and count the
   distinct fingertip features. A normal open or partially-open
   hand shows up to 5 distinct fingertips (thumb + 4). A fist or
   curled hand shows 0–2. Write down the count in your head before
   moving on.

2. **Compare count to gesture.** Is the count consistent with the
   pose?
   - Open hand → exactly 5 fingertips visible.
   - V-sign / peace sign → 2 fingertips extended + 2–3 folded.
   - Point → 1 fingertip extended.
   - Fist → 0 fingertips, knuckles or curled-finger silhouette.
   - Holding object → fingertips wrap around the object, count
     matches the grip.
   - **6+ fingertips** in any pose → defect, regardless of style.

3. **Check finger geometry per finger.** For each finger you can see:
   - The proximal phalanx (closest to palm) should be the longest
     segment, distal phalanx the shortest.
   - Joints (knuckles) should bend toward the palm, not away.
   - Adjacent fingers should not visually fuse into a single blob
     unless the gesture explicitly closes them.

4. **Thumb radial position.** The thumb should be on the radial
   side of the palm (toward the body when the palm faces forward),
   visibly opposing the other four fingers. A thumb that reads as
   "just another finger" out the side is wrong.

5. **Wrist attachment.** The hand must visually connect to the
   wrist with no gap, no kink, no skin-tone mismatch. A hand floating
   above a sleeve cuff with no wrist behind it is a defect.

6. **Knuckle alignment.** When the hand is open or partially open,
   the knuckle bases of the four fingers should roughly align on a
   curve across the palm (not zigzag). The middle knuckle of each
   finger should be visible as a faint joint indication.

### Hand score derivation

Score the hands axis by the **worst** visible hand (if both hands
are visible, take the lower of the two single-hand scores).

| score | meaning |
|---:|---|
| **10** | Hand passes every check. All fingertips correct count, proportional, attached to wrist with clean knuckle alignment. |
| **8–9** | Passes every check at low scale; one barely-visible imprecision allowed. |
| **6–7** | One clear visible defect: a single finger fused with neighbor, OR slight knuckle misalignment, OR one finger drawn slightly too long. |
| **4–5** | **Two failure modes both fall here:** (a) multiple finger-level defects; OR (b) the hand is drawn at a scale where fingertips cannot be reliably counted — that's a defect of either the drawing or the resolution, and either way RL should drive toward making hands clearer / countable. Also: extra fingertip count, mitten fusion, thumb on the wrong side. |
| **2–3** | Hand is barely recognizable as a hand: severe finger fusion, hand floating from wrist, fingers radiating in impossible directions. |
| **0–1** | Catastrophic: no recognizable hand structure. |

**You may not score above 7 without explicitly stating you counted
fingertips and verified the count matches the gesture.** Add
"fingertip count: N, gesture-consistent" or similar to your
`defects` list. If you cannot count, you cannot score above 5.

Anti-cluster rule: do not default to 7 when uncertain. Either count
the fingertips (then score ≥ 7 if clean), or admit you cannot count
(then score ≤ 5).

## `body` scoring guide

Score this axis on whether the body — excluding hands — is
anatomically plausible.

What to look for:

- **Limb attachment**: arms attach at shoulders, legs at hips, head
  at neck. No floating limbs, no dislocations, no joint-on-the-
  wrong-side-of-the-body errors.
- **Joint plausibility**: elbows and knees bend the right way. No
  hyperextension backward.
- **Bilateral symmetry**: when both limbs of a pair are visible
  and should match, they are roughly the same size and length.
- **Pose coherence**: shoulder line and hip line are consistent
  with the pose. In a static stand, they should be parallel; in a
  twist or contrapposto, the twist should be plausible.
- **Proportions**: head, torso, arms, legs proportional to each
  other. Chibi / stylized proportions are fine if internally
  consistent.
- **Foot attachment**: feet anchored to ankles, toe direction
  consistent with leg direction. Footwear shape matches feet.
- **Face**: if visible, 2 eyes, 1 nose, 1 mouth in reasonable
  positions. No melted face. Face is part of body, not hands.

| score | meaning |
|---:|---|
| **10** | Body is anatomically clean across every visible region. Reserve this for cases where you cannot identify even a minor imprecision after looking carefully. |
| **9** | Reads clean at first look, but on a careful pass you note one tiny detail you'd nitpick. |
| **8** | One minor visible issue (slight asymmetry between left/right limb, slightly off shoulder slope, a foot/ankle angle that's not quite right). |
| **7** | One clearly visible body defect — shoulder/hip rotation mismatched in a static pose, elbow slightly hyperextended, one leg subtly shorter or differently angled than the other, foot pointing in an awkward direction relative to leg. |
| **6** | Multiple visible defects, OR one clear dislocation (arm attaching at wrong shoulder position, knee joint placed wrong on the leg). |
| **4–5** | One severe defect (limb fully detached, joint bending wrong way, melted face) OR several clear defects. |
| **2–3** | Body is severely broken. |
| **0–1** | Catastrophic. |

Anti-cluster rule: do not default to 8–9 because the pose "looks
roughly OK". A pose that's roughly OK with one specific identifiable
issue is a **7**. Reserve 8–9 for cases where you have to actively
hunt for an issue and find at most a tiny one.

## Stylization is not a defect

- Chibi / super-deformed proportions — fine if internally consistent.
- Foreshortening in dynamic poses — fine if consistent.
- Long flowing hair occluding a limb — that's occlusion.
- A hand drawn at a small scale with simplified fingers — score
  conservatively on the hand checklist; if you can't reliably count
  fingertips because of scale alone, score 6–7 (one mild issue), not
  4 (severe defect). But if you can clearly see 6 fingertips or a
  mitten shape at any scale, that's a defect.

## Aggregating the two axes into a single score

After you assign `hands` and `body` integers, compute the final
`score` value as a float in `[0, 1]`:

```
visible = [v for v in (hands, body) if v is not None]
geom    = exp(sum(log(v) for v in visible) / len(visible))   # geometric mean
worst   = min(visible)
score   = ((geom + worst) / 2) / 10                          # average of geom_mean and min, normalised
```

This formula has the property that **either** axis being low pulls
the score down, and the lower (weaker) axis pulls harder. If one
axis is `null` (e.g. both hands occluded), use the visible axis
alone: `score = visible_axis / 10`. If `hands` = 0 or `body` = 0
(catastrophic on one axis), `score` = 0.

Compute `score` numerically with 4 decimals — do not round it to an
integer.

## Output format (strict)

Output ONLY this JSON object, no markdown fences, no prose around it:

```json
{
  "axes": {
    "hands": <int 0..10 or null>,
    "body": <int 0..10>
  },
  "defects": ["<short string>", ...],
  "score": <float 0..1>
}
```

- `hands`: integer 0–10 from the checklist, or `null` if every visible
  hand is fully occluded.
- `body`: integer 0–10 from the body scoring guide. Body is almost
  always at least partly visible, so this should rarely be `null`.
- `defects`: 0–6 short strings. Each names the **specific axis**
  ("hands", "body") and the **specific defect** ("right hand: 6+
  fingertips visible", "body: right shoulder dislocated"). When you
  give `hands` a score ≥ 8, include a "fingertip count: N,
  gesture-consistent" note as evidence you ran the checklist.
- `score`: float in `[0, 1]`, computed by the aggregation formula
  above. This is what the RL trainer uses as the per-image reward.

Examples:

```json
{
  "axes": {"hands": 4, "body": 7},
  "defects": ["right hand: 6+ fingertips, irregular knuckle alignment", "body: shoulder/hip rotation mismatched in squat"],
  "score": 0.4646
}
```

```json
{
  "axes": {"hands": null, "body": 9},
  "defects": ["both hands occluded (gauntlets)", "body: dynamic running pose clean"],
  "score": 0.9
}
```

```json
{
  "axes": {"hands": 9, "body": 9},
  "defects": ["fingertip count: 5+5, gesture-consistent", "body: clean standing pose"],
  "score": 0.9
}
```

```json
{
  "axes": {"hands": 6, "body": 8},
  "defects": ["left hand: middle finger fused with ring finger", "fingertip count: right=5 left=4-fused, gesture-inconsistent"],
  "score": 0.6464
}
```
