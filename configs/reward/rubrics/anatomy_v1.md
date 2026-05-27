# Anatomy rubric — 2-axis, compact

Score the **anatomy** of the character in this anime image on two
axes. Output JSON only (no prose).

## Principle: only score what you see

Visible-and-wrong → low. Occluded (hidden behind body, in pocket,
inside gauntlet, fingers folded in a fist) → skip — never penalise.
Visible-and-right → high.

## Axis 1: `hands` (0–10, or null if every visible hand is fully occluded)

For each visible hand, run this checklist:

1. **Count fingertips.** Does the count match the gesture? Open hand →
   exactly 5. V-sign / point / fist → fewer is expected. **≥6 visible
   fingertips on any hand = defect regardless of pose.**
2. **Finger geometry.** Each finger: proximal phalanx ≥ middle ≥
   distal. Knuckles bend toward the palm. Adjacent fingers do not
   visibly fuse into a mitten unless the gesture explicitly closes
   them. Thumb is on the radial side.
3. **Wrist attachment.** Hand connects to wrist with no gap, no kink.

Score = worst single hand. Calibration:

- **10** every check passes
- **8–9** all checks pass at low scale; one barely-visible nit
- **6–7** one clear defect (one fused pair, mild proportion)
- **4–5** multiple finger defects OR one severe (extra finger,
  mitten, thumb on wrong side) OR fingers cannot be reliably
  counted at the rendered scale (unverifiable = defect)
- **2–3** hand barely recognisable
- **0–1** no recognisable hand structure

**You may not score ≥ 7 without explicitly counting fingertips
in the defects list (e.g. "fingertip count: 5+5, ok").**

## Axis 2: `body` (0–10)

Everything except hands: head/neck/torso/arms/legs/feet/face.

Look for: limbs attached at right shoulder/hip positions, joints
bending the right direction, bilateral limbs roughly equal where both
are visible, shoulder line consistent with hip line in the pose,
proportions internally consistent (chibi is fine if consistent),
feet anchored to ankles with sensible toe direction, face has 2
eyes + 1 nose + 1 mouth.

- **10** every region clean after a careful look
- **9** clean at first look, one nitpick after scrutiny
- **8** one minor issue (slight asymmetry, off shoulder slope)
- **7** one clear defect (shoulder/hip mismatched, mild
  hyperextension, awkward limb-to-clothing transition)
- **6** multiple defects, or one clear dislocation
- **4–5** one severe (limb detached, joint backward, melted face)
- **0–3** body severely broken / catastrophic

**Anti-cluster:** "roughly OK with one specific issue" is **7**, not
8. Reserve 9–10 for cases where you have to hunt to find a flaw.

## Aggregation (you compute this yourself)

After scoring `hands` and `body`:

```
visible = [v for v in (hands, body) if v is not None]
geom    = exp(sum(log(v) for v in visible) / len(visible))
worst   = min(visible)
score   = ((geom + worst) / 2) / 10        # final, in [0, 1]
```

If one axis is `null`, use the single visible axis: `score = visible /
10`. If any axis is 0, `score` = 0. Output `score` as a float with 4
decimals.

## Output format (strict — JSON only, no fences, no prose)

```json
{
  "axes": {"hands": <int|null>, "body": <int>},
  "defects": ["axis: specific defect", ...],
  "score": <float 0..1>
}
```

`defects`: 0–4 short strings naming axis + specific issue. For
`hands` ≥ 7, include `"fingertip count: N+M, ok"` as evidence.

Example:

```json
{"axes": {"hands": 4, "body": 7}, "defects": ["right hand: 6+ fingertips, irregular knuckles", "body: shoulder/hip rotation mismatched in squat"], "score": 0.4646}
```
