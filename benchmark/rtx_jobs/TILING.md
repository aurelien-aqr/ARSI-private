# Tiling large regions before judging — 39T sweep (2026-08-17/18)

## Why

On the 39T arm the pipeline misses half the instances. The diagnosis (from the
first full 50-case run) is that the object *is* inside a proposed region but
occupies 0.4–4.2 % of it, so the judge is shown a crop where a bottle is a dozen
pixels. Two earlier hypotheses were tested and rejected first:

- **region merging is not the cause** — MERGE_REGIONS on/off over the 39T cases
  gives 19/26 localized and 5/26 strict either way, and the biggest box is
  identical at 911,360 px. The mega-blob is born in the diff, not in the merge;
- **the blob-vs-bbox area gate is real but marginal** — `MAX_AREA=400000` gates
  the connected component's pixel count while the crop shown to the judge is the
  bounding box, so a sprawling thin blob passes. It affects 7 of 242 candidates
  on 39T (2.9 %) and 2 of 485 on 1762. Worth fixing for hygiene; it explains
  nothing.

## Method

Every region whose bbox exceeds `TRIGGER = 40,000 px` is cut into overlapping
tiles (step = 0.75 × tile) and each tile is judged separately, with
`margin = 20, context = 0.25` instead of the shipped `40 / 0.75` — the shipped
padding would re-shrink the object inside its own tile, which is the whole
point. Every arm applies the pipeline's own post-filters (`is_non_anomaly`,
`is_implausible`, `dedupe_regions`), so the difference between arms is the
tiling alone. Judge: `haervwe/GLM-4.6V-Flash-9B:latest`, conservative prompt.

**Control**: the `baseline` arm is the shipped behaviour re-run through the same
script. It reproduces the official run's object recall (0.500) and strict recall
(0.192) exactly. It does NOT reproduce that run's frame recall (0.643 here vs
0.714) or region precision (0.478 vs 0.542), because this script simplifies FP
attribution and the frame rule. **Cross-arm comparisons below are valid; these
precision numbers must not be quoted against `benchmark/runs/`.**

## Result — 21 cases, 26 instances

| arm | calls | object recall | strict IoU | region precision | frame recall | specificity |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 266 | 0.500 | 0.192 | 0.478 | 0.643 | **1.000** |
| tile120 | 1635 | 0.577 | **0.500** | 0.268 | 0.786 | **1.000** |
| **tile160** | 1034 | **0.615** | 0.423 | 0.333 | **0.857** | **1.000** |
| tile240 | 577 | 0.577 | 0.192 | 0.429 | 0.786 | **1.000** |

**Specificity stays 1.000 in every arm.** No clean frame is ever flagged, even at
six times the calls: the extra boxes land on frames that already hold a real
anomaly, never on empty ones. Against the project's acceptance criterion — an
object counts as found when it is detected and boxed, even if mislabelled —
those two error types do not carry the same price.

`tile160` is the operating point: object recall 0.500 → 0.615, frame recall
0.643 → 0.857, and **strict IoU 0.192 → 0.423 (2.2×)** — the boxes now land on
the objects instead of vaguely containing them. Cost: 3.9× the calls (27 s per
frame on the 3080 Ti) and region precision 0.478 → 0.333.

The ordering is not monotone (tile120 has the best strict recall and the worst
recall and precision). On 26 instances those gaps are a handful of objects; the
fine ordering between 120 and 160 should not be over-read.

## Guard check on 1762 — tiling must NOT be global

Same script, same two arms, over the 29 original cases (47 instances):

| arm | calls | object recall | strict IoU | region precision | frame recall | specificity |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 559 | 0.809 | 0.702 | 0.521 | 1.000 | **1.000** |
| tile160 | 926 | 0.872 | 0.702 | 0.402 | 1.000 | **0.750** |

**Specificity falls from 1.000 to 0.750** — three of the twelve clean 1762 frames
now raise an alarm, which never happened on 39T. Tiling small crops out of the
faint residual regions on an empty frame gives the judge exactly the kind of
ambiguous patch it hallucinates on. Frame recall was already 1.000 here, so the
recall tiling buys (0.809 to 0.872) is worth little, and it is paid for in the
one metric that matters for deployment.

Caveat on this control: the 1762 baseline arm does NOT reproduce the official
object recall (0.809 here vs 0.894 in `benchmark/runs/`) — this script's
simplified FP attribution and matching diverge more on these cases than on 39T,
where recall and strict matched exactly. The **cross-arm** specificity drop is
measured inside one script with one rule, so it stands; the absolute numbers do
not transfer to the published report.

## Verdict

Tiling is a **per-camera option, not a default** — the same shape as the DINOv2
gate, which also failed to transfer between these two camera families. Enable it
on moving-tram footage (the 39T family), leave it off on 1762.

Confidence is limited by the negatives: 39T's specificity 1.000 under tiling
rests on 7 clean frames, and 1762's drop on 12. More clean footage per camera is
what would firm this up - and it is the same gap as the false-alarm rate, which
still rests on a single video.

## Still open

The 2 instances the localizer never proposes at all (one eaten by the person
veto at cam54_083517) - untouched by tiling, and a localizer problem.
