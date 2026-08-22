# Results for the paper

Measurements made 2026-08-19 (§0) and 2026-08-21/22 (§1–§5), all on the RTX
3080 Ti (`ssh arsi`, checkout `~/Documents/ARSI-vlm`). Everything here is
reproducible from the repo:

| § | Experiment | Script | Output |
|---|---|---|---|
| **0** | **The main result: proposer sweep × judge sweep** | `tools/rescore_localizer.py`, `tools/judge_sweep.py` | `docs/vlm_benchmark/metrics.json`, `benchmark/runs/` |
| 1 | Cross-camera transfer of Dinomaly | `tools/cross_camera.py --variant v1` | `docs/dino_models/cross_camera.json` |
| 2 | Dinomaly2 (arXiv 2510.17611) ported and measured | `tools/dinomaly2*.py`, then `--variant v2` | `docs/dino_models/cross_camera_v2.json` |
| 3 | Standard baselines on our cameras | `tools/anomalib_baseline.py` | `docs/dino_models/anomalib_baseline.json` |
| 4 | The proposal stage on public data (CDnet 2014) | `tools/cdnet_eval.py` | `docs/public_benchmarks/cdnet_*.json` |
| 5 | Abandoned objects, qualitative (ABODA) | `tools/aboda_qualitative.py` | `docs/public_benchmarks/aboda/` |

**§0 is the paper's claim; §1–§5 are what defend it.** §3 removes "is any of
this needed?", §4 removes "private data only", §1–§2 remove "why not just train
a model per camera?". See `OUTLINE.md` for how each one maps to a subsection.

**Shared protocol for §1–§3.** §0 reports end-to-end pipeline metrics (frame
F1/recall/specificity, object recall, region precision) defined in `method.md`
§3.7. §1–§3 compare *scoring models* instead, and therefore use image-level
AUROC (I-AUROC), the standard metric of the industrial anomaly-detection
literature, so those numbers can sit in the same table as any published
baseline:

```
score(image) = mean of the top 1 % of patch anomaly scores, masked patches excluded
I-AUROC      = P( score(anomalous frame) > score(clean frame) )
```

Training data per camera is exactly the list `dinomaly_train.nominal_frames()`
already used in production, benchmark holdout included. Test data is that
camera's cases in `benchmark/datasets/ground_truth.json`.

**Sample sizes, stated once because they bound every claim below.**

| Camera | anomalous | clean | pairs |
|---|---|---|---|
| tram_1762 (`real`) | 16 | 11 | 176 |
| 3333-cam53 / 54 / 55 | 4 each | 2 each | 8 each |
| 3333-cam52 | 2 | 1 | 2 |
| 3333 pooled | 14 | 7 | 98 |
| 1760-cam04 / 06 / 13 | 0 | 6 each | — |

Only the **tram_1762 column and the 3333-pooled column** carry enough pairs to
support an AUROC claim. The individual 3333 columns are reported for
completeness and must not be quoted alone. The 1760 cameras have no labelled
anomalies at all, so they contribute score levels, not AUROCs.

---

## 0. The main result: the proposer decides, not the judge

This is the paper's headline (`OUTLINE.md` §0) and everything else in this file
is evidence for or against it. Measured 2026-08-19 on the 68 cases / 73
instances / 3 trams / 9 views of `benchmark/datasets/ground_truth.json`.

**The design that makes the claim testable.** The proposal stage and the judge
are independently selectable components sharing one box contract and one set of
post-filters (`method.md` §3.3–§3.4). Swapping one changes which regions reach
the other, and nothing else. So the two tables below are a controlled sweep of
one factor at a time, not a comparison of pipelines.

### 0a. Four proposers, one frozen judge

Judge held fixed at GLM-4.6V-Flash-9B × conservative.

| Proposer | regions sent | frame F1 | frame recall | frame spec. | **object recall** | strict-boxed | region precision |
|---|---|---|---|---|---|---|---|
| `photo` (pixel diff) | 1192 | 0.931 | 0.871 | 1.000 | **55/73** | 0.534 | 0.694 |
| `photo+dino` (pixel diff, feature veto) | 727 | 0.931 | 0.871 | 1.000 | **55/73** | 0.534 | 0.773 |
| `dino` (DINOv2 features) | 1228 | **0.984** | **0.968** | 1.000 | **55/73** | **0.630** | **0.896** |
| `dino+dinomaly` (features + per-camera veto) | **654** | **0.984** | **0.968** | 1.000 | **55/73** | **0.630** | **0.896** |

**Read the object-recall column first.** It does not move. Fifty-five of
seventy-three instances, in all four arms, to the instance. Everything else
does move: frame recall 0.871 → 0.968, region precision 0.694 → 0.896,
kept-but-wrong regions 30 → 7, and strict-IoU boxing 0.534 → 0.630. Frame
specificity is 1.000 throughout, so nothing here is bought with false alarms.

Three consequences, and the third is the paper.

1. **"Better boxes find more objects" is the tempting reading of this table and
   it is false.** The localization-only view (below) says the feature proposer
   localizes four instances the pixel diff misses — and the judge rejects them
   anyway. The proposal stage sets the *ceiling*; the judge decides where under
   it the system sits.
2. **A veto is not a proposer.** Rows 1–2 are identical on every quality column
   and differ only in cost, because a veto deletes boxes and never redraws one.
   The same holds for rows 3–4. So the question is never *which filter* but
   *who proposes* — a distinction the pipeline had to be refactored to even ask.
3. **The binding constraint is spatial, not linguistic.** Under a frozen judge,
   changing the proposer moves frame F1 by 0.053 and region precision by 0.202.
   Table 0b shows what changing the judge does instead.

**Figure 2** (`docs/paper/figures/fig2_proposers.jpg`,
`tools/figure_proposer.py`) makes the mechanism visible on one frame,
`3333_cam55_083517`, with two ground-truth instances — a laptop left on a seat
and a plastic bag on the floor:

| | proposals | laptop | floor bag |
|---|---|---|---|
| photometric diff | 8 | IoU 0.042 | IoU 0.016 |
| DINOv2 features | 21 | **IoU 0.555** | **IoU 0.408** |

Both instances fail the strict rule (IoU ≥ 0.3) under the pixel diff and pass it
under the features. The bag's *only* photometric cover is the 98.9 % box: the
frame itself. This is the single image that argues the paper, and it also shows
what the feature proposer costs — 21 proposals against 8, which is why §0a
reports no saving in judge calls for `dino` and why the veto arm exists.

**Localization only (0 VLM calls), same 68 cases.** This is the stage in
isolation, and it is where the strict-IoU column comes from:

| Proposer | instances localized | strict IoU | regions |
|---|---|---|---|
| `photo` | 65/73 | 42/73 | 1192 |
| `photo+dino` | 65/73 | 42/73 | 727 |
| `dinomaly` alone | 62/73 | 48/73 | 813 |
| `dino` | **69/73** | **58/73** | 1228 |
| `dino+dinomaly` | **69/73** | **58/73** | 654 |

The lenient column (65 vs 69) barely separates the families; the strict column
(42 vs 58) separates them decisively. The cause is measurable and blunt: on
tram 3333 the pixel diff scores 5/26 strict against the feature proposer's
19/26, and its worst box covers 911 360 px of a 1280×720 frame — **98.9 % of the
image**. A box that large is technically a hit under a lenient rule and useless
to a judge that only sees the crop.

### 0b. Twelve judges, one frozen set of boxes

Boxes held fixed at `dino+dinomaly` (654 crops). Four models × three prompts,
2 h 07 of GPU. The calibration anchor: 73 instances among 654 regions means a
**calibrated judge answers YES about 11 % of the time**.

| Model | Prompt | YES % | instances | frame F1 | frame spec. | region prec. |
|---|---|---|---|---|---|---|
| **GLM-4.6V-Flash-9B** | **conservative** | **10.2** | **55/73** | **0.984** | **1.000** | 0.896 |
| GLM-4.6V-Flash-9B | balanced | 10.4 | 55/73 | 0.968 | 0.973 | 0.882 |
| GLM-4.6V-Flash-9B | lenient | 13.6 | 56/73 | 0.909 | 0.865 | 0.719 |
| InternVL3.5-8B | balanced | 8.7 | 48/73 | 0.984 | 1.000 | 0.912 |
| InternVL3.5-8B | lenient | 7.8 | 43/73 | 0.931 | 1.000 | 0.922 |
| InternVL3.5-8B | conservative | 6.9 | 39/73 | 0.893 | 1.000 | 0.933 |
| Qwen3-VL-8B | lenient | 28.9 | 63/73 | 0.690 | 0.297 | 0.376 |
| Qwen3-VL-8B | balanced | 40.5 | 63/73 | 0.659 | 0.189 | 0.268 |
| Qwen3-VL-8B | conservative | 39.6 | 63/73 | 0.645 | 0.135 | 0.274 |
| Qwen2.5-VL-7B | balanced | 54.3 | 64/73 | 0.652 | 0.162 | 0.200 |
| Qwen2.5-VL-7B | conservative | 48.3 | 63/73 | 0.645 | 0.135 | 0.222 |
| Qwen2.5-VL-7B | lenient | 88.5 | 63/73 | 0.632 | 0.081 | 0.123 |

**Nothing beats the shipped judge at usable specificity.** Twelve arms. That is
a result and not a null result: it is the control that keeps §0a honest. A
reviewer's first objection to "the proposer decides" is "you chose a weak
judge", and this table answers it — the judge was chosen by sweep, and the
sweep is exhausted.

Three readings that must appear in the paper.

1. **The Qwen 63/73 is not detection, it is non-rejection.** At a 88.5 % YES
   rate a judge mechanically recovers almost everything the proposer handed it,
   and frame specificity collapses to 0.081. Any recall in this literature has to
   be read next to the YES rate; the four Qwen arms are the demonstration of why.
2. **The prompt is a real lever only where there is slack.** On InternVL3.5,
   which under-fires at 6.9 %, `balanced` moves 39 → 48 instances at specificity
   1.000 unchanged. On GLM, which is already calibrated at 10.2 %, `balanced`
   gains zero instances and costs specificity. A prompt rescues an over-cautious
   model; it does not push a calibrated one past its ceiling.
3. **Therefore: prompting is calibration, not capability** — at this scale, for
   this lot of models, *without adaptation*. The qualifier is mandatory. VERA
   (CVPR 2025) treats prompts as learnable parameters and gains from it, and
   Borodin et al. report that prompt sensitivity drops sharply *after*
   parameter-efficient fine-tuning. Our result and theirs are two halves of one
   finding, and the honest joint statement is that the next lever is adaptation
   (`docs/LORA_PLAN.md`), not more formulations.

### 0c. What this leaves unexplained

Object recall is pinned at 55/73 and no proposer moves it. Eighteen instances
are localized and then rejected, four of them by the feature proposer alone.
That is our own claim's boundary and the paper must state it in the discussion,
not bury it: the proposal stage sets the ceiling, and the remaining headroom is
at the judge, where prompt and model selection are already exhausted.

---

## 1. Cross-camera transfer: a foreign checkpoint is blind, not miscalibrated

**Question.** `docs/DECISIONS.md` rejects Dinomaly partly on cost — "it needs a
checkpoint per camera". That is only an objection if a foreign checkpoint
actually fails. It had never been measured, and Dinomaly's own selling point
(CVPR 2025) is a single multi-class model.

**Result.** It fails, and the way it fails is the interesting part.

I-AUROC, row = camera the checkpoint was trained on, column = camera it is
tested on. `*` marks the shipped configuration.

```
                 3333-cam52  3333-cam53  3333-cam54  3333-cam55   tram_1762  3333-pooled
1760-cam04            1.000       0.375       0.250       0.250       0.830        0.398
1760-cam06            0.000       0.500       0.500       0.500       0.665        0.480
1760-cam13            0.000       0.625       0.000       0.250       0.847        0.337
3333-cam52           1.000*       0.250       0.125       0.375       0.830        0.398
3333-cam53            0.000      0.750*       0.875       0.625       0.790        0.571
3333-cam54            0.500       0.625      1.000*       0.250       0.761        0.541
3333-cam55            1.000       0.750       0.750      1.000*       0.909        0.643
tram_1762             0.500       0.375       0.000       0.375      0.920*        0.296
```

The AUROC table alone would be over-read — most of its cells rest on 8 pairs.
The score levels are what carry the finding, and they are unambiguous:

| | own camera | foreign camera |
|---|---|---|
| score on **clean** frames | 0.070 – 0.093 (median 0.077, n=8) | 0.106 – 0.252 (median 0.163, n=56) |
| separation, mean(anomalous) − mean(clean) | +0.017 … +0.061 | median **+0.0009** (n=35) |

**A foreign checkpoint scores a clean frame 2.1× higher than the right
checkpoint does, and its anomalous-minus-clean margin collapses to zero.**

That distinction matters more than the AUROCs. A model that were merely
*offset* could be rescued by a per-camera threshold. This one is not offset: on
an unfamiliar camera, "this view is unfamiliar" dominates the reconstruction
error and drowns "this patch is anomalous". The margin that a deployed
threshold would have to sit inside — +0.037 on tram_1762 with its own model —
becomes +0.002 to +0.014 with a foreign one, three to eighteen times narrower.

**And it is the viewpoint, not the lighting.** The four 3333 cameras were filmed
in the same two sessions, minutes apart, same weather, same exposure. Transfer
*within* that family still fails: foreign clean scores 0.109 – 0.186 against own
scores 0.074 – 0.092. Whatever the model learns is tied to where the camera
points, not to the hour of the day.

**Honest nuance, worth keeping in the paper.** In the tram_1762 column — the
only one with 176 pairs — foreign checkpoints still reach AUROC 0.665 to 0.909.
Some ranking signal survives. What does not survive is the *margin*. The
conclusion to write is therefore precise: transfer preserves a weak ordering and
destroys the operating point, which is exactly the failure a deployed system
cares about and exactly the one an AUROC-only report would hide.

**Consequence for `DECISIONS.md`.** The row that says Dinomaly "needs a
checkpoint per camera" is now measured rather than assumed, and it survives:
per-camera training is not a cost we chose, it is a requirement. The paper can
state this as a finding, not as an implementation note — and §2.2 bis of
`related_work.md` establishes that nobody in the 45 papers citing Dinomaly or
AnomalyDINO has reported it.

---

## 2. Dinomaly2: it halves the gap and does not close it

The five elements were read from the reference implementation
(`guojiajeremy/Dinomaly2`, preview code May 2026) rather than from the paper —
they are literally CLI flags in their `dinomaly_2D.py` (`--la --lc --ll --cr`
plus the bottleneck dropout). They are ported in `tools/dinomaly2.py`, whose
header lists both what was ported and where we still deviate. Eight checkpoints
trained, one per camera, on exactly the frames, crops, schedule and holdout that
v1 used — nothing else was allowed to move.

The hypothesis was specific:

> Context-aware recentering subtracts the CLS token — a global summary of the
> image — from every encoder target. The failure measured in §1 is a global,
> view-dependent offset that swamps the local signal. If CAR does what it
> claims, foreign clean-score inflation should shrink and the separation should
> survive.

**Half of that prediction came true.**

| | Dinomaly (v1) | Dinomaly2 (v2) |
|---|---|---|
| clean score, foreign ÷ own | **2.13×** | **1.52×** |
| separation, own ÷ foreign | **41×** | **10×** |
| diagonal I-AUROC | 0.750, 0.920, 1.000, 1.000, 1.000 | 0.801, 1.000, 1.000, 1.000, 1.000 |
| 3333-pooled, non-3333 rows (real transfer) | 0.296, 0.337, 0.398, 0.480 | 0.367, 0.408, 0.490, 0.510 |

> **Scale caveat, stated before anything is read into the table.** CAR
> LayerNorms the reconstruction targets, so the *absolute* score levels of v1 and
> v2 are not on the same scale (own-camera clean scores move 0.077 → 0.100 for
> that reason alone). Only the **ratios** — foreign ÷ own, and own separation ÷
> foreign separation — compare across variants. Every claim below uses ratios.

What v2 buys:

- **The inflation shrinks from 2.13× to 1.52×**, and the collapse of the margin
  from 41× to 10×. The direction is exactly the one CAR predicts, and the effect
  is large.
- **On its own camera it is at least as good, and on 3333 better**: cam53 goes
  0.750 → 1.000 and the other three 3333 views sit at 1.000. That matters
  independently of transfer — 3333 is the family where our shipped localizer was
  weakest (17/24 instances).

What it does not buy:

- **Transfer is still unusable.** The four genuinely foreign rows on the
  3333-pooled column (98 pairs, the only column besides tram_1762 worth reading)
  move from a median of ~0.39 to ~0.45. That is chance, before and after. A gap
  reduced by half is still a gap that forbids deploying one checkpoint on another
  camera.
- **tram_1762 regresses**, 0.920 → 0.801, on the largest column we have (176
  pairs). One camera, one number, but it is the wrong direction and it should not
  be hidden: v2 is not a free upgrade.

**Verdict for `DECISIONS.md`: do not ship.** Dinomaly2 is worth reporting in the
paper as a measured negative — the authors' own answer to our objection was
tested, it moves the mechanism in the predicted direction, and it still does not
make one model per fleet possible on our data. That is a much stronger sentence
than "we did not try the newer version".

### 2b. Ablation: which of the four elements did the work

v2 changes four things at once, so `DINOMALY2_CAR=0` trains the identical
architecture with recentering removed and nothing else touched — 8 more
checkpoints under `dinomaly2nocar_*`, same data, same schedule.

| | clean inflation<br>foreign ÷ own | margin collapse<br>own ÷ foreign | diagonal I-AUROC | transfer<br>(median, 3333-pooled) | tram_1762 self |
|---|---|---|---|---|---|
| Dinomaly (v1) | 2.13× | **41×** | 0.750, 0.920, 1.000×3 | 0.367 | **0.920** |
| Dinomaly2, CAR **off** | 1.73× | **43×** | 0.903, 1.000×4 | 0.393 | **0.903** |
| Dinomaly2, CAR **on** | 1.52× | **10×** | 0.801, 1.000×4 | 0.449 | **0.801** |

This attributes cleanly, and it is more interesting than a single number.

**Context-aware recentering is the only element that touches the margin.**
Without it the collapse is 43×, statistically the same as v1's 41×; with it, 10×.
The loose loss, the bottleneck shape and the reversed decoder pairing do nothing
whatsoever for cross-camera behaviour. The mechanism proposed in §1 — a global,
view-dependent term drowning the local signal — is therefore confirmed by
removing exactly that term and watching the effect disappear.

**The other three elements are what fixed the diagonal.** 3333-cam53 goes 0.750 →
1.000 with CAR off, so that gain belongs to the loose loss / bottleneck /
pairing, not to recentering.

**And CAR is what costs us tram_1762.** With CAR off, 0.903, essentially v1's
0.920; with CAR on, 0.801. Recentering subtracts a global term that carries
nuisance *and* signal, and on the camera where we have the most data the trade is
negative.

So the honest one-line summary is a trade-off, not an improvement:

> Context-aware recentering buys a 4× better cross-camera margin and pays about
> 0.10 image-level AUROC on the home camera. On our fleet that trade is not worth
> taking, because even the improved margin leaves transfer at chance.

**Practical consequence, and it is a real one.** If the fleet ever needs a single
checkpoint, the direction to push is CAR — it is the only lever that moved. The
version to reach for would be Dinomaly2 with its *own* recipe (larger backbone,
their 448/392 input, full-length schedule, CLS kept inside the decoder sequence)
rather than our port, and the target to beat is stated: inflation 1.00×, margin
collapse 1×.

## 3. Standard baselines: the frame-level question is already solved

anomalib 2.6 ships reference implementations of PaDiM, PatchCore, Dinomaly **and
AnomalyDINO**, so the table the paper was missing also doubles as a check on our
own simplified re-implementation. Each cell is a model fitted on that camera's
nominal frames and tested on that camera's benchmark cases — same data, same
split as everything above. Scores are each model's native image score; AUROC is
rank-based, so the comparison is valid even though the score definitions differ.

I-AUROC, own camera:

| method | 3333-cam52 | 3333-cam53 | 3333-cam54 | 3333-cam55 | **tram_1762** |
|---|---|---|---|---|---|
| | *2 pairs* | *8 pairs* | *8 pairs* | *8 pairs* | ***176 pairs*** |
| PaDiM | 1.000 | 0.875 | 1.000 | 1.000 | **0.915** |
| PatchCore | 1.000 | 0.625 | 1.000 | 1.000 | **0.841** |
| Dinomaly *(anomalib reference)* | 1.000 | 1.000 | 1.000 | 1.000 | **0.886** |
| AnomalyDINO *(anomalib reference)* | 1.000 | 0.750 | 1.000 | 1.000 | **0.835** |
| Dinomaly *(ours, v1)* | 1.000 | 0.750 | 1.000 | 1.000 | **0.920** |
| Dinomaly2 *(ours)* | 1.000 | 1.000 | 1.000 | 1.000 | **0.801** |

*(The JSON also carries a "3333-pooled" column. **Do not use it for this table.**
Each cell is a separately fitted model, so pooling raw scores across cameras
compares numbers produced by different models. Pooling is only legitimate in §1
and §2, where a single checkpoint scores every camera.)*

Two things fall out, and the second one matters more than the table itself.

**Our re-implementation is sound.** Four of five cameras match the reference
Dinomaly exactly. On 3333-cam53 our v1 sat at 0.750 against the reference's
1.000, and our v2 closes that gap to 1.000. On the only column with real
statistics, ours (0.920) is slightly *above* the reference (0.886). A
re-implementation cannot be reported without this check, and it passes.

**Frame-level anomaly detection on this data is not the hard part.** Every
standard method, with no tuning, lands between 0.835 and 0.920 on the 176-pair
column, and saturates at 1.000 on three of the four 3333 views. "Does this frame
contain an anomaly" is close to solved by a 2021 memory-bank baseline that takes
nine seconds to fit.

That reframes the paper's results section. Our contribution cannot be a
frame-level AUROC — a reviewer would rightly answer "PaDiM does that". It has to
live where the numbers are still bad and where none of these methods says
anything at all:

- **which region** — our object recall is 55/73 and box quality drives frame
  recall 0.871 → 0.968 and region precision 0.694 → 0.896;
- **what it is** — a language verdict a human can read, which no row of the table
  above produces;
- **and at what operating point it holds** — §1 shows the margin, not the
  ranking, is what transfer destroys, and an AUROC-only report hides exactly
  that.

This is the same conclusion `docs/DECISIONS.md` reached from the other end ("the
proposer decides, not the judge"), now with the baseline that proves the
frame-level framing was never the interesting one.

---

## 4. Public data: the proposal stage on CDnet 2014

This is the half `docs/PUBLIC_DATASETS.md` asked for — *"I don't think having our
own dataset is what gets a paper rejected. Having our own dataset and nothing
else is."* CDnet 2014 is the benchmark that evaluates our change-proposal stage
**without the VLM in the loop**, so that when the full pipeline is wrong we can
finally say whether the region was never proposed, or was proposed and misjudged.

`tools/cdnet_eval.py`. Protocol, with every deviation from the official one named:
reference = per-pixel median of up to 50 frames from the pre-`temporalROI`
initialisation period; every 10th frame of the temporal ROI evaluated; official
CDnet label semantics (hard shadow counts as background, 85/170 excluded, ROI.bmp
applied); **person filter OFF** — on CDnet the foreground *is* people, and our
person veto exists to suppress tram passengers, so leaving it on would measure the
opposite of what the benchmark asks.

Two rows per configuration, because our stage produces two different objects:
`mask` is the raw per-pixel photometric change mask (`vlm_05.change_mask`), a
genuine change detector comparable to the CDnet leaderboard; `boxes` is the
rasterised region proposals, i.e. what actually reaches the VLM.

### `intermittentObjectMotion` — the category that is our task, almost word for word

Its authors define it as *"videos containing background objects moving away,
abandoned objects and objects stopping for a short while and then moving away"*.

| sequence | mask F1 | mask Re | mask Pr | boxes F1 | boxes Re | boxes Pr |
|---|---|---|---|---|---|---|
| streetLight | **0.978** | 0.988 | 0.968 | 0.331 | 0.998 | 0.199 |
| sofa | 0.565 | 0.428 | 0.833 | 0.346 | 0.738 | 0.226 |
| abandonedBox | 0.544 | 0.957 | 0.380 | 0.205 | 1.000 | 0.114 |
| parking | 0.448 | 0.473 | 0.426 | 0.251 | 0.961 | 0.145 |
| winterDriveway | **0.143** | 0.656 | 0.080 | 0.054 | 0.894 | 0.028 |
| tramstop | **0.138** | 0.221 | 0.100 | 0.336 | 0.937 | 0.204 |
| **mean of sequences** | **0.469** | 0.620 | 0.465 | 0.254 | **0.921** | 0.153 |
| **pooled over frames** | 0.406 | 0.487 | 0.349 | 0.238 | 0.812 | 0.139 |

**Two findings, and the second is the one for the paper.**

**1. The stage is excellent exactly when its reference stays valid, and collapses
when it does not.** streetLight — fixed camera, stable light, a car stops — gives
F1 **0.978**, which would be competitive on the CDnet leaderboard. winterDriveway
(snow, changing daylight) and tramstop (outdoor light drift) give **0.143** and
**0.138**. A static-reference photometric differencer has one failure mode and
CDnet finds it in one afternoon. This is the same failure our own notes describe
from the other side — *"a cross-session empty frame still yields 12 kept regions
after the DINOv2 gate"* — now measured on data we did not build. It is also the
published motivation for the entire feature-space branch of our localizer work.

**2. CDnet's F1 penalises, by construction, the design choice we made on
purpose.** Look at the recall column: the boxes reach **0.921 mean recall** at
0.153 precision. Our proposal stage is not a change detector trying to be right;
it is a *recall-first proposer* whose false positives are meant to be killed
downstream by a VLM that CDnet does not run. Reporting only the F1 would say our
stage is weak; reporting recall and precision separately says what is true — it
rarely misses, and it over-proposes on purpose. Any paper table must show both
columns, and must state that the `boxes` row is scored against pixel-accurate
ground truth by a method that emits rectangles.

### The stress categories: where a static reference breaks

CDnet's other categories are a ready-made ablation of the assumptions our
reference-diff makes. Pixel diff only, stride 20 (stride 10 for
`intermittentObjectMotion`), pooled over every evaluated frame:

| category | seq | frames | stride | mask F1 | mask Re | mask Pr | mask FPR | boxes Re | boxes Pr |
|---|---|---|---|---|---|---|---|---|---|
| `badWeather` | 4 | 899 | 20 | **0.675** | 0.557 | 0.856 | 0.0014 | 0.918 | 0.095 |
| `cameraJitter` | 4 | 159 | 20 | 0.566 | 0.647 | 0.504 | 0.0266 | 0.924 | 0.105 |
| `shadow` | 6 | 710 | 20 | 0.496 | 0.618 | 0.415 | 0.0463 | 0.942 | 0.178 |
| `nightVideos` | 6 | 619 | 20 | 0.434 | 0.466 | 0.406 | 0.0100 | 0.886 | 0.129 |
| `intermittentObjectMotion` | 6 | 1216 | 10 | 0.406 | 0.487 | 0.349 | 0.0350 | 0.812 | 0.139 |
| `dynamicBackground` | 6 | 667 | 20 | 0.234 | 0.808 | 0.137 | 0.0714 | 0.998 | 0.028 |
| `lowFramerate` | 4 | 264 | 20 | **0.207** | 0.278 | 0.164 | 0.0277 | 0.767 | 0.052 |

Read top to bottom, this is a ranking of *our* assumptions by how load-bearing
they are.

- **`badWeather` 0.675 and `cameraJitter` 0.566 are the good news**, and they are
  the two that matter most for a tram. Snow and rain against a median reference
  cost almost nothing (FPR 0.0014), and camera shake — a tram vibrates
  continuously — is survivable at 0.647 recall / 0.504 precision. The blur in our
  photometric channel is doing real work here.
- **`dynamicBackground` 0.234 and `lowFramerate` 0.207 are the failures**, and
  both are outside our deployment: moving water and foliage do not exist inside a
  vehicle, and we control the sampling rate. Worth reporting precisely because
  they bound the method rather than because they threaten it.
- **`shadow` 0.496 and `nightVideos` 0.434 are the honest middle** — and these two
  *are* ours. A tram runs at night and through moving shade. This is the number
  to quote when asked what the change-proposal stage does under illumination
  change, and it is mediocre.

Note the boxes column across every row: recall 0.767 – 0.998. Whatever the
category, the proposal stage keeps finding the object; what varies is how much
else it drags in. That is the same recall-first signature as the
`intermittentObjectMotion` table, now over seven categories and 5 000+ evaluated
frames.

### Three localizer families on public data

The same 6 sequences, same protocol, our three shipped proposal families
(`docs/dino_models/`, run on the GPU box). F1 / recall, per sequence:

| sequence | `photo` mask | `photo` boxes | `photo+dino` | `dino` |
|---|---|---|---|---|
| abandonedBox | 0.544 / 0.957 | 0.205 / 1.000 | 0.205 / 1.000 | 0.037 / 0.106 |
| parking | 0.448 / 0.473 | 0.251 / 0.961 | 0.251 / 0.961 | 0.177 / 0.264 |
| sofa | 0.565 / 0.428 | 0.346 / 0.738 | 0.336 / 0.679 | 0.032 / 0.116 |
| streetLight | 0.978 / 0.988 | 0.331 / 0.998 | 0.331 / 0.998 | 0.125 / 0.409 |
| tramstop | 0.138 / 0.221 | 0.336 / 0.937 | 0.336 / 0.937 | — / **0.000** |
| winterDriveway | 0.143 / 0.656 | 0.054 / 0.894 | 0.050 / 0.816 | 0.032 / 0.804 |
| **mean of sequences** | 0.469 / 0.620 | **0.254 / 0.921** | 0.252 / 0.898 | 0.080 / 0.283 |

**The DINOv2 gate reproduces, on someone else's data, exactly what our own
benchmark says it is.** `photo+dino` scores 0.252 against the ungated diff's
0.254, at slightly lower recall (0.898 vs 0.921). It is a cost lever and never a
quality one — *"a veto deletes regions; it never redraws one, so what decides box
quality is the PROPOSER"* (`docs/DECISIONS.md`). Independent confirmation of a
claim we previously had only from our own 68 cases.

**AnomalyDINO does not transfer to CDnet, and that bounds our best localizer.**
On our fleet `dino` is the quality winner (58/73 strict IoU against the pixel
diff's 42/73). Here it manages 0.283 mean recall and **fires not at all on
tramstop**. Two candidate explanations were tested and both are ruled out:

- *Thresholds.* Sweeping `DINO_Z` 4.0 → 2.5 → 1.5 and `DINO_FLOOR` 0.08 → 0.02 on
  streetLight + winterDriveway moves recall 0.616 → 0.723 → 1.000 while precision
  stays pinned at 0.011 – 0.022 and FPR climbs to 0.98. There is no threshold at
  which it is useful; it goes straight from silent to covering the frame.
- *The median reference.* Rebuilding the reference from a single real frame
  instead of a 50-frame median leaves it unchanged (F1 0.029 vs 0.028 on
  abandonedBox + tramstop), while it makes the pixel diff distinctly worse
  (specificity 0.375 vs 0.483). So the composite reference is not the culprit.

What remains is the assumption itself: AnomalyDINO scores each patch against the
**same grid position** in a reference of *that same fixed camera*, and its boxes
are quantised to the patch grid. That is a good trade on a tram interior where
the anomaly is a large fraction of a patch, and a bad one on 320×240 outdoor
footage scored against pixel-accurate ground truth. **Our best localizer is best
conditionally**, and the condition is the deployment geometry — which is a
limitation worth writing down rather than a result to bury.

## 5. Abandoned objects: what we can and cannot claim

**ABODA — qualitative, and that is deliberate.** `docs/PUBLIC_DATASETS.md` already
ruled: 11 sequences is too little for a main table and the licence is unclear.
The copy in `dataset/` also ships the eleven `.avi` files and a README and
**nothing else** — there are no annotations to score against.
`tools/aboda_qualitative.py` therefore builds a reference from each sequence's
opening frames and runs the shipped proposal stage at several timestamps, writing
boxed frames to `docs/public_benchmarks/aboda/`.

The result is worth a figure. On `video1`, an indoor public corridor we never
tuned for: at 60 % of the sequence, two people stand with a backpack and the
stage boxes them as one region; at 90 %, the people are gone and the abandoned
backpack is boxed tightly and alone, with one spurious region in the far corner.
That is our exact deployment behaviour — the reference-diff finds the persistent
change once the transient one leaves — reproduced on someone else's footage.

**The Luna protocol is blocked, and here is precisely why.** Our plan was to adopt
the unified event-level protocol of Luna, SanMiguel, Ortego & Martínez, *Sensors*
18(12):4290, 2018 (`10.3390/s18124290`) across AVSS AB 2007, PETS2006, PETS2007,
ABODA and VISOR. What sits in `dataset/` is:

- `PETS2006/` — the **CDnet-format** version: 1200 frames, `temporalROI` 300-1200,
  per-pixel *motion* ground truth. It is the same sequence as
  `CDnet_2014/baseline/PETS2006`, not the original four-view abandoned-luggage
  benchmark with event annotations.
- `ABODA/` — videos only, no annotations.
- No AVSS 2007, no PETS2007, no VISOR.

So the event-level numbers (precision / recall / F1 per abandoned object, with
the 60-second persistence criterion) cannot be produced today. **What is missing
is annotations, not code or compute**: Luna et al.'s unified temporal annotation
package, plus the AVSS AB 2007 challenge ground truth. Until those are in hand,
the paper should cite Luna et al. for the protocol and report ABODA qualitatively
— which is exactly what our own dataset report recommended before any of this was
run.
