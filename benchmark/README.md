# vlm_05 anomaly-detection benchmark

A reproducible benchmark for `vlm_05_reference_diff.py` (reference-diff + VLM
anomaly detection in a fixed-camera tram).

## What it measures

1. **Frame-level binary detection** (primary): each labelled image is diffed
   against its clean reference; the VLM classifies every candidate region in
   `filter` mode; the frame is **flagged** if ≥ 1 region survives.
   Reported: accuracy, precision, recall, specificity, F1, confusion matrix.
2. **Object-level detection**: every ground-truth instance (typed box) counts
   as detected if any kept region overlaps it (lenient rule: IoU > 0.1 or
   centre containment; a stricter IoU ≥ 0.3 recall is reported alongside).
   Kept regions that match no instance are **false-positive regions**.
   Reported: instance recall (overall / per type / per source), region
   precision, FP-region count.

The benchmark always scores the **current configuration** of
`vlm_05_reference_diff.py` (model, prompt, localizer channels, person filter,
post-filters) — it imports the script and calls the same `localize()` +
`classify_with_vlm()` the live script uses.

## Where things live (reorganised 2026-08-17)

```
benchmark/
├── datasets/           the labelled protocols — the ground truth
│   ├── tram1762.json     29 cases, one camera, July
│   └── 39T.json          21 cases, four cameras, 2026-08-11
├── runs/               one directory per run (see "Runs" below); gitignored
│   └── cli-*/            what run_benchmark.py / rescore_gate.py write
├── archive/            every report published before the move, unchanged
│   ├── report*.md  results*.json
│   └── annotated/       the rendered overlays of the 2026-07/08 runs
├── cache.json          per-region VLM verdicts (image, reference, box, model, prompt)
├── zones_tram_1762.json  Task-3 zones-of-interest (used by bench_grid.py)
├── run_benchmark.py    the frozen CLI: tram1762, whatever vlm_05 is configured with
└── eval_localization.py  localization-only, any dataset, any diff variant
```

The datasets are read through `arsi_core/benchmarks.py`, which is also what the
Studio, the CLI and `tools/export_lora_dataset.py` use — one loader, one set of
scoring rules. It resolves a dataset by **id** (`tram1762`, `39T`) or by path,
and still finds the pre-2026-08-17 `ground_truth*.json` if a copy-back from the
GPU box recreates them.

## The datasets

- `datasets/tram1762.json` — **29 cases: 17 anomalous (45 typed instance boxes),
  12 clean.** Each case: inspection image,
  reference key, `has_anomaly`, `types`, `source` (real / gpt / variant /
  self), instance boxes in reference pixel space, and a human note.
  - Anomalous: 6 real CCTV frames (forgotten objects, one with a real seated
    person) + 10 AI-inpainted frames on the real scene (objects, graffiti,
    damage, litter, one crowd) + 1 second synthetic scene ("variant").
  - Clean: reference-vs-self (×2), a clean AI frame, 4 same-session empty
    frames, and **5 cross-session empty frames** (v2/v3/v4 vs the v1
    reference — the deployment-realistic negatives: different exposure,
    onboard-display content changes, one walking person).
- `datasets/39T.json` — the second protocol, 21 cases / 24 instances, four
  cameras (see "Second protocol" below). **Labelled by Claude, not yet reviewed
  by a human** — correct it in the Studio, which counts how many cases a human
  has confirmed.

## How to run

**From ARSI Studio → Benchmark** (the normal way since 2026-08-17): pick the
dataset, the mode, and — this is what the CLI cannot do — the
localizer × script × judge model × prompt. Two modes:

- **full**, with the judge: every case goes through the ordinary job queue, so it
  gets live progress, cancel, the shared verdict cache, and the run is openable
  in the Results screen like any other. Scored at frame and object level.
- **localization only**: no VLM, no Ollama, ~0.4 s per case. Scores what the
  region proposer found — the ceiling on end-to-end recall, and the number to
  tune thresholds on.

Each run writes `benchmark/runs/<run_id>/` with `run.json`, `score.json`,
`report.md` and **`dataset_snapshot.json`** — the labels exactly as they were
when it ran. The ground truth is editable now, so a run keeps reporting what it
measured and is flagged `stale GT` once the dataset moves on, instead of silently
comparing against corrected boxes.

The CLI still works and is unchanged:

```bash
# from the repository root, with the venv active and `ollama serve` running:
python benchmark/run_benchmark.py                        # tram1762, whatever
                                                         # vlm_05 is configured with
                                                         # -> benchmark/runs/cli-latest/
python benchmark/eval_localization.py                    # localizer-only (no VLM)
python benchmark/eval_localization.py --dataset 39T --variants shipped
```

`run_benchmark.py` deliberately takes no arguments: `tools/rescore_gate.py`
parses its own argv and then calls `main()`, so an argparse there would break the
A/B tooling. Anything you would want a flag for is a control in the Studio.

Cases are processed cheapest-first, so the negatives and the confusion matrix
populate first. If interrupted, **re-run the same command** — cached regions
are skipped. Changing the model or prompt invalidates the cache automatically
(the key includes both); changing localizer thresholds only re-evaluates the
regions whose boxes changed.

Measured cost per FRESH VLM call (side-by-side crop): **~2–4 min on CPU**
(this laptop, 8 cores, shared load), ~1–2 s expected on the target RTX
3080 Ti. A full fresh run is ~300–450 calls ≈ **12–24 h CPU / 10–20 min
GPU** — fresh full runs are GPU work; CPU is fine for cache-only re-scoring
(minutes) and for the localizer-only eval (seconds).

## Two scorers, and why you can trust the new one (2026-08-17)

There are now two implementations of these metrics: `run_benchmark.metrics` (the
CLI, frozen — every published number came from it) and
`arsi_core/benchmarks.score_full` (the Studio). Two implementations of one
definition drift, so:

- the matching rules are **copied** from `vlm_05` (`_iou`, `_boxes_overlap`) and
  `tests/test_benchmarks.py` asserts they still agree with that module;
- the new aggregation is fed the per-case rows of `archive/results.json` and must
  reproduce the totals of `archive/report.md`;
- and the whole Studio path was run against the published reports:

| Studio run (GLM-4.6V-Flash-9B × conservative) | result | matches |
|---|---|---|
| tram1762 · full · `photo` | 17/0/12/0 · 40/45 · strict 33/45 · 20 FP of 74 · precision 0.730 | `archive/report_gateshipped.md` |
| tram1762 · full · `photo+dino` | same matrix · 40/45 · 33/45 · 12 FP of 65 · precision 0.815 · 243 regions | `archive/report_gate0.08.md` |
| tram1762 · localize · `photo` | 45/45 · strict 37/45 · 559 regions | this file, above |
| 39T · localize · `photo` | 17/24 · strict 5/24 · 100 regions on 7 clean frames | this file, below |

Both full runs cost **0 fresh VLM calls** out of 559 and 243 regions — the score
counts them, so that is a measurement, not a claim. 13 s and 59 s on this laptop.

## Localizer (multi-channel, since 2026-07-12)

Measured motivation and design live in the USER CONFIG comments of
`vlm_05_reference_diff.py`. Summary: base photometric diff at thr 40
(untouched, proven) + a low-threshold channel (thr 30, ≤ 8 additions/frame,
recovers low-contrast floor bottles) + an added-edge-energy channel
(≤ 4 additions/frame, recovers the ZORK faint tag: 12× above noise in edge
domain) + a YOLOv8n person veto (IoA ≥ 0.6, loses zero GT instances) + a
salience-ranked MAX_REGIONS cap on the base channel only.
**Localization recall 41/45 → 45/45** (the XRP tag of gpt_03 was long scored
as a miss because its GT box was misplaced onto the ventilation grille next to
it — fixed 2026-07-12; the channels box the real tag and every judge names it).

## GPU results (2026-07-12/13, 29 cases, multi-channel localizer, corrected GT)

| judge × prompt | frame F1 | frame recall | specificity | object recall | region precision | s/call |
|---|---|---|---|---|---|---|
| qwen3-vl:8b × conservative | 0.829 | 1.000 | 0.417 | 0.978 | 0.355 | 1.1 |
| qwen3-vl:8b × lenient      | 0.872 | 1.000 | 0.583 | 0.978 | 0.534 | 1.1 |
| qwen3.5:9b × conservative  | 0.919 | 1.000 | 0.750 | 0.978 | 0.549 | 0.7 |
| qwen3.5:9b × lenient       | 0.872 | 1.000 | 0.583 | 0.978 | 0.496 | 0.6 |
| **GLM-4.6V-Flash-9B × conservative** | **1.000** | **1.000** | **1.000** | 0.889¹ | 0.663⁴ | 0.7 |
| InternVL3_5:8b × conservative | 0.970 | 0.941² | 1.000 | 0.733² | 0.759 | 0.8 |
| minicpm-v4.6 × conservative | disqualified³ | | | | | |

¹ GLM's 5 missed instances are all type `object` inside real multi-object frames
that it flags anyway — frame-level detection stays perfect. **Corrected
2026-07-30: they are NOT all small items.** The miss in `real_f0205` is a
129,648 px instance, so "too small to see" does not explain them; see the merge
A/B section below.

⁴ Measured with `MERGE_REGIONS = False`. The shipped configuration merges, which
raises this to **0.730** at identical recall — see below.

## Merge A/B — ANSWERED 2026-07-30 (GLM × conservative, 29 cases)

Question: `localize()` merges neighbouring regions before the judge
(`MERGE_REGIONS`, gap 24, min fill 0.50). Does that recover the 5 missed
instances? Hypothesis was that the judge rejected fragments it could not name.

| | merge OFF | merge ON |
|---|---|---|
| frame F1 | 1.000 | 1.000 |
| instance recall | 40/45 = 0.889 | 40/45 = 0.889 |
| strict IoU>=0.3 | 33/45 = 0.733 | 33/45 = 0.733 |
| kept boxes | 86 | 74 |
| FP boxes | 29 | 20 |
| region precision | 0.663 | **0.730** |

**Answer: no.** Recall did not move at either IoU threshold, and per-type recall
is identical (object 28/33, graffiti 6/6, damage 4/4, litter 2/2).

**Scope this claim honestly.** The merge-ON run cost only **68 fresh calls**, not
the ~559 estimated: 651 pre-merge regions become 559, but 491 of those keep
byte-identical coordinates and came back from cache. So the shipped merge
re-judged ~14% of regions and flipped none of the 5 misses. That is conclusive
for *this merge at these settings* — it does not prove fragmentation is
irrelevant in general. A more aggressive merge cannot settle it either: fill 0.25
was already tested and rejected because it chains neighbours into a full-frame
box (strict IoU collapses 37/45 → 22/45).

**The merge stays shipped anyway**, per the pre-registered rule: region precision
rose above 0.663 with recall intact. Per-case FP gains: gpt_07 4→1, gpt_02 4→2,
real_f0053 3→2, real_f0100 1→0, real_f0112 1→0, gpt_11 6→5.

Trust signal: the merge-OFF arm reproduced the published `archive/report.md` numbers
exactly (0.889 / 0.663 / 29 FP of 86), so this was a controlled A/B.

**Where the 5 FN really are.** All 5 on `real` footage, all type `object`
(graffiti/damage/litter are perfect). Localization has all 45. Not a crowding or
`MAX_REGIONS` effect either: `real_f0112` scores 4/4 with **77** raw regions, the
busiest frame in the set, while `real_f0037` misses one with only 20. Next probe:
diff the crop pairs GLM rejects against the ones it accepts on the same scene.

Reports: `archive/report_merge_on.md`, `archive/report_merge_off.md`.

## DINOv2 feature gate — ANSWERED 2026-08-12 (GLM × conservative, 29 cases)

The anomalib family (PatchCore / AnomalyDINO / Dinomaly / EfficientAD) had never
been looked at in this project. `tools/dino_localizer.py` implements the
AnomalyDINO idea (arXiv 2405.14529, WACV 2025) specialised for a fixed camera:
DINOv2 ViT-S/14-reg patch features, cosine distance to the nominal reference
within ±1 patch **at the same grid position**, robust-z **and** an absolute floor.

**Replacing the photometric diff is a bad trade.** As a standalone localizer it
halves the cross-session noise, which is exactly what it was expected to do —
but its boxes are quantised to the 24 px patch grid, so strict IoU drops:

| clean-frame candidate regions | shipped | dino z=6 |
|---|---|---|
| same session as the reference (v1_f0151/0181/0211/0241) | 34 | 84 |
| cross-session (v3_f0001, v4_f0004/0016/0022) | **105** | **52** |
| frame containing a person (v2_f0001) | 14 | 7 |

(the same-session blow-up was a normalisation artefact: on a near-identical pair
the MAD collapses and the z amplifies feature jitter — hence `ABS_FLOOR`. Even
fixed, the best standalone setting reaches 45/45 lenient at 32/45 strict vs
shipped's 37/45.)

**Using it as a gate over the shipped boxes is the good trade.** Keep vlm_05's
regions, drop the ones with no feature support (`localize_gated`, max patch
distance in the box). The survivors keep byte-identical bboxes, so every verdict
was already in `cache.json` → the whole A/B ran on this laptop at **0 fresh VLM
calls** (`tools/rescore_gate.py`, 1.3 min per arm):

| | shipped | **gate 0.08** | gate 0.12 | gate 0.15 |
|---|---|---|---|---|
| frame level (TP/FP/TN/FN) | 17/0/12/0 | 17/0/12/0 | 17/0/12/0 | 17/0/12/0 |
| instance recall | 0.889 | **0.889** | 0.889 | 0.844 |
| strict IoU>=0.3 | 0.733 | **0.733** | 0.733 | 0.689 |
| FP boxes / kept | 20 / 74 | **12 / 65** | 9 / 62 | 8 / 59 |
| region precision | 0.730 | **0.815** | 0.855 | 0.864 |
| regions sent to the VLM | 559 | **243** | 165 | 129 |
| instances still localized | 45/45 | **45/45** | 44/45 | 42/45 |

`--gate -1` (filters nothing) reproduces `archive/report.md` exactly → controlled A/B.
Cost: +1.8 s/frame on this CPU, ~20 ms on the GPU, against −57 % VLM calls.

**0.08 is the pick, recall-first**, because it is the largest threshold at which
no GT instance loses its region (last row). From 0.10 up, the far phone of
`real_f0219` (~1 patch wide) is gated away; end-to-end recall does not move only
because GLM already rejected it, which makes it a loss waiting to surface under
another judge — the same trap that disqualified InternVL. Raising the grid does
not buy it back: at `DINO_INPUT_W=1400` (19 px/patch) gate 0.12 loses 3 instances
instead of 1 *and* keeps more regions, DINOv2 drifting from its training scale.

**Scope it honestly.** On a cross-session empty frame the gate still keeps 12 of
31 regions (patch distances 0.15–0.40: a different session really does move the
features), so it does *not* make cross-session negatives clean at proposal time —
GLM was already answering NO to those. The measured wins are the 70 % cost cut and
the FP-box halving on anomaly frames. It also does not touch the 5 FN: they are
judge rejections, and their patch distances are high (the four real_f0037 objects
score 0.34–0.52 against 0.03–0.10 for the dropped noise).

Not tried: Dinomaly (CVPR 2025) needs decoder training → GPU, and its selling
point is multi-class, which is not our problem; anomalib itself as a framework,
worth adopting only for the paper's baseline table (PatchCore/EfficientAD with
standard I-AUROC/PRO), since our empty frames are already a `train/good` set.

Reports: `archive/report_gateshipped.md`, `archive/report_gate0.06/0.08/0.1/0.12/0.15.md`.
Regression check without the VLM: `python benchmark/eval_localization.py
--variants shipped,gate0.12,dino4@0.10`.

## Dinomaly, reference-free — ANSWERED 2026-08-16 (localization only, 0 VLM calls)

The other half of the anomalib question. `photo` and `photo+dino` both compare
the frame to ONE reference, so they inherit that reference's session; Dinomaly
(CVPR 2025) needs no reference at inference — a frozen DINOv2 encoder plus a
trained noisy-bottleneck decoder flags what it cannot reconstruct.
`tools/dinomaly.py` + `dinomaly_train.py` + `dinomaly_localizer.py`, ViT-S/14-reg,
15.4 M trainable params, 135 nominal frames (v1+v3, one in two), 40 epochs, 56 min
on this laptop's CPU.

**The training set is the experiment**, so it is built defensively: every frame
within ±15 of a benchmark frame is dropped (consecutive frames are
near-duplicates and the 12 clean cases come from these very videos), v2 is
excluded (people and staged objects throughout) and **v4 is excluded even though
it is the only dark session** — 3 of its 22 frames are benchmark negatives, so
any use of it would leak. That makes v4 the one fully held-out session.

| | localized | strict IoU≥0.3 | regions to the VLM |
|---|---|---|---|
| shipped (`photo`) | 45/45 | **37/45** | 559 |
| `gate0.08` (`photo+dino`) | 45/45 | **37/45** | **243** |
| `dinomaly4@0.07` alone | 40/45 | 28/45 | 465 |
| `dgate0.05` (dinomaly as a veto) | 45/45 | **37/45** | 314 |
| `bgate0.08+0.05` (both vetoes) | 45/45 | **37/45** | **210** |

**As a localizer it loses**, and for the same reason `dino` alone lost: patch-grid
boxes. 28/45 strict against 37/45, while barely cutting the region count. Two of
its 5 misses are the `variant` scene, which this model was never trained on — in
domain it is 38/41, still below the pixel diff's 41/41.

**As a veto it works but is dominated.** `dgate0.05` reaches the shipped recall
AND the shipped box quality at 314 regions, but the reference-based gate already
does that at 243 — for no training, no per-camera checkpoint and no extra
forward pass. Stacking both vetoes is the only configuration that beats the
shipped gate (210, −14 %), and that is not worth 59 MB and ~1 h of training per
camera, times 26 cameras, for ~3 s/frame more CPU. **Not adopted**: it stays a
`tools/` experiment, not a registry entry in `arsi_core/localizers.py`.

**The diagnostic, and the fix that does not rescue it.** The heatmap says why the
standalone localizer proposes so much: reconstruction error is high on thin
structures — handrails, poster frames, the mask borders — on every frame,
nominal or not. Those are not anomalies, they are where this decoder is weak. So
`_baseline()` subtracts the mean error at each patch POSITION over the training
frames (`DINOMALY_BASELINE=1`), which asks the right question: is this patch
worse than usual *here*. It works as an economy measure and fails the recall
rule:

| with the position baseline | localized | strict | regions |
|---|---|---|---|
| `dinomaly4@0.04` alone | 35/45 | 26/45 | 123 |
| `dgate0.03` | 42/45 | 35/45 | 106 |
| `dgate0.01` | 44/45 | 37/45 | 247 |

Even at a 0.01 veto it still drops an instance: a position where the decoder is
weak has its real objects partially cancelled too. Recall-first therefore keeps
the baseline OFF by default, and the shipped-quality configuration remains
`bgate0.08+0.05` at 210 regions.

Two results worth keeping anyway:
- **The reference-free promise holds at the score level.** On v4 — the dark
  session it never saw — the model scores like v1 (per-frame max 0.10–0.12 vs
  0.09–0.12), so the cross-session exposure shift that floods the pixel diff does
  not move it. What it does not fix is that the *boxes* still come from the pixel
  diff in the gate configurations.
- **It fails safe on an unknown scene.** On the `variant` scene the gate vetoes
  ZERO regions (`-dgate=0`) instead of vetoing everything: an out-of-domain model
  degrades to "no gate", not to "no detections".

Reproduce: `python tools/dinomaly_train.py --camera tram_1762 --epochs 40` then
`python benchmark/eval_localization.py --variants shipped,gate0.08,dinomaly4@0.07,dgate0.05,bgate0.08+0.05`.
The checkpoint (59 MB) and the feature cache live in `weights/`, untracked.

Caveats stated rather than buried: 135 training frames from two daylight sessions
is thin, and the hard-mining loss was made tie-safe (top-k by count instead of a
quantile threshold) *after* this checkpoint was trained — the two select the same
points on continuous features, but a rerun for publication should use the current
code: `python tools/dinomaly_train.py --camera tram_1762 --epochs 40`.
² InternVL's frame F1 flatters it: it systematically says NO to real phones
(2/4 instances on every real multi-object frame) and misses real_f0205 whole.
³ minicpm-v4.6 ignores the reply format AND claims an object appeared on 198 of
199 crops from CLEAN frames ("bag visible on right" on empty seats) — unusable
as a crop judge via Ollama, though it answers sensibly on whole frames (see the
xlsx grid). Details in the note atop `archive/report_openbmb_minicpm-v4.6.md`.

Findings:
- **The prompt effect is model-dependent — measure, never assume.** The
  "conservative" prompt (name a specific object, if unsure say NO) *helps*
  qwen3.5 (F1 0.919 vs 0.872 lenient) but *backfires* on qwen3-vl — asked to
  name a specific new object, it actively finds one ("Blue seat cushion
  slightly shifted", "Black cable snake-like") and answers YES: region
  precision 0.355 vs 0.534 for the shorter lenient prompt.
- **GLM-4.6V-Flash-9B × conservative is the best frame-level judge**: 17/17
  anomalous frames flagged, 0 false alarms on 12 clean frames (including all
  5 cross-session negatives). Caveat: 29 cases is small — a perfect confusion
  matrix here is a strong signal, not a guarantee.
- **qwen3.5:9b × conservative is the best instance-inventory judge** (object
  recall 0.978 at the best specificity of the high-recall group) and the
  fastest. Its lack of grounding is irrelevant here — vlm_05 never asks for
  coordinates.
- Deployment recommendation: **GLM as the alarm judge**; qwen3.5 ×
  conservative when a per-object inventory matters more than false-alarm
  rate. A GLM-then-qwen3.5 two-stage (qwen3.5 only on GLM-flagged frames)
  would give both at +0.7 s/region on flagged frames only.
- The preserved `archive/report_lenient_qwen3vl.md` is the OLD baseline (24-case GT,
  single-channel localizer, CPU) — not comparable to the table above.

## Second protocol: 39T, 4 cameras — NEW 2026-08-16

Everything above was measured on ONE camera of tram 1762, filmed in July.
`benchmark/datasets/39T.json` (built by `tools/build_39T_benchmark.py`) is a
second, independent protocol: **21 cases, 24 instances, 4 viewpoints of tram
39T**, from the 2026-08-11 multi-camera capture. Every frame is masked with its
own camera's mask, each camera is its own reference (moment 08-55-37, verified
empty on all 15 interior cameras), and the other moments are **separate runs of
the line** — so a negative here means clean under a different sun, at a different
place on the track.

Labelled by Claude, not yet reviewed by a human: objects were found by eye on a
6-moment strip per camera — never by running our own localizer, which would have
made the ground truth agree with the system under test — then every instance was
confirmed by comparing the same crop in the reference and in the inspection
frame. That test rejected two candidates (a seat cover present in both frames, a
sunlit floor patch) and it is also what tells the tram's yellow validators from a
left object.

**First measurement, and it is a bad one** (`--dataset 39T --variants shipped`,
0 VLM calls):

| | 1762 (29 cases) | **39T (21 cases)** |
|---|---|---|
| instances localized | 45/45 | **17/24** |
| strict IoU ≥ 0.3 | 37/45 | **5/24** |
| regions on clean frames | 170 | 100 (7 frames) |
| biggest box | 734,400 px | **911,360 px** (63 % of the frame) |

The localizer that finds everything on 1762 misses 7 of 24 instances here, and
its boxes are mega-blobs: 5/24 strict against 37/45. Two mechanisms, both
visible in the per-case output:

- **the reference ages in seconds.** The `*_ref_t120_clean` cases are the same
  run as the reference, 60 s later, and they still produce 3–15 regions each.
  On 1762 the same-session negatives were nearly silent. A moving tram changes
  the light through the windows continuously, so "one clean reference per
  camera" is a much weaker assumption here than the July footage suggested;
- **the DINOv2 gate does not transfer.** `gate0.08`, tuned on 1762, removes
  166 → 153 regions here (8 %, against 57 % there) — its threshold is calibrated
  on that camera's feature scale.

The small instances are what is missed: the green item on a rail (twice), the
pink item hanging from a pole, two items on seats at distance. The person veto
also ate the plastic bag at cam54/08-35-17, which sits right at a staff member's
feet (`-person=5`).

Not labelled in this pass: the other 11 interior cameras, and cam52 at
08-35/08-40/08-59 (unconfirmable distant items, and a passenger case). They are
absent from the file rather than declared clean.

## Extending the dataset

Add cases from the Studio's Benchmark screen, or by hand in
`benchmark/datasets/<id>.json` (paths repo-relative; `reference` is a key
into the `references` map; give every case a `source`). Images of a different
size are uniformly resized onto the reference (fine while the camera framing /
aspect ratio matches). For a new camera: create its reference + a `references`
entry, add same-session AND cross-session empty frames as negatives, and
retune `DIFF_THRESHOLD` / `MIN_AREA` with `eval_localization.py`.
