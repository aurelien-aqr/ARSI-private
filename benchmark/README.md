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

## Files

- `ground_truth.json` — the labelled dataset. **29 cases: 17 anomalous
  (45 typed instance boxes), 12 clean.** Each case: inspection image,
  reference key, `has_anomaly`, `types`, `source` (real / gpt / variant /
  self), instance boxes in reference pixel space, and a human note.
  - Anomalous: 6 real CCTV frames (forgotten objects, one with a real seated
    person) + 10 AI-inpainted frames on the real scene (objects, graffiti,
    damage, litter, one crowd) + 1 second synthetic scene ("variant").
  - Clean: reference-vs-self (×2), a clean AI frame, 4 same-session empty
    frames, and **5 cross-session empty frames** (v2/v3/v4 vs the v1
    reference — the deployment-realistic negatives: different exposure,
    onboard-display content changes, one walking person).
- `run_benchmark.py` — runs the pipeline on every case, writes the report
  after each case (crash-safe), caches VLM verdicts (resumable).
- `eval_localization.py` — **localization-only** scoring (no VLM, seconds per
  run): instance recall + region counts per diff variant. Use it to tune
  thresholds/channels on numbers; variant `shipped` is exactly
  `vlm_05.localize()` and doubles as a regression test after localizer edits.
- `zones_tram_1762.json` — zones-of-interest for the official Task-3 protocol
  (used by `bench_grid.py`, exported from the results spreadsheet).
- `report.md` / `results.json` — latest run (regenerated per case).
- `report_lenient_qwen3vl.md` / `results_lenient_qwen3vl.json` — preserved
  baseline: lenient prompt, single-channel localizer, 24 cases.
- `cache.json` — per-region VLM verdicts keyed by
  (image, reference, box, model, prompt).
- `annotated/<id>.jpg` — blue = ground truth, green = correct detection,
  red = false-positive region.

## How to run

```bash
# from the repository root, with the venv active and `ollama serve` running:
python benchmark/run_benchmark.py        # full benchmark (VLM)
python benchmark/eval_localization.py    # localizer-only check (no VLM)
```

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

## GPU results (2026-07-12, 29 cases, multi-channel localizer, corrected GT)

| judge × prompt | frame F1 | specificity | object recall | region precision | s/call |
|---|---|---|---|---|---|
| qwen3-vl:8b × conservative | 0.829 | 0.417 | 0.978 | 0.355 | 1.1 |
| qwen3-vl:8b × lenient      | 0.872 | 0.583 | 0.978 | 0.534 | 1.1 |
| **qwen3.5:9b × conservative** | **0.919** | **0.750** | **0.978** | **0.549** | **0.7** |
| InternVL3_5:8b × conservative | 0.970¹ | 1.000 | 0.733¹ | 0.759 | 0.8 |

¹ InternVL's frame-level scores flatter it: object recall 0.733 means it
systematically says NO to real phones (2/4 instances on every real multi-object
frame, full-frame FN on real_f0205). Frame recall is 1.000 for both qwen judges
under both prompts.

Findings:
- **The "conservative" prompt backfires on qwen3-vl** — asked to "name a
  specific new object", it actively finds one ("Blue seat cushion slightly
  shifted", "Black cable snake-like") and answers YES: region precision 0.355
  vs 0.534 for the shorter lenient prompt. Prompt A/B must be measured, never
  assumed.
- **qwen3.5:9b is the best judge tested** (its lack of grounding is irrelevant
  here — vlm_05 never asks for coordinates) and the fastest.
- Missing cell to run on GPU: **qwen3.5:9b × lenient** (likely champion).
  GLM-4.6V-Flash and minicpm-v4.6 sweeps aborted on the ":latest" check_model
  bug (fixed) and still need their first real run.
- The preserved `report_lenient_qwen3vl.md` is the OLD baseline (24-case GT,
  single-channel localizer, CPU) — not comparable to the table above.

## Extending the dataset

Add entries to `ground_truth.json` (paths repo-relative; `reference` is a key
into the `references` map; give every case a `source`). Images of a different
size are uniformly resized onto the reference (fine while the camera framing /
aspect ratio matches). For a new camera: create its reference + a `references`
entry, add same-session AND cross-session empty frames as negatives, and
retune `DIFF_THRESHOLD` / `MIN_AREA` with `eval_localization.py`.
